import numpy as np
import argparse
import os
import sys
import math

from scipy.ndimage import gaussian_filter

# SciPy 可选但强烈推荐：KDTree & 统计更快
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Open3D 用于读取点云（可选）
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


# --------------------------
# 基础 I/O
# --------------------------
def load_tsdf_npz(npz_path: str):
    data = np.load(npz_path)
    phi = data["phi"]             # (nx, ny, nz) float32
    grad = data["grad"]           # (3, nx, ny, nz) float32
    origin = data["origin"]       # (3,) float32
    voxel_size = float(data["voxel_size"])
    dims = data["dims"]           # (3,) int32
    meta = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files if k not in ["phi", "grad", "origin", "voxel_size", "dims"]}
    return phi, grad, origin.astype(np.float64), voxel_size, dims.astype(int), meta


def load_point_cloud(pcd_path: str):
    ext = os.path.splitext(pcd_path)[1].lower()
    if not _HAS_O3D:
        raise RuntimeError("未检测到 open3d，请安装后读取点云：pip install open3d")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError("点云为空，请检查路径与文件。")
    return np.asarray(pcd.points, dtype=np.float64)


def grid_coords(origin, voxel_size, dims):
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    xs = origin[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = origin[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = origin[2] + (np.arange(nz) + 0.5) * voxel_size
    return xs, ys, zs


def query_nearest_voxel(phi, grad, origin, voxel_size, x_world):
    """ 最近邻体素查询 ϕ 与 ∇ϕ/||∇ϕ|| """
    rel = (np.asarray(x_world, dtype=np.float64) - origin) / voxel_size
    idx = np.floor(rel - 0.5).astype(int)
    nx, ny, nz = phi.shape
    ix = np.clip(idx[0], 0, nx - 1)
    iy = np.clip(idx[1], 0, ny - 1)
    iz = np.clip(idx[2], 0, nz - 1)
    val = float(phi[ix, iy, iz])
    g = grad[:, ix, iy, iz].astype(np.float64)
    gnorm = np.linalg.norm(g) + 1e-12
    normal = (g / gnorm).ravel()
    return val, normal, g, gnorm


# --------------------------
# 检测 0：基本健康检查
# --------------------------
def basic_health_checks(phi, grad, voxel_size):
    issues = []
    if not np.isfinite(phi).all():
        issues.append("ϕ 中含有 NaN/Inf。")
    if not np.isfinite(grad).all():
        issues.append("∇ϕ 中含有 NaN/Inf。")
    if np.any(np.isnan(phi)) or np.any(np.isnan(grad)):
        issues.append("发现 NaN 值。")
    if np.min(phi) == np.max(phi):
        issues.append("ϕ 全场常数，似乎未正确构建。")
    if voxel_size <= 0:
        issues.append("voxel_size 非法（<=0）。")
    return issues


# --------------------------
# 检测 1：等值面贴合度（需要点云）
# 近似做法：在体素网格中找“符号翻转”的边界体素，收集其中心点作为近似等值面点集 Sϕ，
# 然后计算 Sϕ 到点云 P 的最近距离分布（cKDTree）。
# --------------------------
def gather_zero_crossing_points(phi, origin, voxel_size, max_points=50000):
    nx, ny, nz = phi.shape
    xs = origin[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = origin[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = origin[2] + (np.arange(nz) + 0.5) * voxel_size

    mask = np.zeros_like(phi, dtype=bool)

    # 轴向相邻体素符号是否不同（排除零），有不同则标记两者之一为“近零面体素”
    # x 邻接
    sign_x = phi[:-1,:,:] * phi[1:,:,:]
    mask[:-1,:,:] |= (sign_x < 0)
    mask[1: ,:,:] |= (sign_x < 0)
    # y 邻接
    sign_y = phi[:,:-1,:] * phi[:,1:,:]
    mask[:,:-1,:] |= (sign_y < 0)
    mask[:,1: ,:] |= (sign_y < 0)
    # z 邻接
    sign_z = phi[:,:,:-1] * phi[:,:,1:]
    mask[:,:,:-1] |= (sign_z < 0)
    mask[:,:,1: ] |= (sign_z < 0)

    idxs = np.argwhere(mask)
    if idxs.shape[0] == 0:
        return np.zeros((0,3), dtype=np.float64)

    if idxs.shape[0] > max_points:
        sel = np.random.choice(idxs.shape[0], size=max_points, replace=False)
        idxs = idxs[sel]

    # 体素中心坐标
    pts = np.zeros((idxs.shape[0], 3), dtype=np.float64)
    pts[:,0] = xs[idxs[:,0]]
    pts[:,1] = ys[idxs[:,1]]
    pts[:,2] = zs[idxs[:,2]]
    return pts


def eval_isosurface_fit_to_pointcloud(phi, origin, voxel_size, pcd_points, report_prefix="(1) 0等值面贴合度"):
    S = gather_zero_crossing_points(phi, origin, voxel_size, max_points=80000)
    if S.shape[0] == 0:
        print(f"{report_prefix}：未找到符号翻转体素，可能 ϕ 场没有覆盖到表面或符号均一。")
        return

    if not _HAS_SCIPY:
        print(f"{report_prefix}：需要 SciPy 才能计算最近距离（cKDTree）。跳过此项。")
        return

    tree = cKDTree(pcd_points)
    dists, _ = tree.query(S, k=1, workers=-1)
    d_med = float(np.median(dists))
    d_p95 = float(np.percentile(dists, 95))
    d_max = float(np.max(dists))
    cov = dists < (2.5 * voxel_size)  # “贴合度阈值”建议 ~ 2-3 个体素
    coverage = float(np.mean(cov)) * 100.0

    print(f"{report_prefix}：")
    print(f"  - 等值面采样点数：{S.shape[0]}")
    print(f"  - 到点云最近距离：median={d_med:.4f} m, p95={d_p95:.4f} m, max={d_max:.4f} m")
    print(f"  - 覆盖率(ϕ≈0 点落在 {2.5}x 体素内)：{coverage:.1f}%")
    # 结论
    if d_med < 2.0 * voxel_size and coverage > 85.0:
        print("  → 结论：等值面对原始点云的贴合度 **良好**。")
    else:
        print("  → 警示：贴合度一般/偏差较大。考虑：减小 voxel_size、增大 trunc_margin、加 smooth 或改法向一致化。")


# --------------------------
# 检测 2：符号正确性（需要点云）
# 抽样点云上的点 p，并沿法向外侧与内侧各偏移 δ，检查 ϕ(p±δ n̂) 的符号。
# 若点云没有法向，可用球面微扰 + 最近值代替（近似）。
# --------------------------
def eval_sign_correctness(phi, grad, origin, voxel_size, pcd_points, pcd_normals=None, samples=2000, delta=0.01, report_prefix="(2) 符号正确性"):
    if not _HAS_SCIPY:
        print(f"{report_prefix}：需要 SciPy（KDTree）进行采样与近邻。跳过此项。")
        return

    # 若无点云法向：用 TSDF 梯度在最近体素处作为近似法向
    tree = cKDTree(pcd_points)
    idxs = np.random.choice(pcd_points.shape[0], size=min(samples, pcd_points.shape[0]), replace=False)
    p_sel = pcd_points[idxs]

    if pcd_normals is None:
        n_list = []
        for p in p_sel:
            _, _, g, gnorm = query_nearest_voxel(phi, grad, origin, voxel_size, p)
            n = g / (gnorm + 1e-12)
            n_list.append(n)
        n_sel = np.asarray(n_list)
    else:
        n_sel = pcd_normals[idxs]
        # 归一化
        n_sel = n_sel / (np.linalg.norm(n_sel, axis=1, keepdims=True) + 1e-12)

    pos = p_sel + delta * n_sel  # 外部
    neg = p_sel - delta * n_sel  # 内部

    pos_vals = []
    neg_vals = []
    for a, b in zip(pos, neg):
        va, _, _, _ = query_nearest_voxel(phi, grad, origin, voxel_size, a)
        vb, _, _, _ = query_nearest_voxel(phi, grad, origin, voxel_size, b)
        pos_vals.append(va)
        neg_vals.append(vb)
    pos_vals = np.asarray(pos_vals)
    neg_vals = np.asarray(neg_vals)

    rate_pos = float(np.mean(pos_vals > 0.0)) * 100.0
    rate_neg = float(np.mean(neg_vals < 0.0)) * 100.0
    print(f"{report_prefix}：")
    print(f"  - 外侧抽检（+δ）为正比例：{rate_pos:.1f}%")
    print(f"  - 内侧抽检（-δ）为负比例：{rate_neg:.1f}%")

    if rate_pos > 92.0 and rate_neg > 92.0:
        print("  → 结论：符号方向 **正确**（外正内负）。")
    else:
        print("  → 警示：符号有明显错误或法向一致化失败。建议：在构建 TSDF 时指定 --orient_camera，或最后统一 phi,grad 取反。")


# --------------------------
# 检测 3：法向连续性（表面附近，统计相邻样点夹角）
# --------------------------
def eval_normal_continuity(phi, grad, origin, voxel_size, band=0.01, samples=20000, report_prefix="(3) 法向连续性"):
    nx, ny, nz = phi.shape
    xs, ys, zs = grid_coords(origin, voxel_size, (nx, ny, nz))

    # 抽样 band 内的体素（|ϕ| < band）
    mask = np.abs(phi) < band
    idxs = np.argwhere(mask)
    if idxs.shape[0] == 0:
        print(f"{report_prefix}：在 |ϕ|<{band} m 的窄带内未找到体素。可适当增大 band。")
        return
    if idxs.shape[0] > samples:
        sel = np.random.choice(idxs.shape[0], size=samples, replace=False)
        idxs = idxs[sel]

    # 取法向
    G = grad[:, idxs[:,0], idxs[:,1], idxs[:,2]].T  # (M, 3)
    G_norm = np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    N = G / G_norm

    # 和近邻的夹角：用体素 6 邻域近似
    # 这里简单实现：随机挑一半样本，和“偏移一格”的点做内积
    half = idxs.shape[0] // 2
    a = N[:half]
    b = N[1:half+1]
    cosang = np.sum(a * b, axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))  # 相邻法向夹角（度）

    med = float(np.median(ang))
    p95 = float(np.percentile(ang, 95))
    print(f"{report_prefix}：")
    print(f"  - 相邻法向夹角分布：median={med:.2f}°, p95={p95:.2f}°")
    if med < 8.0 and p95 < 25.0:
        print("  → 结论：法向在表面附近 **较为连续稳定**。")
    else:
        print("  → 警示：法向波动偏大，可能导致控制时抖动。建议：增大 smooth、trunc_margin 或增大法向估计半径。")


# --------------------------
# 检测 4：梯度幅值与连续性
# --------------------------
def eval_gradient_magnitude(grad, report_prefix="(4) 梯度幅值与连续性"):
    gx, gy, gz = grad[0], grad[1], grad[2]
    gnorm = np.sqrt(gx*gx + gy*gy + gz*gz) + 1e-12

    stats = {
        "min": float(np.min(gnorm)),
        "max": float(np.max(gnorm)),
        "median": float(np.median(gnorm)),
        "p95": float(np.percentile(gnorm, 95)),
    }
    # 粗略离群：小于 1e-6 或大于 1/voxel_size 的 3 倍（此处不知 voxel_size，保守不设上限）
    near_zero_ratio = float(np.mean(gnorm < 1e-6)) * 100.0

    print(f"{report_prefix}：")
    print(f"  - ||∇ϕ|| 统计：min={stats['min']:.3e}, median={stats['median']:.3e}, p95={stats['p95']:.3e}, max={stats['max']:.3e}")
    print(f"  - 近零梯度占比(<1e-6)：{near_zero_ratio:.2f}%")
    if near_zero_ratio < 5.0:
        print("  → 结论：梯度幅值分布 **正常**。")
    else:
        print("  → 警示：大量近零梯度，可能导致法向不稳定或等值面模糊。建议：减小 voxel_size、增加 smooth 或改进 TSDF 估计。")


# --------------------------
# 检测 5：单调性/震荡（随机射线抽检）
# 在若干随机射线方向上，记录 ϕ 值随步进的变化，统计“非单调段比例”和“符号翻转次数”。
# --------------------------
def eval_monotonicity(phi, origin, voxel_size, n_rays=64, steps=120, step_m=0.004, report_prefix="(5) 单调性/震荡"):
    nx, ny, nz = phi.shape
    dims = np.array([nx, ny, nz], dtype=int)

    def in_bounds(x):
        rel = (x - origin) / voxel_size - 0.5
        idx = np.floor(rel).astype(int)
        return np.all((idx >= 0) & (idx < dims))

    def nn_phi(x):
        rel = (x - origin) / voxel_size - 0.5
        idx = np.floor(rel).astype(int)
        ix, iy, iz = np.clip(idx, 0, dims - 1)
        return float(phi[ix, iy, iz])

    # 从体素中心随机选若干起点 & 随机方向
    rng = np.random.default_rng(12345)
    start_idx = np.column_stack([
        rng.integers(0, nx, size=n_rays),
        rng.integers(0, ny, size=n_rays),
        rng.integers(0, nz, size=n_rays),
    ])
    # 随机方向（单位向量）
    dirs = rng.normal(size=(n_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    nonmono_rates = []
    sign_flip_counts = []

    for k in range(n_rays):
        ix, iy, iz = start_idx[k]
        x0 = origin + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * voxel_size
        d = dirs[k]

        vals = []
        x = x0.copy()
        for s in range(steps):
            if not in_bounds(x):
                break
            vals.append(nn_phi(x))
            x += step_m * d

        vals = np.asarray(vals, dtype=np.float64)
        if vals.size < 5:
            continue

        # 局部单调性：统计 sign(diff) 的变化次数与波动比例
        dif = np.diff(vals)
        sgn = np.sign(dif)
        flips = np.sum(sgn[1:] * sgn[:-1] < 0)   # 一维导数的符号翻转次数
        sign_flip_counts.append(int(flips))

        # 简单“非单调率”：导数符号与首段保持一致的比例
        if len(sgn) > 0:
            main = 1.0 if sgn[0] >= 0 else -1.0
            non_mono = np.mean(sgn * main < 0) * 100.0
            nonmono_rates.append(float(non_mono))

    if len(nonmono_rates) == 0:
        print(f"{report_prefix}：射线样本不足（可能越界过早）。请增大 grid_pad 或减少 step_m。")
        return

    med_nonmono = float(np.median(nonmono_rates))
    p95_nonmono = float(np.percentile(nonmono_rates, 95))
    med_flips = float(np.median(sign_flip_counts))
    p95_flips = float(np.percentile(sign_flip_counts, 95))

    print(f"{report_prefix}：")
    print(f"  - 非单调率（%）：median={med_nonmono:.2f}, p95={p95_nonmono:.2f}")
    print(f"  - 导数符号翻转次数：median={med_flips:.1f}, p95={p95_flips:.1f}")
    if med_nonmono < 10.0 and p95_flips <= 3.0:
        print("  → 结论：ϕ 场沿随机方向整体 **平滑单调**，未见明显震荡。")
    else:
        print("  → 警示：存在较多非单调/震荡，建议：增大 trunc_margin、适度 smooth，或改用更稳健的局部平面估计。")


# --------------------------
# 主流程
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="TSDF 质量检查（无 GUI 版）")
    ap.add_argument("--npz", required=True, help="TSDF 网格 .npz（build_tsdf_from_pointcloud.py 的输出）")
    ap.add_argument("--pcd", default=None, help="原始点云（.ply/.pcd/.xyz/.obj 等）可选，用于几何/符号更完整的检查")
    ap.add_argument("--band", type=float, default=0.01, help="表面附近带宽 |ϕ|<band（m），默认 1 cm）")
    ap.add_argument("--samples", type=int, default=20000, help="抽样上限（法向与等值面点）")
    ap.add_argument("--delta", type=float, default=0.01, help="符号抽检偏移量 δ（m），默认 1 cm）")
    args = ap.parse_args()

    print("[I] 加载 TSDF：", args.npz)
    phi, grad, origin, voxel_size, dims, meta = load_tsdf_npz(args.npz)
    print(f"[I] 网格信息：dims={tuple(dims)}, voxel={voxel_size:.4f} m, origin={origin.tolist()}")
    if meta:
        print("[I] 其他元数据：", meta)

    # 0) 基础健康检查
    issues = basic_health_checks(phi, grad, voxel_size)
    print("\n(0) 基础健康检查：")
    if not issues:
        print("  → OK：未发现明显格式/数值问题。")
    else:
        for it in issues:
            print("  -", it)
        print("  → 请先修复上述问题，再继续其他检测。")
        if len(issues) > 0:
            # 不中断，继续给出其他信息作为参考
            pass

    # 1) 等值面贴合度（若提供点云）
    if args.pcd is not None:
        print("\n[I] 加载点云：", args.pcd)
        try:
            pts = load_point_cloud(args.pcd)
        except Exception as e:
            print("  加载点云失败：", e)
            pts = None
        if pts is not None:
            eval_isosurface_fit_to_pointcloud(phi, origin, voxel_size, pts, report_prefix="(1) ϕ=0 等值面贴合度")
            # 2) 符号正确性（基于点云法向/或由 TSDF 近似法向替代）
            eval_sign_correctness(phi, grad, origin, voxel_size, pts, pcd_normals=None,
                                  samples=min(args.samples, 4000), delta=args.delta,
                                  report_prefix="(2) 符号正确性（外正内负）")
    else:
        print("\n(1)(2) 跳过：未提供 --pcd，无法做“等值面贴合度/符号正确性”对点云的验证。")

    # 3) 法向连续性（表面附近）
    eval_normal_continuity(phi, grad, origin, voxel_size, band=args.band,
                           samples=args.samples, report_prefix="(3) 法向连续性（表面附近）")

    # 4) 梯度幅值与连续性
    eval_gradient_magnitude(grad, report_prefix="(4) 梯度幅值与连续性")

    # 5) 单调性/震荡（随机射线抽检）
    eval_monotonicity(phi, origin, voxel_size, n_rays=64, steps=120, step_m=max(0.004, 2*voxel_size),
                      report_prefix="(5) 单调性/震荡")

    print("\n[总结建议]")
    print("  - 若(1)贴合度差：减小 voxel_size、增大 trunc_margin 或加 smooth；")
    print("  - 若(2)符号异常：构建时设置 --orient_camera，或统一 phi/grad 取反；")
    print("  - 若(3)法向不稳：增大 smooth、normal_radius，或改良局部平面估计；")
    print("  - 若(4)梯度近零多：提高点云质量、减小体素、平滑；")
    print("  - 若(5)震荡明显：增大 trunc_margin、平滑，并在控制器端对 d,n 做低通 + 限幅。")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("用法示例：")
        print("  python test_tsdf.py --npz tsdf_grid.npz --pcd ManualFilteredPointCloud.ply")
        print("可用参数：--band 0.01 --samples 20000 --delta 0.01")
    else:
        main()
import argparse
import os
import sys
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter

_HAS_SCIPY = True

def load_point_cloud(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".obj", ".stl"]:
        pcd = o3d.io.read_point_cloud(path)
    else:
        raise ValueError(f"不支持的点云格式：{ext}")
    if pcd.is_empty():
        raise RuntimeError("点云为空，请检查输入文件。")
    return pcd


def preprocess_point_cloud(
    pcd,
    voxel_downsample,
    nb_neighbors,
    std_ratio,
    normal_radius,
    normal_max_nn,
    orient_consistent_k,
    orient_camera,
):
    """
    点云预处理：
    1. 体素下采样
    2. 统计滤波去掉离群点
    3. 估计法向
    4. 法向朝外一致化
    5. 法向切平面一致化（全局平滑）
    """

    # 1) 体素下采样（如果 voxel_downsample>0）
    if voxel_downsample is not None and voxel_downsample > 0:
        pcd_proc = pcd.voxel_down_sample(voxel_downsample)
    else:
        pcd_proc = pcd.clone()

    # 2) 统计滤波 (去除孤立点 / 噪声)
    #    注意：StatisticalOutlierRemoval 返回的是 (filtered_pcd, indices)
    try:
        pcd_proc, _ = pcd_proc.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    except Exception as e:
        print(f"[WARN] 统计滤波失败，继续不滤波: {e}")

    # 3) 估计法向
    pcd_proc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )

    # 4) 统一法向朝向
    # orient_camera 可以是：
    #   - True: 我们自动猜一个“相机位置”
    #   - False/None: 不做这个步骤
    #   - [x,y,z] / np.array(3,): 用这个坐标当相机位置
    if orient_camera is not None and orient_camera is not False:
        if orient_camera is True:
            # 自动猜相机：点云质心往 +Z 外推 20 cm
            pts_np = np.asarray(pcd_proc.points)
            center = pts_np.mean(axis=0)
            cam_loc = center + np.array([0.0, 0.0, 0.20], dtype=np.float64)
        else:
            cam_loc = np.asarray(orient_camera, dtype=np.float64)

        try:
            pcd_proc.orient_normals_towards_camera_location(cam_loc)
        except TypeError as e:
            # 某些 open3d 版本要求 cam_loc 形状是 (3,) float64
            print(f"[WARN] orient_normals_towards_camera_location 形状不兼容: {e}")
            cam_loc = cam_loc.reshape(3,)
            pcd_proc.orient_normals_towards_camera_location(cam_loc)

    # 5) 全局法向一致化（切平面一致化 / smoothing）
    # 有的 open3d 版本没有这个API，或者 k 太大可能报错，所以 try/except
    if orient_consistent_k is not None and orient_consistent_k > 0:
        try:
            pcd_proc.orient_normals_consistent_tangent_plane(k=int(orient_consistent_k))
        except Exception as e:
            print(f"[WARN] orient_normals_consistent_tangent_plane 失败，跳过: {e}")

    return pcd_proc


def build_grid_from_bounds(bounds_min, bounds_max, voxel_size: float, pad: float = 0.0):
    bmin = np.array(bounds_min, dtype=np.float64) - pad
    bmax = np.array(bounds_max, dtype=np.float64) + pad
    dims = np.ceil((bmax - bmin) / voxel_size).astype(int)
    bmax = bmin + dims * voxel_size  # 对齐
    return bmin, bmax, dims


def compute_tsdf_from_point_cloud(pcd: o3d.geometry.PointCloud,
                                  voxel_size: float,
                                  trunc_margin: float,
                                  grid_pad: float = 0.0,
                                  smooth_iters: int = 0,
                                  smooth_sigma_vox: float = 0.75,
                                  progress: bool = True):
    """
    基于最近邻 + 局部切平面的带符号距离近似（TSDF）。
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    nors = np.asarray(pcd.normals, dtype=np.float64)
    if nors.shape != pts.shape:
        raise RuntimeError("点云尚未包含法向，请检查预处理。")

    # 构建 KDTree（Open3D 或 Numpy + scipy.spatial 均可）
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 计算体素网格边界
    bounds_min = pts.min(axis=0)
    bounds_max = pts.max(axis=0)
    origin, bmax, dims = build_grid_from_bounds(bounds_min, bounds_max, voxel_size, pad=grid_pad)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

    # 网格中心坐标
    xs = origin[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = origin[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = origin[2] + (np.arange(nz) + 0.5) * voxel_size

    phi = np.zeros((nx, ny, nz), dtype=np.float32)

    t0 = time.time()
    for ix, x in enumerate(xs):
        if progress and ix % max(1, nx // 20) == 0:
            print(f"[TSDF] {ix}/{nx} ({ix / nx:.0%})")
        for iy, y in enumerate(ys):
            # 以向量化方式处理 z 维度
            xyz = np.stack([np.full_like(zs, x),
                            np.full_like(zs, y),
                            zs], axis=1)  # (nz, 3)

            # 最近邻：对每个体素中心，找最近点与法向
            # 这里逐个查询（为稳定性与内存考虑），也可批量加速（需自写 KDTree 批量接口）
            d_vals = np.empty((nz,), dtype=np.float32)
            for iz in range(nz):
                _, idx, _ = pcd_tree.search_knn_vector_3d(xyz[iz], 1)
                i0 = idx[0]
                p = pts[i0]
                n = nors[i0]
                # 局部切平面符号距离
                d_plane = np.dot((xyz[iz] - p), n)
                # 截断 TSDF
                if d_plane > trunc_margin:
                    d_vals[iz] = trunc_margin
                elif d_plane < -trunc_margin:
                    d_vals[iz] = -trunc_margin
                else:
                    d_vals[iz] = d_plane

            phi[ix, iy, :] = d_vals

    # 平滑（可选）
    if smooth_iters > 0:
        if not _HAS_SCIPY:
            print("[WARN] 未安装 SciPy，无法进行高斯平滑。可 pip install scipy 后重试。")
        else:
            sigma = smooth_sigma_vox
            for _ in range(smooth_iters):
                phi = gaussian_filter(phi, sigma=sigma, mode="nearest")

    # 数值梯度（中心差分），得到 ∇ϕ
    # 注意：np.gradient 接受体素间距（米），可以传 voxel_size
    gx, gy, gz = np.gradient(phi, voxel_size, voxel_size, voxel_size, edge_order=2)
    grad = np.stack([gx, gy, gz], axis=0).astype(np.float32)  # (3, nx, ny, nz)

    meta = {
        "origin": origin.astype(np.float32),
        "voxel_size": float(voxel_size),
        "dims": np.array([nx, ny, nz], dtype=np.int32),
        "trunc_margin": float(trunc_margin),
        "grid_pad": float(grid_pad),
    }
    elapsed = time.time() - t0
    return phi, grad, meta, elapsed


def save_tsdf(path_out_npz: str, phi: np.ndarray, grad: np.ndarray, meta: dict):
    np.savez_compressed(path_out_npz, phi=phi, grad=grad, **meta)


def main():
    parser = argparse.ArgumentParser(description="从点云直接构建 TSDF/ESDF 网格（供阻抗控制查询 d 与 n）")
    parser.add_argument("--input", type=str, required=True, help="输入点云路径（.ply/.pcd/.obj 等）")
    parser.add_argument("--output", type=str, default="tsdf_grid.npz", help="输出 npz")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="体素大小（米），默认 5mm")
    parser.add_argument("--trunc_margin", type=float, default=0.01, help="TSDF 截断距离 μ（米），默认 10mm")
    parser.add_argument("--grid_pad", type=float, default=0.02, help="边界外扩（米），默认 20mm")
    parser.add_argument("--downsample_voxel", type=float, default=0.003, help="点云下采样体素（米）")
    parser.add_argument("--nb_neighbors", type=int, default=24, help="统计滤波邻居点数")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="统计滤波标准差阈值")
    parser.add_argument("--normal_radius", type=float, default=0.01, help="法向估计半径（米）")
    parser.add_argument("--normal_max_nn", type=int, default=50, help="法向估计最大邻居数")
    parser.add_argument("--orient_k", type=int, default=50, help="法向一致化 k（切平面一致化）")
    parser.add_argument("--orient_camera", type=float, nargs=3, default=None, help="若已知相机位置，可用该向量进行法向朝向一致化")
    parser.add_argument("--smooth", type=int, default=0, help="高斯平滑迭代次数（需 scipy）")
    parser.add_argument("--smooth_sigma_vox", type=float, default=0.75, help="平滑 sigma（单位：体素）")
    args = parser.parse_args()

    print("[I] 载入点云：", args.input)
    pcd = load_point_cloud(args.input)

    print("[I] 预处理/下采样/去噪/法向估计 ...")
    pcdp = preprocess_point_cloud(
        pcd,
        downsample_voxel=args.downsample_voxel,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        orient_consistent_k=args.orient_k,
        orient_camera=list(args.orient_camera) if args.orient_camera is not None else None
    )
    print(pcdp)

    print("[I] 构建 TSDF 网格 ...")
    phi, grad, meta, elapsed = compute_tsdf_from_point_cloud(
        pcdp,
        voxel_size=args.voxel_size,
        trunc_margin=args.trunc_margin,
        grid_pad=args.grid_pad,
        smooth_iters=args.smooth,
        smooth_sigma_vox=args.smooth_sigma_vox,
        progress=True
    )
    print(f"[I] TSDF 完成，用时 {elapsed:.2f}s，grid dims={meta['dims']} voxel_size={meta['voxel_size']}")

    print("[I] 保存 npz：", args.output)
    save_tsdf(args.output, phi, grad, meta)
    print("[OK] 完成。")


# ———— 可选：查询接口示例（供控制器调用） ————
def load_tsdf_npz(npz_path: str):
    data = np.load(npz_path)
    phi = data["phi"]
    grad = data["grad"]
    origin = data["origin"]
    voxel_size = float(data["voxel_size"])
    dims = data["dims"]
    return phi, grad, origin, voxel_size, dims


def query_tsdf(phi, grad, origin, voxel_size, x_world):
    """
    最近邻/三线性插值查询 ϕ 与 ∇ϕ。
    这里用最近邻；若需要更平滑，可改为三线性（自己实现或用 griddata）。
    """
    rel = (np.asarray(x_world, dtype=np.float32) - origin) / voxel_size
    idx = np.floor(rel - 0.5).astype(int)  # 对齐中心
    nx, ny, nz = phi.shape
    ix = np.clip(idx[0], 0, nx - 1)
    iy = np.clip(idx[1], 0, ny - 1)
    iz = np.clip(idx[2], 0, nz - 1)
    val = float(phi[ix, iy, iz])
    g = grad[:, ix, iy, iz].astype(np.float32)
    gnorm = np.linalg.norm(g) + 1e-8
    normal = (g / gnorm).ravel()
    return val, normal

PARAMS = dict(
    input_path="ManualFilteredPointCloud.ply",  # 输入点云
    output_path="tsdf_grid.npz",                # 输出 npz
    voxel_size=0.005,                           # 体素米
    trunc_margin=0.020,                          # 截断距离米
    grid_pad=0.02,                              # 外扩边界米
    voxel_downsample=0.003,                     # 下采样米（0 关闭）
    nb_neighbors=24,
    std_ratio=2.0,
    normal_radius=0.04,
    normal_max_nn=100,
    orient_k=50,
    orient_camera=True,                         # 例如 [0,0,0]；None 用一致切平面
    smooth=1,                                   # 高斯平滑迭代（需 SciPy）
    smooth_sigma_vox=0.5,                      # 以体素为单位
    verbose=False                               # True 才会打印进度与信息
)

def _run_from_params(p):
    # 载入与预处理
    pcd = load_point_cloud(p["input_path"])
    pcdp = preprocess_point_cloud(
        pcd,
        voxel_downsample=p["voxel_downsample"],
        nb_neighbors=p["nb_neighbors"],
        std_ratio=p["std_ratio"],
        normal_radius=p["normal_radius"],
        normal_max_nn=p["normal_max_nn"],
        orient_consistent_k=p["orient_k"],
        orient_camera=p["orient_camera"],
    )
    # 构建 TSDF
    phi, grad, meta, elapsed = compute_tsdf_from_point_cloud(
        pcdp,
        voxel_size=p["voxel_size"],
        trunc_margin=p["trunc_margin"],
        grid_pad=p["grid_pad"],
        smooth_iters=p["smooth"],
        smooth_sigma_vox=p["smooth_sigma_vox"],
        progress=p["verbose"]
    )
    # 保存
    save_tsdf(p["output_path"], phi, grad, meta)

    # 可选：做一个查询验证（不打印）
    x_test = meta["origin"] + np.array([0.02, 0.02, 0.02], dtype=np.float32)
    val, nrm = query_tsdf(phi, grad, meta["origin"], meta["voxel_size"], x_test)

    # 仅在 verbose=True 时打印
    if p["verbose"]:
        print(f"ϕ={val:.4f}, n={nrm}, dims={meta['dims']}, time={elapsed:.2f}s, out={p['output_path']}")

    return val, nrm, meta, elapsed

if __name__ == "__main__":
    _run_from_params(PARAMS)



"""
Point Cloud -> Cleaned Surface Mesh Pipeline (Open3D)
Usage:
  python read_point_cloud.py --in ManualFilteredPointCloud.ply --out out_dir
Requirements: pip install open3d
"""

import os
import argparse
import numpy as np
import open3d as o3d

def size_of(path):
    try:
        return os.path.getsize(path) / (1024*1024)
    except:
        return None

def save_pcd(pcd, path, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd, print_progress=True)
    np_pts = np.asarray(pcd.points)
    bbox = np.array(pcd.get_axis_aligned_bounding_box().get_box_points())
    bbmin, bbmax = bbox.min(axis=0), bbox.max(axis=0)
    return {
        "file": path,
        "type": "PointCloud",
        "points": int(np_pts.shape[0]),
        "faces": None,
        "bbox_min": bbmin.tolist(),
        "bbox_max": bbmax.tolist(),
        "size_MB": round(size_of(path), 3),
        "note": note
    }

def save_mesh(mesh, path, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_triangle_mesh(path, mesh, print_progress=True)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    bbox = np.array(mesh.get_axis_aligned_bounding_box().get_box_points())
    bbmin, bbmax = bbox.min(axis=0), bbox.max(axis=0)
    return {
        "file": path,
        "type": "TriangleMesh",
        "points": int(V.shape[0]),
        "faces": int(F.shape[0]),
        "bbox_min": bbmin.tolist(),
        "bbox_max": bbmax.tolist(),
        "size_MB": round(size_of(path), 3),
        "note": note
    }

def strip_large_planes(pcd, dist_th, min_ratio=0.02, max_rounds=5, ransac_n=3, num_iter=1000):
    """反复 RANSAC 剥离占比较大的平面（地板/墙面/板子等）"""
    total = len(pcd.points)
    remain = pcd
    removed_counts = []
    for _ in range(max_rounds):
        plane_model, inliers = remain.segment_plane(distance_threshold=dist_th,
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iter)
        if len(inliers) / total < min_ratio:
            break
        remain = remain.select_by_index(inliers, invert=True)
        removed_counts.append(len(inliers))
    return remain, removed_counts

def estimate_spacing(pcd):
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else 0.005

def main(args):
    in_path = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    report = []

    # 0) 读取原始点云
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    pcd_raw = o3d.io.read_point_cloud(in_path)
    report.append(save_pcd(pcd_raw, os.path.join(out_dir, "00_raw_pointcloud.ply"), "raw"))

    # 1) 统计离群点去除
    pcd_in, _ = pcd_raw.remove_statistical_outlier(
        nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio
    )
    report.append(save_pcd(pcd_in, os.path.join(out_dir, "01_statistical_inliers.ply"),
                           f"nb_neighbors={args.nb_neighbors}, std_ratio={args.std_ratio}"))

    # 2) 体素下采样
    pcd_ds = pcd_in.voxel_down_sample(voxel_size=args.voxel_size)
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, f"02_voxel_down_{args.voxel_size*1000:.0f}mm.ply"),
                           f"voxel_size={args.voxel_size} m"))

    # 3) 法向估计与一致化（局部一致）+ 朝向相机（单侧扫描很关键）
    pcd_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size*4.0, max_nn=args.max_nn))
    pcd_ds.orient_normals_consistent_tangent_plane(args.orient_k)
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, "03_normals_estimated.ply"),
                           f"radius~{args.voxel_size*4.0:.4f}, max_nn={args.max_nn}, orient_k={args.orient_k}"))

    # 让法向整体朝向“相机/观察点”（可按数据实际调整相机位置）
    pcd_ds.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 1.0))
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, "03c_normals_facing_camera.ply"),
                           "normals toward camera (0,0,1)"))

    # 3b) 剥离大平面
    pcd_stripped, removed = strip_large_planes(
        pcd_ds, dist_th=args.voxel_size * 1.2, min_ratio=0.02, max_rounds=5
    )
    print(f"[Plane strip] removed planes (points each): {removed}")
    report.append(save_pcd(pcd_stripped, os.path.join(out_dir, "03b_plane_stripped.ply"),
                           f"dist_th~{args.voxel_size*1.2:.4f}"))

    # 4) Alpha-Shape 重建（开放表面首选）
    spacing = estimate_spacing(pcd_stripped)
    # alphas_mul = [1.2, 1.6, 2.0, 2.5]   # 可改为 [2.0, 2.5, 3.0, 3.5] 处理更稀疏的条带
    alphas_mul = [2.0, 2.5, 3.0, 12.0]
    alpha_vals = [a * spacing for a in alphas_mul]
    print(f"[AlphaShape] median nn-spacing ~ {spacing:.6f} m; try alphas = {alpha_vals}")

    alpha_meshes = []
    for a_mul, alpha in zip(alphas_mul, alpha_vals):
        mesh_a = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_stripped, alpha)
        mesh_a.remove_degenerate_triangles()
        mesh_a.remove_duplicated_triangles()
        mesh_a.remove_duplicated_vertices()
        mesh_a.remove_non_manifold_edges()
        mesh_a.remove_unreferenced_vertices()
        mesh_a.compute_vertex_normals()
        fn = os.path.join(out_dir, f"04_alpha_{a_mul:.1f}.ply")
        alpha_meshes.append((a_mul, mesh_a))
        report.append(save_mesh(mesh_a, fn, note=f"alpha={alpha:.6f}"))

    # 选三角形数最多的一版（通常连贯性最好）
    counts = [np.asarray(m.triangles).shape[0] for _, m in alpha_meshes]
    best_idx = int(np.argmax(counts))
    a_best, mesh_best = alpha_meshes[best_idx]
    print(f"[AlphaShape] pick alpha multiplier = {a_best:.1f}")

    # 命名为 mesh_trim 以兼容后续流程
    mesh_trim = mesh_best
    report.append(save_mesh(mesh_trim, os.path.join(out_dir, "05_alpha_best.ply"),
                            note=f"picked_alpha_mul={a_best:.1f}"))

    # 5) Taubin 平滑（保体积）
    mesh_smooth = mesh_trim.filter_smooth_taubin(number_of_iterations=args.taubin_iters)
    mesh_smooth.compute_vertex_normals()
    report.append(save_mesh(mesh_smooth, os.path.join(out_dir, f"06_taubin_{args.taubin_iters}it.ply"),
                            f"taubin_iters={args.taubin_iters}"))

    # 6) 仅保留最大连通组件（去小碎片）
    triangle_clusters, cluster_n_tris, _ = mesh_smooth.cluster_connected_triangles()
    cluster_n_tris = np.asarray(cluster_n_tris)
    keep_cluster = int(cluster_n_tris.argmax())
    tri_idx = np.where(np.asarray(triangle_clusters) == keep_cluster)[0].astype(int).tolist()
    mesh_main = mesh_smooth.select_by_index(tri_idx)
    mesh_main.remove_unreferenced_vertices()
    mesh_main.compute_vertex_normals()
    report.append(save_mesh(mesh_main, os.path.join(out_dir, "07_largest_component.ply"),
                            "largest component"))

    # 7) 多分辨率简化
    def decimate_and_save(mesh, target_tris, tag):
        target_tris = int(min(max(target_tris, 1000), np.asarray(mesh.triangles).shape[0]-1))
        dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        dec.remove_degenerate_triangles()
        dec.remove_duplicated_triangles()
        dec.remove_duplicated_vertices()
        dec.remove_non_manifold_edges()
        dec.compute_vertex_normals()
        return save_mesh(dec, os.path.join(out_dir, f"08_decimated_{tag}.ply"),
                         f"target_tris={target_tris}")

    nF = np.asarray(mesh_main.triangles).shape[0]
    targets = [min(100000, nF), min(50000, nF), min(20000, nF)]
    tags = [f"{t//1000}k" for t in targets]
    for t, z in zip(targets, tags):
        report.append(decimate_and_save(mesh_main, t, z))

    # 8) 质量报告 CSV
    import csv
    csv_path = os.path.join(out_dir, "report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        for r in report:
            writer.writerow(r)

    print("\n=== Processing Summary ===")
    for r in report:
        print(f"{r['type']:12s}  pts={str(r['points']):>8s}  faces={str(r['faces']):>8s}  file={r['file']}  size={r['size_MB']}MB  note={r['note']}")
    print(f"\nCSV report: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=str,
                        default="ManualFilteredPointCloud.ply",
                        help="Input PLY point cloud")
    parser.add_argument("--out", dest="output", type=str, default="output_mesh",
                        help="Output directory")
    # 可调参数
    parser.add_argument("--nb_neighbors", type=int, default=20)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--voxel_size", type=float, default=0.003, help="meters")
    parser.add_argument("--max_nn", type=int, default=50)
    parser.add_argument("--orient_k", type=int, default=30)
    # 保留以下参数以兼容旧命令行（本脚本不再使用它们）
    parser.add_argument("--poisson_depth", type=int, default=10)
    parser.add_argument("--density_keep_percentile", type=float, default=5.0,
                        help="(unused in Alpha-Shape path)")
    parser.add_argument("--taubin_iters", type=int, default=10)
    args = parser.parse_args()
    main(args)

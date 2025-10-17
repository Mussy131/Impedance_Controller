"""
Point Cloud -> Cleaned Surface Mesh Pipeline (Open3D)
Author: You
Usage:
  python pc_to_mesh_pipeline.py --in ManualFilteredPointCloud.ply --out out_dir

Requirements:
  pip install open3d

Tips:
  - 若模型尺度很大/很小，调 voxel_size 与 poisson_depth。
  - 若噪声较多：调小 std_ratio（更严格），或提高密度裁剪阈值百分位。
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

    # 3) 法向估计与一致化
    pcd_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size*4.0, max_nn=args.max_nn))
    pcd_ds.orient_normals_consistent_tangent_plane(args.orient_k)
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, "03_normals_estimated.ply"),
                           f"radius~{args.voxel_size*4.0:.4f}, max_nn={args.max_nn}, orient_k={args.orient_k}"))

    # 4) Poisson 重建
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_ds, depth=args.poisson_depth)
    mesh_poisson.compute_vertex_normals()
    report.append(save_mesh(mesh_poisson, os.path.join(out_dir, f"04_poisson_d{args.poisson_depth}.ply"),
                            f"poisson_depth={args.poisson_depth}"))

    # 5) 密度裁剪（去掉支撑不足的薄片/外壳）
    dens_np = np.asarray(densities)
    thr = np.percentile(dens_np, args.density_keep_percentile)
    keep = dens_np > thr
    mesh_trim = mesh_poisson.select_by_index(np.where(keep)[0])
    mesh_trim.remove_unreferenced_vertices()
    mesh_trim.remove_degenerate_triangles()
    mesh_trim.remove_duplicated_triangles()
    mesh_trim.remove_duplicated_vertices()
    mesh_trim.remove_non_manifold_edges()
    mesh_trim.compute_vertex_normals()
    report.append(save_mesh(mesh_trim, os.path.join(out_dir, f"05_poisson_trim_{int(args.density_keep_percentile)}p.ply"),
                            f"density_keep_percentile={args.density_keep_percentile}"))

    # 6) Taubin 平滑（保体积）
    mesh_smooth = mesh_trim.filter_smooth_taubin(number_of_iterations=args.taubin_iters)
    mesh_smooth.compute_vertex_normals()
    report.append(save_mesh(mesh_smooth, os.path.join(out_dir, f"06_taubin_{args.taubin_iters}it.ply"),
                            f"taubin_iters={args.taubin_iters}"))

    # 7) 仅保留最大连通组件（去小碎片）
    triangle_clusters, cluster_n_tris, _ = mesh_smooth.cluster_connected_triangles()
    cluster_n_tris = np.asarray(cluster_n_tris)
    keep_cluster = int(cluster_n_tris.argmax())
    tri_idx = np.where(np.asarray(triangle_clusters) == keep_cluster)[0].astype(int).tolist()
    mesh_main = mesh_smooth.select_by_index(tri_idx)
    mesh_main.remove_unreferenced_vertices()
    mesh_main.compute_vertex_normals()
    report.append(save_mesh(mesh_main, os.path.join(out_dir, "07_largest_component.ply"), "largest component"))

    # 8) 多分辨率简化
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

    # 9) 质量报告保存为 CSV
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
    # 可调参数（默认对大多数人体/器械点云通用）
    parser.add_argument("--nb_neighbors", type=int, default=20)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--voxel_size", type=float, default=0.003, help="meters")
    parser.add_argument("--max_nn", type=int, default=50)
    parser.add_argument("--orient_k", type=int, default=30)
    parser.add_argument("--poisson_depth", type=int, default=10)
    parser.add_argument("--density_keep_percentile", type=float, default=5.0, help="keep top (100-p)% densest verts")
    parser.add_argument("--taubin_iters", type=int, default=10)
    args = parser.parse_args()
    main(args)

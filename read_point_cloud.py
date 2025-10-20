"""
Point Cloud -> Closed & Smooth Surface Mesh (single alpha)
Usage:
  python read_point_cloud.py --in ManualFilteredPointCloud.ply --out out_mesh \
      --alpha_mul 10.3 --subdivide_iters 1 --taubin_iters 20 --project_back
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

def estimate_spacing(pcd):
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else 0.005

# ---- smoothing (Loop subdiv + Laplacian(opt) + Taubin + optional project-back) ----
def project_vertices_to_local_plane(mesh, pcd_ref, knn=30):
    if knn <= 3:
        return mesh
    if not pcd_ref.has_normals():
        pcd_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    kdtree = o3d.geometry.KDTreeFlann(pcd_ref)
    V = np.asarray(mesh.vertices)
    P = np.asarray(pcd_ref.points)
    N = np.asarray(pcd_ref.normals)
    V_new = np.empty_like(V)
    for i, v in enumerate(V):
        _, idx, _ = kdtree.search_knn_vector_3d(v, knn)
        q = P[idx]
        n = N[idx].mean(axis=0)
        n_norm = np.linalg.norm(n) + 1e-9
        n = n / n_norm
        q0 = q.mean(axis=0)
        V_new[i] = v - n * np.dot(v - q0, n)
    mesh.vertices = o3d.utility.Vector3dVector(V_new)
    return mesh

def smooth_mesh_keep_watertight(mesh_in,
                                pcd_ref=None,
                                subdivide_iters=1,
                                laplacian_iters=0,
                                taubin_iters=20,
                                project_back=False,
                                project_knn=30):
    mesh = mesh_in
    if subdivide_iters > 0:
        mesh = mesh.subdivide_loop(number_of_iterations=int(subdivide_iters))
    if laplacian_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=int(laplacian_iters))
    if taubin_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(taubin_iters))
    if project_back and (pcd_ref is not None):
        mesh = project_vertices_to_local_plane(mesh, pcd_ref, knn=int(project_knn))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh

def main(args):
    in_path = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    report = []

    # 0) 读取
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    pcd_raw = o3d.io.read_point_cloud(in_path)
    report.append(save_pcd(pcd_raw, os.path.join(out_dir, "00_raw_pointcloud.ply"), "raw"))

    # 1) 统计离群
    pcd_in, _ = pcd_raw.remove_statistical_outlier(
        nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio
    )
    report.append(save_pcd(pcd_in, os.path.join(out_dir, "01_statistical_inliers.ply"),
                           f"nb_neighbors={args.nb_neighbors}, std_ratio={args.std_ratio}"))

    # 2) 体素下采样
    pcd_ds = pcd_in.voxel_down_sample(voxel_size=args.voxel_size)
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, f"02_voxel_down_{args.voxel_size*1000:.0f}mm.ply"),
                           f"voxel_size={args.voxel_size} m"))

    # 3) 法向估计 + 一致化
    pcd_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size*4.0, max_nn=args.max_nn))
    pcd_ds.orient_normals_consistent_tangent_plane(args.orient_k)
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, "03_normals_estimated.ply"),
                           f"radius~{args.voxel_size*4.0:.4f}, max_nn={args.max_nn}, orient_k={args.orient_k}"))
    pcd_ds.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 1.0))
    report.append(save_pcd(pcd_ds, os.path.join(out_dir, "03c_normals_facing_camera.ply"),
                           "normals toward camera (0,0,1)"))

    # 4) Alpha-Shape（单一 alpha_mul）
    spacing = estimate_spacing(pcd_ds)
    alpha = args.alpha_mul * spacing
    print(f"[AlphaShape] median nn-spacing ~ {spacing:.6f} m; alpha_mul={args.alpha_mul} -> alpha={alpha:.6f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ds, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    report.append(save_mesh(mesh, os.path.join(out_dir, "04_alpha_single.ply"),
                            note=f"alpha_mul={args.alpha_mul}, alpha={alpha:.6f}"))

    # 5) 平滑（保闭合）
    mesh_smooth = smooth_mesh_keep_watertight(
        mesh,
        pcd_ref=pcd_ds if args.project_back else None,
        subdivide_iters=args.subdivide_iters,
        laplacian_iters=args.laplacian_iters,
        taubin_iters=args.taubin_iters,
        project_back=args.project_back,
        project_knn=args.project_knn
    )
    report.append(save_mesh(mesh_smooth, os.path.join(out_dir, "06_smooth_closed.ply"),
                            note=f"subdiv={args.subdivide_iters}, lap={args.laplacian_iters}, "
                                 f"taubin={args.taubin_iters}, project_back={args.project_back}"))

    # 6) 仅保留最大连通域
    triangle_clusters, cluster_n_tris, _ = mesh_smooth.cluster_connected_triangles()
    cluster_n_tris = np.asarray(cluster_n_tris)
    keep_cluster = int(cluster_n_tris.argmax())
    tri_idx = np.where(np.asarray(triangle_clusters) == keep_cluster)[0].astype(int).tolist()
    mesh_main = mesh_smooth.select_by_index(tri_idx)
    mesh_main.remove_unreferenced_vertices()
    mesh_main.compute_vertex_normals()
    report.append(save_mesh(mesh_main, os.path.join(out_dir, "07_largest_component.ply"),
                            "largest component"))

    # 7) （可选）多分辨率简化
    def decimate_and_save(mesh_in, target_tris, tag):
        target_tris = int(min(max(target_tris, 1000), np.asarray(mesh_in.triangles).shape[0]-1))
        dec = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_tris)
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
    parser.add_argument("--in", dest="input", type=str, default="ManualFilteredPointCloud.ply", help="Input PLY point cloud")
    parser.add_argument("--out", dest="output", type=str, default="output_mesh", help="Output directory")
    # 点云预处理
    parser.add_argument("--nb_neighbors", type=int, default=20)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--voxel_size", type=float, default=0.00005, help="meters")
    parser.add_argument("--max_nn", type=int, default=100)
    parser.add_argument("--orient_k", type=int, default=50)
    # ---- 单一 alpha ----
    parser.add_argument("--alpha_mul", type=float, default=10.6, help="single alpha multiplier (default 10.6)")
    # ---- 平滑参数 ----
    parser.add_argument("--subdivide_iters", type=int, default=1, help="Loop subdivision iterations (1~2)")
    parser.add_argument("--laplacian_iters", type=int, default=0, help="Optional Laplacian iterations (0=skip)")
    parser.add_argument("--taubin_iters", type=int, default=20, help="Taubin smoothing iterations (15~40)")
    parser.add_argument("--project_back", action="store_true", help="Project vertices back to local plane of original point cloud after smoothing")
    parser.add_argument("--project_knn", type=int, default=30, help="KNN for project_back (20~50)")
    args = parser.parse_args()
    main(args)

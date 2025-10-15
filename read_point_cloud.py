import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("ManualFilteredPointCloud.ply")

# Print the point cloud details
print(pcd)
print(np.array(pcd.points).shape)

# Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=30)

# Visualize the point cloud with normals
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh.compute_vertex_normals()

# Remove low density vertices to clean up the mesh
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# mesh.remove_vertices_by_mask(vertices_to_remove)

# Visualize the reconstructed mesh
# o3d.visualization.draw_geometries([mesh])
# o3d.io.write_triangle_mesh("PoissonReconstructedMesh.obj", mesh)
import numpy as np
import open3d as o3d
from skimage import measure

# -------------------------------------------------
# 用户可调
TSDF_PATH = "tsdf_grid.npz"
PCD_PATH  = "ManualFilteredPointCloud.ply"
ISO_LEVEL = 0.0  # 我们要的是 phi = 0 的等值面
# -------------------------------------------------

print("[I] loading TSDF npz ...")
data = np.load(TSDF_PATH)

# 期望 npz 里至少有:
#   phi           (Dz, Dy, Dx) or (Dx, Dy, Dz) 取决于你保存时的顺序
#   origin        (3,)
#   voxel_size    (float,)
phi = data["phi"]
origin = data["origin"].astype(np.float64)
voxel_size = float(data["voxel_size"])

print("phi shape:", phi.shape)
print("origin:", origin)
print("voxel_size:", voxel_size)

# marching_cubes 期望输入是 (Z,Y,X)
# 我们现在不知道你phi存的轴顺序，但 test_tsdf.py 在采样时应该是按 [nx, ny, nz]
# 常见保存方式是 phi[x,y,z] (Dx,Dy,Dz).
# marching_cubes 要 (z,y,x)，所以我们需要把轴调一下。
# 假设 phi 是 (Dx, Dy, Dz). 我们转成 (Dz, Dy, Dx) 给 marching_cubes.
phi_for_mcubes = np.transpose(phi, (2,1,0))

print("[I] running marching_cubes ...")
verts_vox, faces, normals, values = measure.marching_cubes(phi_for_mcubes, level=ISO_LEVEL)

# marching_cubes 给的 verts_vox 是体素坐标，单位是 "voxel index"：
#   verts_vox[:, 0] corresponds to z
#   verts_vox[:, 1] corresponds to y
#   verts_vox[:, 2] corresponds to x
#
# 我们要把它还原回真实世界坐标:
#   world = origin + voxel_size * [x, y, z]
# 先把 (z,y,x) 还原成 (x,y,z)，然后再缩放和平移。
vx = verts_vox[:, 2]
vy = verts_vox[:, 1]
vz = verts_vox[:, 0]
verts_world = np.stack([vx, vy, vz], axis=1) * voxel_size + origin[None, :]

# 构建 Open3D 三角网格
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

# 顶点法向，如果 marching_cubes 已经给了 normals，它的顺序同 verts_vox
# normals 同样是 (z,y,x) 方向导数。我们同样需要把它调到 (x,y,z)，
# 然后把它当作近似法线塞进 mesh.vertex_normals。
nx = normals[:, 2]
ny = normals[:, 1]
nz = normals[:, 0]
normals_world = np.stack([nx, ny, nz], axis=1)
# 归一化一下，避免极端值
norms = np.linalg.norm(normals_world, axis=1, keepdims=True) + 1e-9
normals_world = normals_world / norms
mesh.vertex_normals = o3d.utility.Vector3dVector(normals_world.astype(np.float64))

# 上色（TSDF表面用橙色）
mesh.paint_uniform_color([1.0, 0.4, 0.1])

# 读取原始点云
print("[I] loading point cloud ...")
pcd = o3d.io.read_point_cloud(PCD_PATH)
pcd.paint_uniform_color([0.1, 0.7, 1.0])  # 点云用蓝绿色

print("[I] visualizing ...")
o3d.visualization.draw_geometries(
    [mesh, pcd],
    window_name="TSDF surface (orange) vs Point Cloud (blue)",
    mesh_show_back_face=True
)

print("Done.")

import numpy as np
import open3d as o3d

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("calib_zed/calib_image_zed_02.ply")

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.7, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
                                  
#pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
#pcd.colors = o3d.utility.Vector3dVector(np_colors)
#o3d.visualization.draw_geometries([pcd])
#

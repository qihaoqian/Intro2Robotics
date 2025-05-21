import os
import scipy.io as sio
import numpy as np
import open3d as o3d
import time

def read_canonical_model(model_name):
  '''
  Read canonical model from .mat file
  model_name: str, 'drill' or 'liq_container'
  return: numpy array, (N, 3)
  '''
  model_fname = os.path.join('code/icp_warm_up/data', model_name, 'model.mat')
  model = sio.loadmat(model_fname)

  cano_pc = model['Mdata'].T / 1000.0 # convert to meter

  return cano_pc


def load_pc(model_name, id):
  '''
  Load point cloud from .npy file
  model_name: str, 'drill' or 'liq_container'
  id: int, point cloud id
  return: numpy array, (N, 3)
  '''
  pc_fname = os.path.join('code/icp_warm_up/data', model_name, '%d.npy' % id)
  pc = np.load(pc_fname)

  return pc


def visualize_icp_result(source_pc, target_pc, pose, obj_name, i):
  '''
  Visualize the result of ICP
  source_pc: numpy array, (N, 3)
  target_pc: numpy array, (N, 3)
  pose: SE(4) numpy array, (4, 4)
  '''
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  # o3d.visualization.draw_geometries([source_pcd, target_pcd])

  vis = o3d.visualization.Visualizer()
  # 如果设置 visible=True，会弹出窗口，并自动进行相机视点的初始化
  vis.create_window(visible=True)

  vis.add_geometry(source_pcd)
  vis.add_geometry(target_pcd)

  # 让可视化器“跑”起来，短暂让系统更新视点和渲染
  vis.run()  # 这一步会自动根据场景内容来初始化相机视点

  # 截图
  save_path = f"code/icp_warm_up/result/{obj_name}_{i}.png"
  vis.capture_screen_image(save_path)
  vis.destroy_window()

  print(f"Saved ICP visualization result to: {save_path}")

def save_icp_result_image(source_pc, target_pc, transformation, save_path="icp_result.png", width=800, height=600):
  """
  使用 Open3D 的 OffscreenRenderer 将 ICP 配准结果保存为图片。
  
  参数：
    source_pc: numpy 数组，形状为 (N, 3) 的源点云。
    target_pc: numpy 数组，形状为 (N, 3) 的目标点云。
    transformation: 变换矩阵。如果为 3x3 则视为旋转矩阵，会自动转换为 4x4；
                    如果为 4x4 则直接使用。
    save_path: 图片保存路径（需要包含文件名和扩展名，如果只传入文件夹则自动生成文件名）。
    width: 渲染图片的宽度（像素）。
    height: 渲染图片的高度（像素）。
  """
  # 如果传入的保存路径是文件夹，则自动生成文件名
  if os.path.isdir(save_path):
      save_path = os.path.join(save_path, "icp_result.png")
  
  # 如果 transformation 是 3x3，则转换为 4x4 齐次变换矩阵
  if transformation.shape == (3, 3):
      T = np.eye(4)
      T[:3, :3] = transformation
      transformation = T

  # 构造源点云
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])  # 蓝色

  # 构造目标点云
  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])  # 红色

  # 将变换矩阵作用于源点云
  source_pcd.transform(transformation)

  # 创建 OffscreenRenderer 对象
  renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
  scene = renderer.scene

  # 设置背景色（这里设置为白色）
  scene.set_background([1.0, 1.0, 1.0, 1.0])

  # 定义一个简单的材质（点云可以使用 Unlit shader）
  material = o3d.visualization.rendering.MaterialRecord()
  material.shader = "defaultUnlit"

  # 添加两个点云到场景中
  scene.add_geometry("source", source_pcd, material)
  scene.add_geometry("target", target_pcd, material)

  # 计算整个场景的包围盒，并据此设置相机位置
  bbox = source_pcd.get_axis_aligned_bounding_box().union(target_pcd.get_axis_aligned_bounding_box())
  center = bbox.get_center()
  # 为保证点云完整出现在视野中，这里适当放大观察距离
  extent = max(bbox.get_extent()) * 1.2
  # 设置摄像机位置：这里简单地让摄像机位于 y 轴负方向
  eye = [center[0], center[1] - extent, center[2]]
  
  # 使用 renderer.setup_camera(field_of_view, bbox, eye) 设置相机参数，
  # 其中 field_of_view 单位为度，这里选择 60°
  renderer.setup_camera(60.0, bbox, eye)
  
  # 渲染场景到图像
  image = renderer.render_to_image()
  # 将图像保存到指定路径
  o3d.io.write_image(save_path, image)
  print(f"已保存图片到：{save_path}")
  
  # 释放渲染器资源
  renderer.release()


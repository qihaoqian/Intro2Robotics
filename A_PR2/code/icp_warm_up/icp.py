import numpy as np
from scipy.spatial import KDTree
from utils import read_canonical_model, load_pc, visualize_icp_result, save_icp_result_image

def decenter(pointcloud):
    # 1) 计算质心
    m = np.mean(pointcloud, axis=0)
    # 2) 去中心化
    pointcloud = pointcloud - m
    return pointcloud

def compute_mean_error(src, tgt):
    """
    计算 src 和 tgt 点云之间的平均距离。
    - 简单做法: 用KDTree找到每个点的最近邻距离的平均值
    - 或者做 src->tgt 与 tgt->src 双向的平均值
    """
    tree = KDTree(tgt)
    distances, _ = tree.query(src)
    mean_dist = np.mean(distances)
    return mean_dist

def make_z_axis_transform(angle_deg, target):
    """
    构造一个只绕Z轴旋转 angle_deg，再平移的齐次变换。
    此处的思路：先绕Z轴旋转，再将点云平移使得质心对齐（实际中可根据需求修改）。
    """
    theta = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # 这里简单地将 target 旋转，后续可添加平移使质心对齐
    target_rotated = target @ R.T

    return target_rotated, R

def global_alignment_z_axis(source, target, angle_step=5):
    """
    在 0~360 度内，每隔 angle_step 度枚举一次，找到使 source 与 target 之间的平均距离最小的旋转矩阵。
    返回 (best_R_yaw, best_error)
    """
    best_error = np.inf
    best_R_yaw = np.eye(3)

    angles = np.arange(0, 360, angle_step)
    for a in angles:
        # 对 target 仅绕 Z 轴旋转 a 度
        target_rotated, R_yaw = make_z_axis_transform(a, target)
    
        # 计算 source 与旋转后 target 之间的平均距离
        error = compute_mean_error(source, target_rotated)

        if error < best_error:
            best_error = error
            best_R_yaw = R_yaw
    
    return best_R_yaw, best_error

def estimate_rigid_transform_3D(src, tgt):
    """
    计算刚性变换 T，使得 src 点云经过 T 变换后与 tgt 点云的平方误差最小.
    利用 SVD 分解计算旋转和平移：
      1. 分别计算两组点云的质心；
      2. 去质心后计算协方差矩阵 H；
      3. 对 H 做 SVD，并得到旋转矩阵 R；
      4. 平移 t = centroid_tgt - R * centroid_src.
    返回一个 4x4 的齐次变换矩阵 T.
    """
    # 计算质心
    centroid_src = np.mean(src, axis=0)
    centroid_tgt = np.mean(tgt, axis=0)
    
    # 去质心
    src_centered = src - centroid_src
    tgt_centered = tgt - centroid_tgt
    
    # 计算协方差矩阵
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 如果 R 表示反射（行列式为负），则修正
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_tgt - R @ centroid_src
    
    # 构造齐次变换矩阵 T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def icp_least_squares(source, target, max_iterations=50, tolerance=1e-6):
    """
    任意三维刚体变换（绕 x,y,z 轴旋转+平移）的 ICP 算法，步骤如下：
      - 使用 KDTree 对 target 进行最近邻查询；
      - 利用 estimate_rigid_transform_3D() 对两组对应点做最小二乘配准；
      - 将每次迭代的变换累乘到总变换 T_icp 上。
    """
    src = np.copy(source)
    T_icp = np.eye(4)
    
    # 对固定的目标点云建立 KDTree，加速最近邻查询
    tree = KDTree(target)
    
    prev_error = np.inf
    for i in range(max_iterations):
        # 1. 查找 src 中每个点在 target 中的最近邻
        distances, indices = tree.query(src)
        target_corr = target[indices]
        
        # 2. 计算当前配准误差
        mean_error = np.mean(distances)
        # 若误差变化低于阈值，则认为已收敛
        if np.abs(prev_error - mean_error) < tolerance:
            print(f"ICP 在第 {i} 次迭代时收敛，平均误差：{mean_error:.6f}")
            break
        prev_error = mean_error
        
        # 3. 根据当前对应点对估计刚性变换
        T_iter = estimate_rigid_transform_3D(src, target_corr)
        
        # 4. 更新 src，将 T_iter 应用到 src 上
        src_hom = np.hstack((src, np.ones((src.shape[0], 1))))  # 转换为齐次坐标
        src_hom = (T_iter @ src_hom.T).T
        src = src_hom[:, :3]
        
        # 5. 累计整体变换 T_icp（注意矩阵乘法顺序）
        T_icp = T_iter @ T_icp
        
    return T_icp

if __name__ == "__main__":
    obj_name = 'drill'  # 可选 'drill' 或 'liq_container'
    num_pc = 4  # 点云数量

    # 1. 加载原始模型并去中心化
    source_pc = read_canonical_model(obj_name)
    source_pc = decenter(source_pc)

    for i in range(num_pc):
        # 2. 加载目标点云并去中心化
        target_pc = load_pc(obj_name, i)
        target_pc = decenter(target_pc)
    
        # ---- (A) 全局搜索：获得初始变换 T_init ----
        # init_R_yaw, init_error = global_alignment_z_axis(source_pc, target_pc, angle_step=30)

        # ---- (B) 使用 T_init 变换 source，再进入 ICP 做 refine ----
        # 注意：这里将旋转矩阵作用于 source 点云（也可以根据需求调整乘法顺序）
        # source_init = source_pc @ init_R_yaw.T
        # T_icp = icp_least_squares(source_init, target_pc)
        
        # 注意：最终从原始 source_pc 到 target_pc 的总变换为 T_final = T_icp @ T_init
        # 这里需要将 init_R_yaw 转换为齐次变换矩阵
        # T_init = np.eye(4)
        # T_init[:3, :3] = init_R_yaw
        # T_final = T_icp @ T_init

        # 3. 可视化 ICP 配准结果
        # visualize_icp_result(source_pc, target_pc, T_final, obj_name, i)
        # save_icp_result_image(source_pc, target_pc, T_final, save_path=f'code/icp_warm_up/result/{obj_name}_{i}.png')
        T_icp = icp_least_squares(source_pc, target_pc)
        visualize_icp_result(source_pc, target_pc, T_icp, obj_name, i)


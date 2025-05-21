import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_data import *


def inverse_transform(dx, dy, dtheta):
    c = np.cos(dtheta)
    s = np.sin(dtheta)
    dx_inv = - (c * dx + s * dy)
    dy_inv =   s * dx - c * dy
    return dx_inv, dy_inv, -dtheta

def transform_points_2d(points, dx, dy, dtheta):
    """
    将二维点集 points (N,2) 经过刚体变换 (dx, dy, dtheta)。
    返回变换后的点集 (N,2)。
    """
    c = np.cos(dtheta)
    s = np.sin(dtheta)
    R = np.array([[c, -s],
                  [s,  c]])
    return points @ R.T + np.array([dx, dy])

def best_fit_transform_2d(src, dst):
    """
    给定对应的源点集 src 和目标点集 dst (均为 N×2)，
    使用 SVD 求解刚体变换 T (dx, dy, dtheta)，使得 dst ~ T(src)。
    返回 (dx, dy, dtheta)。
    假设 src[i] 对应 dst[i]。
    """
    # 计算质心
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    # 去质心
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    # 计算协方差矩阵
    W = src_centered.T @ dst_centered
    # SVD 分解
    U, _, Vt = np.linalg.svd(W)
    R_2x2 = Vt.T @ U.T
    # 如果检测到反射，则修正
    if np.linalg.det(R_2x2) < 0:
        Vt[-1, :] *= -1
        R_2x2 = Vt.T @ U.T
    # 平移
    t = dst_mean - R_2x2 @ src_mean
    # 从旋转矩阵中恢复 dtheta
    dtheta = np.arctan2(R_2x2[1, 0], R_2x2[0, 0])
    return (t[0], t[1], dtheta)

def nearest_neighbor_association_kdtree(src_points, dst_points, max_distance=None):
    """
    使用 KD-Tree 对 src_points 的每个点在 dst_points 中找到最近邻。
    增加 max_distance 参数，若匹配距离大于 max_distance，则剔除此匹配。
    返回：
      - matched_src: 来自 src_points 的点 (N,2)
      - matched_dst: 与 matched_src 对应的 dst_points 中的点 (N,2)
    """
    kdtree = cKDTree(dst_points)
    distances, nn_indices = kdtree.query(src_points)
    if max_distance is not None:
        mask = distances < max_distance
        matched_src = src_points[mask]
        matched_dst = dst_points[nn_indices[mask]]
    else:
        matched_src = src_points
        matched_dst = dst_points[nn_indices]
    return matched_src, matched_dst

def icp_2d(src, tgt,
           max_iterations, tolerance):
    
    dx_icp, dy_icp, dtheta_icp = 0.0, 0.0, 0.0
    # current_scan_t 将在迭代中不断更新：初始设置为 source_points 的拷贝
    src_t = src.copy()
    mean_error = np.inf
    for iter_idx in range(max_iterations):
        # 1. 最近邻匹配：在 target_points 中寻找 current_scan_t 中每个点的最近邻
        matched_src, matched_dst = nearest_neighbor_association_kdtree(src_t, tgt)
        # 若匹配点数太少，则可能ICP无法继续
        if matched_src.shape[0] < 500:
            print("Iteration {}: Not enough correspondences ({}), break out.".format(iter_idx, matched_src.shape[0]))
            break
        # 2. 计算最佳刚体变换，使得 matched_dst ~ T(matched_src)
        dx, dy, dtheta = best_fit_transform_2d(matched_src, matched_dst)
        # 3. 
        src_t = transform_points_2d(matched_src, dx, dy, dtheta)
        # 4. 将该次迭代的变换累积到 dx_icp, dy_icp, dtheta_icp
        c_ = np.cos(dtheta_icp)
        s_ = np.sin(dtheta_icp)
        dx_global = c_ * dx - s_ * dy
        dy_global = s_ * dx + c_ * dy
        dx_icp    += dx_global
        dy_icp    += dy_global
        dtheta_icp += dtheta
        # 5. 计算当前误差，判断是否收敛
        new_mean_error = np.mean(np.linalg.norm(src_t - matched_dst, axis=1))
        if np.abs(mean_error - new_mean_error) < tolerance:
            break
        mean_error = new_mean_error
        if iter_idx == max_iterations - 1:
            print("ICP did not converge after {} iterations".format(max_iterations))
    return dx_icp, dy_icp, dtheta_icp, new_mean_error

def scan_matching(dataset_num):
    # ========== 1. 读取激光数据 ==========
    (lidar_angle_min, lidar_angle_max, lidar_angle_increment,
     lidar_range_min, lidar_range_max,
     lidar_ranges, lidar_stamps) = read_lidar_data(dataset_num)

    # ========== 2. 简单降采样 ==========
    # downsample_factor_t = 10  # 每隔 10 帧取 1 帧
    # lidar_ranges = lidar_ranges[:, ::downsample_factor_t]
    # lidar_stamps = lidar_stamps[::downsample_factor_t]

    num_angles, num_scans = lidar_ranges.shape
    angles = np.linspace(lidar_angle_min, lidar_angle_max, num_angles)

    # 初始化全局位姿（世界坐标系下）
    rx, ry, rtheta = 0.0, 0.0, 0.0

    # 用于存储上一帧的雷达原始扫描及其对应的全局位姿
    prev_scan_radar = None
    prev_pose = (0.0, 0.0, 0.0)

    trajectory = []
    max_error = 0.0
    iter_idx = 0
    # ========== 3. 逐帧处理 ==========
    for scan_idx in tqdm(range(num_scans), desc="Processing scans"):
        # 3.1 取出一帧数据
        distances = lidar_ranges[:, scan_idx]
        valid_mask = (distances > lidar_range_min) & (distances < lidar_range_max)
        distances_valid = distances[valid_mask]
        angles_valid = angles[valid_mask]
        if len(distances_valid) == 0:
            continue

        # 3.2 计算激光器坐标系下的 (x, y)
        x_lidar = distances_valid * np.cos(angles_valid)
        y_lidar = distances_valid * np.sin(angles_valid)
        # 当前帧的原始雷达扫描（雷达坐标系下）
        current_scan_radar = np.vstack([x_lidar, y_lidar]).T

        if prev_scan_radar is None:
            # 第一帧，直接初始化
            prev_scan_radar = current_scan_radar
            prev_pose = (rx, ry, rtheta)
            trajectory.append((rx, ry, rtheta))
            continue

        # 3.3 利用 ICP（在雷达坐标系下）求解相对变换 T，使得
        # T(current_scan_radar) ~ prev_scan_radar
        # 这样得到的 (dx_icp, dy_icp, dtheta_icp) 表示从上一帧到当前帧的运动（激光器坐标系下）
        dx_icp, dy_icp, dtheta_icp, error = icp_2d(prev_scan_radar, current_scan_radar,
                                             max_iterations=200, tolerance=1e-10)
        if error > max_error:
            max_error = error
            iter_idx = scan_idx
        # 对变换取逆，得到从上一帧到当前帧的运动
        dx_icp, dy_icp, dtheta_icp = inverse_transform(dx_icp, dy_icp, dtheta_icp)
        # dx_icp, dy_icp, dtheta_icp = -dx_icp, -dy_icp, -dtheta_icp

        # 3.4 更新全局位姿
        # 由于 dx_icp, dy_icp 是在激光器坐标系下，需要转换到世界坐标系下，
        # 这里用当前全局航向 rtheta 对其旋转
        dx_global = np.cos(rtheta) * dx_icp - np.sin(rtheta) * dy_icp
        dy_global = np.sin(rtheta) * dx_icp + np.cos(rtheta) * dy_icp
        rx += dx_global
        ry += dy_global
        rtheta += dtheta_icp

        # 3.5 可视化：将雷达扫描从雷达坐标系变换到世界坐标系进行绘图
        # 此处同时绘制上一帧和当前帧（注意：上一帧的全局位置用其采集时的全局位姿 prev_pose）
        # if scan_idx == 3462:
        #     prev_scan_world = transform_points_2d(prev_scan_radar, prev_pose[0], prev_pose[1], prev_pose[2])
        #     current_scan_world = transform_points_2d(current_scan_radar, rx, ry, rtheta)
        #     plt.clf()
        #     plt.scatter(current_scan_world[:, 0], current_scan_world[:, 1], c='r', s=2, label="current")
        #     plt.scatter(prev_scan_world[:, 0], prev_scan_world[:, 1], c='b', s=2, label="prev")
        #     plt.axis('equal')
        #     plt.legend()
        #     plt.draw()
        #     plt.pause(0.01)

        # 3.6 更新上一帧数据和采集时的全局位姿
        prev_scan_radar = current_scan_radar
        prev_pose = (rx, ry, rtheta)
        trajectory.append((rx, ry, rtheta))
        
    print("max error: ", max_error)
    print("iter_idx: ", iter_idx)
    # ========== 4. 绘制机器人轨迹 ==========
    trajectory = np.array(trajectory)
    np.save(f"npys/trajectory_icp_{dataset_num}.npy", trajectory)
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
    plt.axis('equal')
    plt.title("Robot Trajectory")
    # plt.show()
    plt.savefig(f"pics/trajectory_icp_{dataset_num}.png")

    return trajectory

if __name__ == "__main__":
    traj = scan_matching(dataset_num=21)

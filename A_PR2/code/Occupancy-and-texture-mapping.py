from scan_matching import *
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d


def bresenham(x0, y0, x1, y1):
    """
    Bresenham 算法生成从 (x0, y0) 到 (x1, y1) 的网格点列表
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def first_scan():

    (lidar_angle_min, lidar_angle_max, lidar_angle_increment,
     lidar_range_min, lidar_range_max,
     lidar_ranges, lidar_stamps) = read_lidar_data(20)

    num_angles, num_scans = lidar_ranges.shape
    angles = np.linspace(lidar_angle_min, lidar_angle_max, num_angles)

    first_scan = lidar_ranges[:, 0]
    valid_mask = (first_scan > lidar_range_min) & (first_scan < lidar_range_max)
    first_scan_valid = first_scan[valid_mask]
    angles_valid = angles[valid_mask]

    x_lidar = first_scan_valid * np.cos(angles_valid)
    y_lidar = first_scan_valid * np.sin(angles_valid)

    scan_lidar = np.vstack([x_lidar, y_lidar]).T

    # 将雷达扫描从雷达坐标系变换到世界坐标系
    rx, ry, rtheta = 0,0,0
    current_scan_world = transform_points_2d(scan_lidar, rx, ry, rtheta)
    # ========== 4. 定义地图参数 & 初始化 ==========
    resolution = 0.05            # 每个栅格 0.05 米
    grid_size = 1000              # 500x500 的地图
    cx = grid_size // 2
    cy = grid_size // 2
    log_free = -1                # 自由区域的 log-odds 增量
    log_occ = +1                 # 占据区域的 log-odds 增量

    log_odds_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    # 计算机器人的地图坐标
    robot_gx = int(rx / resolution + cx)
    robot_gy = int(ry / resolution + cy)

    # 5.3 对每个激光束，利用 Bresenham 算法更新自由区和障碍物
    for point in current_scan_world:
        xw, yw = point
        gx = int(xw / resolution + cx)
        gy = int(yw / resolution + cy)

        # 确保点在地图内
        if gx < 0 or gx >= grid_size or gy < 0 or gy >= grid_size:
            continue

        # 获取从机器人位置到终点的有序网格点列表
        line_points = bresenham(robot_gx, robot_gy, gx, gy)
        if len(line_points) == 0:
            continue

        # 更新自由区域：除了最后一个点，沿线上的所有点更新为自由
        for (px, py) in line_points[:-1]:
            log_odds_map[px, py] += log_free

        # 更新障碍物：最后一个点更新为占据
        end_px, end_py = line_points[-1]
        log_odds_map[end_px, end_py] += log_occ

    # ========== 6. 将 log-odds 转换为概率地图进行可视化 ==========
    prob_map = 1.0 / (1.0 + np.exp(np.clip(-log_odds_map, -200, 200)))
    # np.save("log_odds_map.npy", log_odds_map)
    occupancy_grid = np.zeros_like(log_odds_map, dtype=np.uint8)
    occupancy_grid[prob_map > 0.7] = 255  #障碍
    occupancy_grid[prob_map < 0.3] = 0    #自由
    occupancy_grid[(prob_map >= 0.3) & (prob_map <= 0.7)] = 128  #未知

    plt.imshow(occupancy_grid, cmap='gray_r')
    plt.title("Occupancy Grid (Bresenham update)")
    plt.colorbar()
    plt.show()

def generate_pc(disp_path, rgb_path):
    # generate sample pc from disparity images

    # IMREAD_UNCHANGED ensures we preserve the precision on depth
    disp_img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)

    # note that cv2 imports as bgr, so colors may be wrong.
    bgr_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # from writeup, compute correspondence
    height, width = disp_img.shape

    dd = np.array(-0.00304 * disp_img + 3.31)
    depth = 1.03 / dd

    mesh = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')  
    i_idxs = mesh[0].flatten()
    j_idxs = mesh[1].flatten()

    rgb_i = np.array((526.37 * i_idxs + 19276 - 7877.07 * dd.flatten()) / 585.051, dtype=np.int32)  # force int for indexing
    rgb_j = np.array((526.37 * j_idxs + 16662) / 585.051, dtype=np.int32)

    # some may be out of bounds, just clip them
    rgb_i = np.clip(rgb_i, 0, height - 1)
    rgb_j = np.clip(rgb_j, 0, width - 1)

    colors = rgb_img[rgb_i, rgb_j]

    # lets visualize the image using our transformation to make sure things look correct (using bgr for opencv)
    bgr_colors = bgr_img[rgb_i, rgb_j]
    # cv2.imshow("color", bgr_colors.reshape((height, width, 3)))

    uv1 = np.vstack([j_idxs, i_idxs, np.ones_like(i_idxs)])
    K = np.array([[585.05, 0, 242.94],
                [0, 585.05, 315.84],
                [0, 0, 1]])

    # project images to 3d points
    points = depth.flatten() * (np.linalg.inv(K) @ uv1)

    oRr = np.array([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
    # we want rRo because we have points in optical frame and want to move them to the regular frame.
    points = oRr.T @ points

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # open3d expects color channels 0-1, opencv uses uint8 0-255

    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5)  # visualize the camera regular frame for reference.

    # o3d.visualization.draw_geometries([pcd, origin])  # display the pointcloud and origin

    return pcd, points, colors

def build_OGM(dataset_num, traj_path):
    # ========== 1. 读取激光数据 ==========
    (lidar_angle_min, lidar_angle_max, lidar_angle_increment,
     lidar_range_min, lidar_range_max,
     lidar_ranges, lidar_stamps) = read_lidar_data(dataset_num)

    # ========== 2. 简单降采样 ==========
    downsample_factor_t = 10  # 每隔 10 帧取 1 帧
    lidar_ranges = lidar_ranges[:, ::downsample_factor_t]
    lidar_stamps = lidar_stamps[::downsample_factor_t]

    num_angles, num_scans = lidar_ranges.shape
    angles = np.linspace(lidar_angle_min, lidar_angle_max, num_angles)

    # ========== 3. 读取轨迹数据 ==========
    lidar_trajectory = np.load(traj_path)
    # trajectory = trajectory[::downsample_factor_t]

    # ========== 4. 定义地图参数 & 初始化 ==========
    resolution = 0.05            # 每个栅格 0.05 米
    grid_size = 1000              # 500x500 的地图
    cx = grid_size // 2
    cy = grid_size // 2
    log_free = -1                # 自由区域的 log-odds 增量
    log_occ = +1                 # 占据区域的 log-odds 增量

    log_odds_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    # ========== 5. 逐帧处理 ==========
    for scan_idx in tqdm(range(num_scans), desc="Processing scans"):
        # 5.1 取出一帧数据
        distances = lidar_ranges[:, scan_idx]
        valid_mask = (distances > lidar_range_min) & (distances < lidar_range_max)
        distances_valid = distances[valid_mask]
        angles_valid = angles[valid_mask]
        if len(distances_valid) == 0:
            continue

        # 5.2 计算激光器坐标系下的 (x, y)
        x_lidar = distances_valid * np.cos(angles_valid)
        y_lidar = distances_valid * np.sin(angles_valid)
        current_scan_radar = np.vstack([x_lidar, y_lidar]).T

        # 获取当前帧的位姿（ICP 已提供较为精确的位姿）
        rx, ry, rtheta = lidar_trajectory[scan_idx]
        # 将雷达扫描从雷达坐标系变换到世界坐标系
        current_scan_world = transform_points_2d(current_scan_radar, rx, ry, rtheta)

        # 计算机器人的地图坐标
        robot_gx = int(rx / resolution + cx)
        robot_gy = int(ry / resolution + cy)

        # 5.3 对每个激光束，利用 Bresenham 算法更新自由区和障碍物
        for point in current_scan_world:
            xw, yw = point
            gx = int(xw / resolution + cx)
            gy = int(yw / resolution + cy)

            # 确保点在地图内
            if gx < 0 or gx >= grid_size or gy < 0 or gy >= grid_size:
                continue

            # 获取从机器人位置到终点的有序网格点列表
            line_points = bresenham(robot_gx, robot_gy, gx, gy)
            if len(line_points) == 0:
                continue

            # 更新自由区域：除了最后一个点，沿线上的所有点更新为自由
            for (px, py) in line_points[:-1]:
                log_odds_map[px, py] += log_free

            # 更新障碍物：最后一个点更新为占据
            end_px, end_py = line_points[-1]
            log_odds_map[end_px, end_py] += log_occ

    # ========== 6. 将 log-odds 转换为概率地图进行可视化 ==========
    prob_map = 1.0 / (1.0 + np.exp(np.clip(-log_odds_map, -200, 200)))
    np.save(f"npys/log_odds_map_optimized_downsample_{dataset_num}.npy", log_odds_map)
    occupancy_grid = np.zeros_like(log_odds_map, dtype=np.uint8)
    occupancy_grid[prob_map > 0.7] = 255  #障碍
    occupancy_grid[prob_map < 0.3] = 0    #自由
    occupancy_grid[(prob_map >= 0.3) & (prob_map <= 0.7)] = 128  #未知

    plt.imshow(occupancy_grid, cmap='gray_r')
    plt.title("Occupancy Grid (Bresenham update)")
    plt.colorbar()
    plt.savefig(f"pics/ogm_optimized_downsample_{dataset_num}.png")
    plt.show()

def texture_mapping(dataset_num, traj_path):
    disp_stamps, rgb_stamps = read_rgbd_data(dataset_num)
    # ========== 1. 读取激光数据 ==========
    (lidar_angle_min, lidar_angle_max, lidar_angle_increment,
     lidar_range_min, lidar_range_max,
     lidar_ranges, lidar_stamps) = read_lidar_data(dataset_num)
    # ========== 2. 简单降采样 ==========
    downsample_factor_t = 10  # 每隔 10 帧取 1 帧
    lidar_ranges = lidar_ranges[:, ::downsample_factor_t]
    lidar_stamps = lidar_stamps[::downsample_factor_t]
    # ========== 3. 读取轨迹数据 ==========
    lidar_trajectory = np.load(traj_path)

    # ========== 4. 定义地图参数 & 初始化 ==========
    resolution = 0.05            # 每个栅格 0.05 米
    grid_size = 1000              # 500x500 的地图
    cx = grid_size // 2
    cy = grid_size // 2
    texture_map = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)


    log_odds_map = np.load(f"npys/log_odds_map_optimized_downsample_{dataset_num}.npy")
    prob_map = 1.0 / (1.0 + np.exp(np.clip(-log_odds_map, -10, 10)))
    occupancy_grid = np.zeros_like(prob_map, dtype=np.uint8)
    occupancy_grid[prob_map > 0.9] = 255  #障碍
    occupancy_grid[prob_map < 0.3] = 0    #自由
    occupancy_grid[(prob_map >= 0.3) & (prob_map <= 0.9)] = 128  #未知
    for (rx, ry, rtheta) in lidar_trajectory:
        gx = int(rx / resolution + cx)
        gy = int(ry / resolution + cy)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            occupancy_grid[gx, gy] = 255  # 你可以自行选择合适的灰度值

    # ========== 5. 找到每个lidar_stamps对应最近的disp_stamps和rgb_stamps ==========
    lidar_stamps_matched = []
    disp_stamps_matched = []
    rgb_stamps_matched = []
    for stamp in lidar_stamps:
        idx_disp = np.argmin(np.abs(disp_stamps - stamp))
        disp_stamps_matched.append(idx_disp)
        idx_rgb = np.argmin(np.abs(rgb_stamps - stamp))
        rgb_stamps_matched.append(idx_rgb)
        # lidar_stamps_matched.append(stamp)


    # 定义相机坐标系转换到激光坐标系的变换矩阵
    # 定义欧拉角
    roll = 0.0      # rad
    pitch = 0.45    # rad
    yaw = 0.021     # rad

    # 绕 x 轴旋转矩阵 (roll)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll),  np.cos(roll)]])

    # 绕 y 轴旋转矩阵 (pitch)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                [ 0,             1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])

    # 绕 z 轴旋转矩阵 (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,           0,            1]])

    # 整体旋转矩阵（假设欧拉角应用顺序为 roll -> pitch -> yaw）
    R_cam = Rz @ Ry @ Rx
    T_cam = np.array([0.34, 0, -0.134]).reshape(3, 1)

    for i in tqdm(range(len(lidar_stamps)), desc="Processing lidar data"):
        # stamp = lidar_stamps_matched[i]
        rgb_stamp = rgb_stamps_matched[i]
        disp_stamp = disp_stamps_matched[i]
        disp_path = "./data/dataRGBD/Disparity20/disparity20_" + str(disp_stamp+1) + ".png"
        rgb_path = "./data/dataRGBD/RGB20/rgb20_" + str(rgb_stamp+1) + ".png"
        # 获取当前帧的激光数据
        pc, points_xyz, colors = generate_pc(disp_path, rgb_path)
        # 将点云转换到雷达坐标系
        points_xyz_lidar = R_cam @ points_xyz + T_cam
        points_xyz_lidar = points_xyz_lidar.T

        # 点云转世界坐标系
        rx, ry, rtheta = lidar_trajectory[i]
        R_world = np.array([
            [np.cos(rtheta), -np.sin(rtheta), 0],
            [np.sin(rtheta),  np.cos(rtheta), 0],
            [0, 0, 1]
        ])
        T_world = np.array([rx, ry, 0])

        points_world = points_xyz_lidar @ R_world.T + T_world
    # ========== 6.1 根据 z 值筛选地板点 ==========
        # 你可以通过观察 points_xyz[:, 2] 的直方图来确定合适的阈值。
        # 例如这里假设地板点的 z 坐标在 -0.1 到 0.3 之间（需要根据实际情况调整）。
        z_values = points_world[:, 2]
        # plt.figure(figsize=(8, 6))
        # plt.hist(z_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        # plt.xlabel("Z 值")
        # plt.ylabel("点的数量")
        # plt.title("转换后点云中 Z 坐标直方图")
        # plt.show()
        floor_mask = (z_values > -0.65) & (z_values < -0.5)
        floor_points = points_world[floor_mask]
        floor_colors = colors[floor_mask]

        # pcd = o3d.geometry.PointCloud()
        
        # pcd.points = o3d.utility.Vector3dVector(floor_points)
        # pcd.colors = o3d.utility.Vector3dVector(floor_colors / 255)  # open3d expects color channels 0-1, opencv uses uint8 0-255

        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5)  # visualize the camera regular frame for reference.

        # o3d.visualization.draw_geometries([pcd, origin])
        
        # ========== 6.2 将每个地板点投影到 2D 地图上 ==========
        for j in range(floor_points.shape[0]):
            xw, yw, _ = floor_points[j]
            # 根据地图分辨率和中心坐标计算栅格索引（注意 numpy 的索引顺序为行,列）
            grid_x = int(xw / resolution + cx)
            grid_y = int(yw / resolution + cy)
            
            # 检查是否在地图范围内
            if grid_x < 0 or grid_x >= grid_size or grid_y < 0 or grid_y >= grid_size:
                continue
            if occupancy_grid[grid_x, grid_y] == 0:
                # 简单赋值：直接将该点的颜色映射到对应的网格上
                texture_map[grid_x, grid_y] = floor_colors[j]
    
    # ========== 7. 显示纹理地图 ==========
    np.save(f"npys/texture_map_optimized_{dataset_num}.npy", texture_map)
    plt.figure()
    plt.imshow(texture_map)
    plt.title("Texture Map")
    plt.savefig(f"pics/texture_map_optimized_{dataset_num}.png")
    plt.show()

if __name__ == "__main__":
    build_OGM(dataset_num=20, traj_path="npys/trajectory_optimized_downsample_20.npy")
    # first_scan()
    texture_mapping(20, traj_path="npys/trajectory_optimized_downsample_20.npy")

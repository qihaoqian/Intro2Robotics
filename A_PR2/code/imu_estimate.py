from load_data import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import cKDTree

def plot_trajectory(dataset_num):
    # 读取数据
    encoder_counts, encoder_stamps = read_encoders_data(dataset_num)
    imu_angular_velocity, imu_linear_acceleration, imu_stamps = read_imu_data(dataset_num)
    
    # 机器人参数
    TICKS_PER_REV = 360        # 每圈360个刻度
    WHEEL_DIAMETER = 0.254     # 轮子直径0.254米
    TICK_DISTANCE = 0.0022     # 每个刻度对应0.0022米
    WHEEL_BASE = (476.25 + 311.15) / 100      # 轮子间距

    imu_yaws = []      # 存储IMU提供的偏航角速度
    enc_yaws = []      # 存储编码器差速计算的偏航角速度
    time_series = []   # 存储时间戳

    # 初始化位置和姿态
    x, y, theta = 0.0, 0.0, 0.0
    trajectory = [(x, y, theta)]

    # 同步数据：找到每个编码器时间戳对应的IMU数据（最近邻匹配）
    imu_indices = np.searchsorted(imu_stamps, encoder_stamps, side='right') - 1
    imu_indices = np.clip(imu_indices, 0, len(imu_stamps) - 1)  # 防止越界

    # 遍历编码器数据
    for i in range(1, len(encoder_stamps)):
        dt = encoder_stamps[i] - encoder_stamps[i-1]
        if dt <= 0:
            continue  # 跳过无效数据

        # 读取编码器数据（FR, FL, RR, RL）
        FR = encoder_counts[0, i]
        FL = encoder_counts[1, i]
        RR = encoder_counts[2, i]
        RL = encoder_counts[3, i]
        
        # 计算左右轮行驶距离
        right_distance = ((FR + RR) / 2) * TICK_DISTANCE
        left_distance = ((FL + RL) / 2) * TICK_DISTANCE
        
        # 计算线速度
        v_t = (right_distance + left_distance) / 2 / dt  # m/s
        
        # 获取最近的IMU偏航角速度
        imu_index = imu_indices[i]
        yaw_imu = imu_angular_velocity[2, imu_index]   # rad/s

        # 差动驱动运动模型更新位置
        x += v_t * np.cos(theta) * dt
        y += v_t * np.sin(theta) * dt
        theta += yaw_imu * dt
        
        trajectory.append((x, y, theta))

    trajectory = np.array(trajectory)

    # 保存轨迹数据
    np.save(f'npys/trajectory_imu_{dataset_num}.npy', trajectory)

    # 绘制轨迹
    # trajectory_odom = np.load('trajectory.npy')
    plt.figure(figsize=(8, 6))
    # plt.plot(trajectory_odom[:, 0], trajectory_odom[:, 1], label='Lidar Trajectory')
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Encoder + IMU Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Robot Trajectory using Odometry and Encoder + IMU')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    # plt.show()
    plt.savefig(f'pics/trajectory_imu_{dataset_num}.png', dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    plot_trajectory(21)

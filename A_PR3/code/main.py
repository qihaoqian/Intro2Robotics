import numpy as np
from pr3_utils import *
from imu_localization import *
from landmark_mapping import *
import scipy.sparse as sp
from Visual_inertial_SLAM import inertial_slam


if __name__ == '__main__':
      # Load the measurements
      dataset_num = 1
      filename = f"./data/dataset0{dataset_num}/dataset0{dataset_num}.npy"
      v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu_camera, extR_T_imu_camera = load_data(filename)
      R_axis = np.array([
          [0, -1, 0],
          [0,  0, -1],
          [1,  0,  0]
      ])
      T_axis = np.eye(4)
      T_axis[:3, :3] = R_axis
      extL_T_camera_imu = inversePose(extL_T_imu_camera)
      extR_T_camera_imu = inversePose(extR_T_imu_camera)
  
      extL_T_camera_imu = T_axis.T @ extL_T_camera_imu
      extR_T_camera_imu = T_axis.T @ extR_T_camera_imu
      # --------------------------------------------------------------------------
      # (a) IMU Localization via EKF Prediction
      T_imu_world_list, P_est_list = imu_localization_ekf(v_t, w_t, timestamps)
      
      # 降采样
    #   T_imu_world_list = T_imu_world_list[::5]
    #   features = features[:, :, ::5]
      
      # Convert T_est_list to a Nx4x4 matrix	
      pose = np.stack([T_imu_world_list[i][:4, :4] for i in range(len(T_imu_world_list))], axis=0)
      np.save(f"pose_imu_ekf_{dataset_num}.npy", pose)

      # fig, ax = visualize_trajectory_2d(pose, show_ori=False)
      # plt.show()

      # --------------------------------------------------------------------------
      # (b) Landmark Mapping via EKF Prediction
      mu_final, P_final, landmarks_init = stereo_landmark_mapping_ekf(
          features,
          T_imu_world_list,
          K_l, K_r,
          extL_T_camera_imu, extR_T_camera_imu,
          min_obs=150,
          sigma_pix=5.0,        # 测量噪声标准差（像素）
          pixel_threshold=30.0, # 残差粗滤门限（像素）
          chi2_threshold=9.488, # 卡方检验门限
          # chi2_threshold= 13.277, # 卡方检验门限
          init_landmark_cov=1e4 # 初始路标协方差
      )

      # mu_final -> (3M,) 表示所有路标的最终坐标
      # 可 reshape 成 (M,3)
      landmarks = mu_final.reshape(-1, 3) # -1  means the length of the array is inferred
      np.save(f"landmarks_{dataset_num}.npy", landmarks)
      np.save(f"landmarks_init_{dataset_num}.npy", landmarks_init)

      # --------------------------------------------------------------------------
      # 可视化路标与轨迹验证预测结果
      trajectory_xy = pose[:, :2, 3] 
      landmarks = np.load(f"landmarks_{dataset_num}.npy")
      # 生成一个布尔掩码，True 表示该行（点）所有坐标都在 [-200, 200] 范围内
      mask = (landmarks[:, 0] > -150) & (landmarks[:, 0] < 50) & (landmarks[:, 1] > -50) & (landmarks[:, 1] < 75)
      # 根据掩码保留合格点
      landmarks_filtered = landmarks[mask]
      print("Filtered landmarks number:", landmarks_filtered.shape[0])

      # 绘图
      fig, ax = plt.subplots()
      ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], label="Estimated Trajectory", color="blue")
      ax.scatter(landmarks_filtered[:, 0], landmarks_filtered[:, 1], label="Landmarks", color="red", marker="o", s=0.3)
      ax.scatter(landmarks_init[:, 0], landmarks_init[:, 1], label="Initial Landmarks", color="green", marker="o", s=0.3)

      ax.set_xlabel("X (meters)")
      ax.set_ylabel("Y (meters)")
      ax.legend()
      ax.set_title("Trajectory and Landmarks (XY)")
      ax.axis("equal")  # 保证 x 和 y 轴比例一致
      
      plt.savefig(f"results/landmarks_{dataset_num}.png")
      plt.show()
      plt.close()

      # (c) Visual-Inertial SLAM
      trajectory_inertial = inertial_slam(
        features,
        T_imu_world_list,
        K_l, K_r,
        extL_T_camera_imu, extR_T_camera_imu,
        min_obs=350,
        sigma_pix=5.0,
        init_landmark_cov=1e4
      )
      trajectory_inertial_xy = np.array(trajectory_inertial)[:, :2, 3]
      np.save(f"trajectory_inertial_{dataset_num}.npy", trajectory_inertial_xy)
      
      trajectory_initial_xy = np.load(f"pose_imu_ekf_{dataset_num}.npy")[:, :2, 3]
      fig, ax = plt.subplots()
      ax.plot(trajectory_inertial_xy[:, 0], trajectory_inertial_xy[:, 1], label="Inertial SLAM Trajectory", color="blue")
      ax.plot(trajectory_initial_xy[:, 0], trajectory_initial_xy[:, 1], label="Initial Trajectory", color="red")
      ax.scatter(trajectory_inertial_xy[0, 0], trajectory_inertial_xy[0, 1], marker='s', label="start")
      ax.scatter(trajectory_inertial_xy[-1, 0], trajectory_inertial_xy[-1, 1], marker='o', label="end")
      ax.set_xlabel("X (meters)")
      ax.set_ylabel("Y (meters)")
      ax.legend()
      ax.set_title("Trajectory inertial SLAM (XY)")
      ax.axis("equal")  # 保证 x 和 y 轴比例一致
      
      plt.savefig(f"results/trajectory_inertial_{dataset_num}.png")
      plt.show()
      plt.close()
      # You may use the function below to visualize the robot pose over time
      # visualize_trajectory_2d(world_T_imu, show_ori = True)
      



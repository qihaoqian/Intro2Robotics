import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag, expm, inv
from pr3_utils import inversePose, hat_so3  # 假定 pr3_utils 中已有这些函数
from imu_localization import imu_localization_ekf
from landmark_mapping import *
from tqdm import tqdm

# ----------------------- 辅助函数 -----------------------
def exp_so3(omega):
    """利用 Rodrigues 公式计算 so(3) 的指数映射"""
    return expm(hat_so3(omega))

def exp_se3(delta):
    """
    给定 6 维小扰动 delta = [δphi, δp]，构造 SE(3) 上的小位姿更新
    输出 4x4 齐次变换矩阵，形式为 [ exp(δphi^), δp; 0,1 ]
    """
    delta = np.array(delta).flatten()
    delta_phi = delta[0:3]
    delta_p   = delta[3:6]
    T = np.eye(4)
    T[0:3, 0:3] = exp_so3(delta_phi)
    T[0:3, 3]   = delta_p
    return T

def compute_jacobian_stereo_pose(T_imu_world, mu_j_world,
                                 R_imu_camL, p_imu_camL, K_l,
                                 R_imu_camR, p_imu_camR, K_r,
                                 eps_z=1e-12):
    """
    计算单个路标的观测函数关于 IMU 位姿（以误差状态 [δphi; δp] 表示）的雅可比。
    其中观测函数为：z = [u_L, v_L, u_R, v_R]^T，计算过程为
      1. 先利用 T_world_imu = inversePose(T_imu_world) 将路标从世界坐标转换到 IMU 坐标： q = R*q_world + p
      2. 利用相机外参，将 q 投影到左、右相机图像平面
      3. 对于小扰动，T_world_imu 受到更新：δq = - (hat(q) δphi + δp)
    """
    # 先计算 T_world_imu = inversePose(T_imu_world)
    T_world_imu = inversePose(T_imu_world)
    R_world_imu = T_world_imu[0:3, 0:3]
    p_world_imu = T_world_imu[0:3, 3]
    # 路标在 IMU 坐标下
    q = R_world_imu @ mu_j_world + p_world_imu  # 3-vector

    # 对左相机：
    qL = R_imu_camL @ q + p_imu_camL  # 3-vector
    xL, yL, zL = qL
    zL_safe = zL if np.abs(zL) >= eps_z else eps_z
    fx_l, fy_l = K_l[0,0], K_l[1,1]
    # 投影雅可比：∂[u,v]/∂qL
    J_proj_left = np.array([[fx_l/zL_safe, 0, -fx_l*xL/(zL_safe**2)],
                            [0, fy_l/zL_safe, -fy_l*yL/(zL_safe**2)]])
    
    # 对右相机：
    qR = R_imu_camR @ q + p_imu_camR
    xR, yR, zR = qR
    zR_safe = zR if np.abs(zR) >= eps_z else eps_z
    fx_r, fy_r = K_r[0,0], K_r[1,1]
    J_proj_right = np.array([[fx_r/zR_safe, 0, -fx_r*xR/(zR_safe**2)],
                             [0, fy_r/zR_safe, -fy_r*yR/(zR_safe**2)]])
    
    # 对 q 关于 IMU 位姿误差的偏导：
    # 记 q = R_world_imu * mu + p_world_imu，且 T_world_imu 受到左乘扰动：q 变化为 q - (hat(q) δphi + δp)
    # 因此：d q/d(δphi, δp) = -[hat(q), I]，其中 δphi 与 δp 均为 3×1
    J_q_pose = - np.hstack((hat_so3(q), np.eye(3)))  # shape (3,6)

    # 利用链式法则，左相机对位姿的雅可比：
    J_pose_left = J_proj_left @ R_imu_camL @ J_q_pose  # (2,6)
    J_pose_right = J_proj_right @ R_imu_camR @ J_q_pose  # (2,6)
    
    # 合并左右相机：
    J_pose = np.vstack((J_pose_left, J_pose_right))  # (4,6)
    return J_pose

# ----------------------- 联合视觉-惯性 EKF 更新 -----------------------
def ekf_update_visual_inertial(T_imu_world, P_pose, 
                               mu_landmarks, P_landmarks,
                               visual_measure, R_meas,
                               j_list,
                               R_imu_camL, p_imu_camL, K_l,
                               R_imu_camR, p_imu_camR, K_r,
                               eps_z=1e-12):
    """
    对联合状态 X = [IMU_pose, landmarks] 进行 EKF 更新
    参数：
      T_imu_world: IMU 在世界坐标系下的位姿变换矩阵
      P_pose: 6x6 的 IMU 位姿协方差
      mu_landmarks: 路标状态向量（3M维，每个路标3维）
      P_landmarks: 路标协方差（3M x 3M 的稀疏矩阵）
      visual_measure: 所有视觉观测，形状为 (n_meas*4,) 或 (n_meas,4)
      R_meas: 视觉测量噪声（4x4矩阵）
      j_list: 长度为 n_meas 的列表，每个元素为观测对应的路标索引
      其他参数为相机与 IMU 的标定参数
    """

    n_meas = len(j_list)
    if n_meas == 0:
        return T_imu_world, P_pose, mu_landmarks, P_landmarks
    

    # 计算 T_world_imu = inversePose(T_imu_world)
    T_world_imu = inversePose(T_imu_world)

    # 初始化保存观测数据的列表
    H_pose_list = []          # 每个观测对 IMU 位姿部分的雅可比 (4x6)
    H_land_list = []          # 每个观测对应路标的雅可比 (4x3)
    valid_measurement_indices = []  # 有效观测对应的路标索引
    r_list = []               # 每个观测的残差 (4,)

    # 遍历所有观测，直接采纳每个测量
    for i, j in enumerate(j_list):
        # 提取对应的路标估计（3,）
        m_j = mu_landmarks[3*j:3*j+3]
        
        z_pred = project_landmarks_stereo(m_j.reshape(1, 3), T_world_imu,
                                          R_imu_camL, p_imu_camL, K_l,
                                          R_imu_camR, p_imu_camR, K_r, eps_z=eps_z)[0]
        r_i = visual_measure[i] - z_pred  # (4,)

        # 计算雅可比矩阵
        J_pose = compute_jacobian_stereo_pose(T_imu_world, m_j,
                                              R_imu_camL, p_imu_camL, K_l,
                                              R_imu_camR, p_imu_camR, K_r, eps_z=eps_z)
        J_land = compute_jacobian_stereo_analytic_batch(m_j.reshape(1,3), T_world_imu,
                                                        R_imu_camL, p_imu_camL, K_l,
                                                        R_imu_camR, p_imu_camR, K_r, eps_z=eps_z)[0]  # (4,3)

        valid_measurement_indices.append(j)
        H_pose_list.append(J_pose)
        H_land_list.append(J_land)
        r_list.append(r_i)

    m_inliers = len(valid_measurement_indices)

    # 构造联合状态的雅可比矩阵 H：状态顺序 [pose(6), landmarks(3M)]
    state_dim = 6 + mu_landmarks.shape[0]
    H = sp.lil_matrix((4*m_inliers, state_dim), dtype=float)
    for i in range(m_inliers):
        landmark_idx = valid_measurement_indices[i]
        H[i*4:(i+1)*4, 0:6] = H_pose_list[i]
        # 对应路标部分在联合状态中的位置为 6 + 3*landmark_idx : 6 + 3*landmark_idx+3
        H[i*4:(i+1)*4, 6 + 3*landmark_idx: 6 + 3*landmark_idx + 3] = H_land_list[i]
    H = H.tocsr()

    # 拼接所有残差向量
    r_big = np.hstack(r_list)  # (4*m_inliers,)

    # 构造测量噪声大矩阵 R_big：对每个测量都采用相同的 R_meas
    mat_list = [R_meas for _ in range(m_inliers)]
    R_big = sp.block_diag(mat_list, format='csr')

    # 构造联合状态协方差：P_full = block_diag(P_pose, P_landmarks)
    P_pose_dense = np.array(P_pose)       # 应该是 6x6 的数组
    P_landmarks_dense = P_landmarks.toarray()  # 转为 dense 数组
    P_full = block_diag(P_pose_dense, P_landmarks_dense)  # 直接返回 dense 数组


    # Kalman 增益计算：K = P_full H^T (H P_full H^T + R_big)^{-1}
    HP = H @ P_full
    S = HP @ H.T + R_big
    K = P_full @ H.T @ inv(S)

    # 状态更新
    delta_x = K @ r_big  # (6+3M,)
    # 更新 IMU 位姿（左乘更新，对应误差状态 delta_x[0:6]）
    T_update = exp_se3(delta_x[0:6])
    T_updated = T_update @ T_imu_world

    # 更新路标状态：遍历所有路标
    mu_updated = mu_landmarks.copy()
    n_landmarks = mu_landmarks.shape[0] // 3
    for landmark_idx in range(n_landmarks):
        idx = 6 + 3*landmark_idx
        mu_updated[3*landmark_idx:3*landmark_idx+3] += delta_x[idx: idx+3]

    # 更新联合协方差：P_full_updated = (I - K H) P_full
    I_full = np.eye(state_dim)
    P_full_updated = (I_full - K @ H.toarray()) @ P_full

    # 分离更新后的协方差
    P_pose_updated = P_full_updated[0:6, 0:6]
    P_landmarks_updated = sp.csr_matrix(P_full_updated[6:, 6:])

    return T_updated, P_pose_updated, mu_updated, P_landmarks_updated



def inertial_slam(features,
                  T_imu_world_list,
                  K_l, K_r,
                  extL_T_camera_imu, extR_T_camera_imu,
                  min_obs=150,
                  sigma_pix=5.0,
                  init_landmark_cov=1e4): 
    
    N = len(T_imu_world_list)
    
    # 选择关键路标
    key_ids = select_key_landmarks(features, min_obs)
    print(f"Selected {len(key_ids)} key landmarks out of {features.shape[1]}")
    features_filtered = features[:, key_ids, :]
    # 仅对选中的路标进行初始化
    M_new = len(key_ids)
    mu_landmarks = np.zeros(3 * M_new)
    # 初始化路标协方差（对每个路标用较大初始不确定性）
    diag_vals = np.ones(3 * M_new, dtype=float) * init_landmark_cov
    P_landmarks = sp.diags(diag_vals, format='csr')
    
    # 利用多帧三角化为每个路标赋初值
    # 这里利用所有帧对每个路标进行三角化（也可选取关键帧）
    landmarks_init_inertial = np.zeros((M_new, 3))
    for j in range(M_new):
        m_init = multi_view_triangulate_landmark(j, features_filtered,
                                                 T_imu_world_list,
                                                 inversePose(extL_T_camera_imu), K_l,
                                                 inversePose(extR_T_camera_imu), K_r)
        if m_init is not None:
            mu_landmarks[3*j:3*j+3] = m_init
            landmarks_init_inertial[j, :] = m_init
            
    # 初始化 IMU 位姿协方差
    # 假设状态为6维：[x, y, z, roll, pitch, yaw]
    # 这里给出一个简单的初始化方法：
    position_cov = 1e-2   # 位置的不确定性（可根据实际情况调整）
    rotation_cov = 1e-3   # 旋转的不确定性（可根据实际情况调整）
    P_pose0 = np.diag([position_cov, position_cov, position_cov,
                       rotation_cov, rotation_cov, rotation_cov])
    # 这里为了简单起见，我们假设每一时刻的 IMU EKF 预测都具有相同的初始协方差，
    # 实际应用中应由IMU传播模型根据时间步长进行更新
    P_pose_list = [P_pose0.copy() for _ in range(N)]
    
    # 预先计算相机外参（从 IMU 到相机）
    T_imu_camL = inversePose(extL_T_camera_imu)
    T_imu_camR = inversePose(extR_T_camera_imu)
    R_imu_camL = T_imu_camL[:3, :3]
    p_imu_camL = T_imu_camL[:3, 3]
    R_imu_camR = T_imu_camR[:3, :3]
    p_imu_camR = T_imu_camR[:3, 3]

    # 设定测量噪声 R_meas （单位：像素方差）
    R_measure = np.eye(4) * (sigma_pix**2)

    # 存放融合后的轨迹
    trajectory_inertial = []

    # 对每个时刻进行视觉更新
    for t in tqdm(range(1, N)):
        # 取当前时刻的 IMU 预测作为先验
        mu_pose = T_imu_world_list[t]
        P_pose = P_pose_list[t]
        
        # 收集当前帧中所有可见的路标
        landmarks_list = []
        z_list = []
        for j in range(M_new):
            meas = features[:, j, t]
            if np.all(meas == -1):
                continue
            if (meas[0] - meas[2]) <= 0.5:
                continue
            landmarks_list.append(j)
            z_list.append(meas)
        if len(landmarks_list) > 0:
            visual_measure = np.hstack(z_list).reshape(-1,4) 
            mu_pose, P_pose, mu_landmarks, P_landmarks = ekf_update_visual_inertial(
                mu_pose,
                P_pose,
                mu_landmarks,
                P_landmarks,
                visual_measure,
                R_measure,
                landmarks_list,
                R_imu_camL, p_imu_camL, K_l,
                R_imu_camR, p_imu_camR, K_r,
                eps_z=1e-12
            )
            # 将更新后的状态反馈到先验中
            T_imu_world_list[t] = mu_pose
            P_pose_list[t] = P_pose
        trajectory_inertial.append(mu_pose)
    
    return trajectory_inertial



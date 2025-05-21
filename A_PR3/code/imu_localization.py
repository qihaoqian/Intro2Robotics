import numpy as np
from pr3_utils import *
from scipy.linalg import expm

def hat_so3(omega):
    """
    将 3x1 向量 omega=[wx, wy, wz]^T 转换为反对称矩阵 [omega]_x。
    """
    return np.array([
        [0,       -omega[2],  omega[1]],
        [omega[2],      0,    -omega[0]],
        [-omega[1], omega[0],      0   ]
    ])

def exp_so3(omega):
    """
    在 so(3) 上进行指数映射。
    这里直接调用 expm(hat_so3(omega)) 即可。
    """
    return expm(hat_so3(omega))

def ekf_prediction_step(T_prev, P_prev, v, w, dt, Q):
    """
    对单步进行 EKF 预测：
    ----------
    输入:
      T_prev:  4x4 齐次变换矩阵，表示上一步 IMU 的位姿 (R, p)
      P_prev:  6x6 协方差矩阵 (误差状态为 [delta_phi, delta_p])
      v:       3x1 当前线速度测量
      w:       3x1 当前角速度测量
      dt:      当前离散步长 (timestamps[i+1] - timestamps[i])
      Q:       6x6 过程噪声协方差

    输出:
      T_pred:  4x4 齐次变换矩阵，预测后的位姿
      P_pred:  6x6 预测后的协方差
    """

    # --- 1) 从 T_prev 中提取 R, p ---
    R_prev = T_prev[0:3, 0:3]
    p_prev = T_prev[0:3, 3]

    # --- 2) 根据离散运动模型更新姿态和位置 ---
    # R_{k+1} = R_k * Exp( w_k * dt )
    R_pred = R_prev @ exp_so3(w * dt)
    # p_{k+1} = p_k + R_k * v_k * dt
    p_pred = p_prev + R_prev @ (v * dt)

    # 将更新后的 R_pred, p_pred 封装回 T_pred
    T_pred = np.eye(4)
    T_pred[0:3, 0:3] = R_pred
    T_pred[0:3, 3]   = p_pred

    # --- 3) 计算雅可比 F, G 用于协方差传播 ---
    # 状态误差: delta_x = [delta_phi, delta_p]
    # 简化的线性化结果 (不考虑重力与加速度计建模的更复杂情形):
    # 参考文献/笔记中常见的一阶近似:
    #   delta_phi_{k+1} = delta_phi_k - [w_k]_x delta_phi_k * dt  (近似 I - [w]_x dt)
    #   delta_p_{k+1}   = delta_p_k + -R_k [v_k]_x delta_phi_k * dt + ... 
    
    F = np.eye(6)
    # 对旋转误差线性化 (orientation part)
    F[0:3, 0:3] = np.eye(3) - hat_so3(w) * dt
    # 对平移误差线性化 (position part, 这里最简单的写法可能仅保留对旋转误差的影响)
    F[3:6, 0:3] = - R_prev @ hat_so3(v) * dt  # 依赖具体模型

    # G 矩阵用于映射过程噪声到状态误差
    G = np.zeros((6, 6))
    # 对应 [delta_phi, delta_p] 受到 [noise_in_w, noise_in_v] 的影响
    # 这里假设前 3 维噪声是角速度噪声，后 3 维是线速度噪声
    G[0:3, 0:3] = -np.eye(3) * dt       # 对应角速度噪声
    G[3:6, 3:6] = -R_prev * dt         # 对应速度噪声 (在世界系或载体系需根据定义做调整)

    # --- 4) 进行协方差传播 ---
    P_pred = F @ P_prev @ F.T + G @ Q @ G.T

    return T_pred, P_pred

def imu_localization_ekf(v_t, w_t, timestamps,
                         T0=None, P0=None, Q=None):
    """
    基于 IMU 测量 (v_t, w_t) 的离散时间 EKF 预测，返回每个时刻的位姿和协方差。
    ----------
    输入:
      v_t:        shape=(N,3)，每个时刻的线速度测量
      w_t:        shape=(N,3)，每个时刻的角速度测量
      timestamps: shape=(N, )，时间戳数组
      T0:         4x4 初始位姿，不传则默认单位阵
      P0:         6x6 初始协方差，不传则默认较小对角阵
      Q:          6x6 过程噪声协方差，不传则用一个示例对角阵

    输出:
      T_list:     长度 N 的列表，每个元素为 4x4 的齐次变换矩阵
      P_list:     长度 N 的列表，每个元素为 6x6 的协方差矩阵
    """
    N = len(timestamps)
    if T0 is None:
        T0 = np.eye(4)
    if P0 is None:
        P0 = np.eye(6) * 1e-4  # 一个比较小的初始不确定性
    if Q is None:
        # 这里给一个示例对角矩阵, 可以根据 IMU 噪声特性调整
        # 假设角速度噪声和速度噪声都在 3 维上均匀
        Q = np.diag([1e-4, 1e-4, 1e-4,  # w 噪声
                     1e-3, 1e-3, 1e-3]) # v 噪声

    T_list = [None]*N
    P_list = [None]*N

    # 初始化
    T_list[0] = T0
    P_list[0] = P0

    for i in range(1, N):
        dt = timestamps[i] - timestamps[i-1]
        T_prev = T_list[i-1]
        P_prev = P_list[i-1]
        # 进行 EKF 预测
        T_pred, P_pred = ekf_prediction_step(T_prev, P_prev,
                                             v_t[i], w_t[i],
                                             dt, Q)
        T_list[i] = T_pred
        P_list[i] = P_pred

    return T_list, P_list
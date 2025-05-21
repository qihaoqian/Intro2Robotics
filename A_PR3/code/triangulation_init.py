import numpy as np
from pr3_utils import *

def build_cam_matrix(K, T_world_cam):
    """
    输入:
      K            : (3,3) 相机内参矩阵
      T_world_cam : (4,4) 把世界坐标系下的点变到相机坐标系下的刚体变换
    返回:
      P : (3,4) 相机投影矩阵, 使得 x_pix = P * X_world_hom (其中 X_world_hom是齐次坐标)
    """
    # 取 T_world_cam 的前3行作为 [R | t]
    RT_3x4 = T_world_cam[:3, :]
    P = K @ RT_3x4  # 矩阵乘法, 得到 3x4
    return P

def stereo_triangulate_old(uL, vL, uR, vR,
                       T_world_imu,
                       T_imu_camL,
                       T_imu_camR,
                       K_l, K_r):
    """
    用线性最小二乘，根据左右相机像素坐标 (uL, vL) 和 (uR, vR),
    以及 T_imu_world 与 T_imu_cam{L,R}（即 IMU->world 与 IMU->相机）,
    计算出该点在 "世界坐标系" 下的三维坐标 [X_w, Y_w, Z_w]。
    """

    # 左相机: 世界 -> IMU -> camL
    T_world_camL = T_imu_camL @ T_world_imu
    P_L = build_cam_matrix(K_l, T_world_camL)

    # 右相机: 世界 -> IMU -> camR
    T_world_camR = T_imu_camR @ T_world_imu
    P_R = build_cam_matrix(K_r, T_world_camR)

    #=== 2) 组装线性方程: DLT 三角化方案 ===#
    x1 = np.array([uL, vL, 1.0], dtype=float)
    x2 = np.array([uR, vR, 1.0], dtype=float)

    A = np.zeros((4, 4), dtype=float)
    A[0, :] = x1[0] * P_L[2, :] - P_L[0, :]
    A[1, :] = x1[1] * P_L[2, :] - P_L[1, :]
    A[2, :] = x2[0] * P_R[2, :] - P_R[0, :]
    A[3, :] = x2[1] * P_R[2, :] - P_R[1, :]

    #=== 3) SVD 求解 A·X = 0 ===#
    U, S, VT = np.linalg.svd(A)
    X_hom = VT[-1, :]
    X_hom /= X_hom[-1]  # 齐次归一化

    #=== 4) 提取世界坐标 [X, Y, Z] ===#
    X_world = X_hom[:3]

    return X_world


def stereo_triangulate(
    uL, vL, uR, vR,
    T_world_imu,
    T_imu_camL,
    T_imu_camR,
    K_l, K_r
):
    """
    利用“先归一化，再DLT”的思路，估计该点在世界坐标系下的三维坐标 [X, Y, Z].
    """

    T_world_camL = T_imu_camL @ T_world_imu
    T_world_camR = T_imu_camR @ T_world_imu

    # 从 T_world_camL 中提取 3x3 旋转 & 3x1 平移
    R_L = T_world_camL[:3, :3]
    t_L = T_world_camL[:3, 3].reshape(3, 1)

    R_R = T_world_camR[:3, :3]
    t_R = T_world_camR[:3, 3].reshape(3, 1)

    # 只保留 [R | t], 后续 DLT 中使用
    P_L_ext = np.hstack([R_L, t_L])  # 3x4
    P_R_ext = np.hstack([R_R, t_R])  # 3x4

    # 2) 将像素坐标归一化 (u,v,1) -> (x,y,1)
    #   x_n = K^{-1} [u, v, 1]^T
    pix_L = np.array([uL, vL, 1.0], dtype=float)
    pix_R = np.array([uR, vR, 1.0], dtype=float)

    x1n = np.linalg.inv(K_l) @ pix_L  # 左相机归一化坐标
    x2n = np.linalg.inv(K_r) @ pix_R  # 右相机归一化坐标

    # 3) 组装线性方程 (DLT)，对 (x1n, x2n, P_L_ext, P_R_ext) 做最小二乘
    #   对应公式:  x = [x1n, y1n, 1],  P = [R|t]   (不含内参K)
    #   A[0,:] = x1n[0]*P_L_ext[2,:] - P_L_ext[0,:]
    A = np.zeros((4, 4), dtype=float)
    A[0, :] = x1n[0] * P_L_ext[2, :] - P_L_ext[0, :]
    A[1, :] = x1n[1] * P_L_ext[2, :] - P_L_ext[1, :]
    A[2, :] = x2n[0] * P_R_ext[2, :] - P_R_ext[0, :]
    A[3, :] = x2n[1] * P_R_ext[2, :] - P_R_ext[1, :]

    # 4) SVD 求解 A·X = 0
    U, S, VT = np.linalg.svd(A)
    X_hom = VT[-1, :]
    X_hom /= X_hom[-1]  # 齐次归一化

    # 5) 提取世界坐标 [X, Y, Z]
    X_world = X_hom[:3]
    return X_world


def multi_view_triangulate_landmark(j,
                                    features_filtered,    # shape=(4, M, N)
                                    T_imu_world_list,           # IMU->world, shape=(N,4,4)
                                    T_imu_camL, K_l,
                                    T_imu_camR, K_r,
                                    disparity_min=1.0):
    """
    对路标 j, 在多帧观测中, 逐帧做 'stereo_triangulate_general' 得到一批 3D 点,
    最后做简单去离群值, 再返回平均或中值 作为该路标的初始位置.
    """
    _, M, N = features_filtered.shape
    triangulated_points = []

    for t in range(N):
        meas = features_filtered[:, j, t]  # (4,)
        if np.all(meas == -1):
            continue
        uL,vL,uR,vR = meas
        if (uL - uR) < disparity_min:
            # 视差太小, 不可信
            continue

        # Triangulate with full extrinsic
        T_imu_world = T_imu_world_list[t]
        T_world_imu = inversePose(T_imu_world)
        Xw = stereo_triangulate(uL,vL,uR,vR,
                                T_world_imu,
                                T_imu_camL,
                                T_imu_camR,
                                K_l, K_r)
        triangulated_points.append(Xw)

    if len(triangulated_points)==0:
        return None  # 此路标没有可用帧做三角化

    pts_array = np.vstack(triangulated_points)  # shape=(m,3)

    # 示例: 先求中值, 再以3倍MAD剔除离群值
    median_pt = np.median(pts_array, axis=0)
    dist = np.linalg.norm(pts_array - median_pt, axis=1)
    thr = 3.0 * np.median(dist)
    inliers = pts_array[dist < thr]

    if len(inliers)==0:
        return median_pt  # 全被干掉, 就拿 median_pt
    # 否则就用 inliers 的中值当做初始化:
    m_init = np.median(inliers, axis=0)
    return m_init

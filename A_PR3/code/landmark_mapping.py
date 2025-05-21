import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pr3_utils import *
from triangulation_init import multi_view_triangulate_landmark


def select_key_landmarks(features, min_observations=1000):
    valid_map = (features != -1).any(axis=0)  # shape=(M,N)
    count = np.count_nonzero(valid_map, axis=1)  # shape=(M,)
    key_ids = np.where(count >= min_observations)[0]
    return key_ids

def init_sparse_cov(M):
    diag_vals = np.ones(3*M, dtype=float) * 1e4
    P_init_sparse = sp.diags(diag_vals, format='csr')
    return P_init_sparse

def project_landmarks_stereo(mu_j_world, T_world_imu,
                             R_imu_camL, p_imu_camL, K_l,
                             R_imu_camR, p_imu_camR, K_r,
                             eps_z=1e-12):
    """
    向量化版本：将 (N,3) 的路标在世界坐标下投影到左右相机平面，
    返回 (N,4) 数组，每行 [uL, vL, uR, vR]
    """
    R_world_imu = T_world_imu[:3, :3]
    p_world_imu = T_world_imu[:3, 3]
    # world -> IMU
    mu_j_imu = (R_world_imu @ mu_j_world.T).T + p_world_imu  # (N,3)

    # 左相机
    mu_j_camL = (R_imu_camL @ mu_j_imu.T).T + p_imu_camL  # (N,3)
    xL, yL, zL = mu_j_camL[:, 0], mu_j_camL[:, 1], mu_j_camL[:, 2]
    zL_safe = np.where(np.abs(zL) < eps_z, eps_z, zL)
    fx_l, fy_l = K_l[0, 0], K_l[1, 1]
    cx_l, cy_l = K_l[0, 2], K_l[1, 2]
    uL = fx_l * (xL / zL_safe) + cx_l
    vL = fy_l * (yL / zL_safe) + cy_l

    # 右相机
    mu_j_camR = (R_imu_camR @ mu_j_imu.T).T + p_imu_camR  # (N,3)
    xR, yR, zR = mu_j_camR[:, 0], mu_j_camR[:, 1], mu_j_camR[:, 2]
    zR_safe = np.where(np.abs(zR) < eps_z, eps_z, zR)
    fx_r, fy_r = K_r[0, 0], K_r[1, 1]
    cx_r, cy_r = K_r[0, 2], K_r[1, 2]
    uR = fx_r * (xR / zR_safe) + cx_r
    vR = fy_r * (yR / zR_safe) + cy_r

    z_pred = np.stack([uL, vL, uR, vR], axis=1)
    return z_pred

def compute_jacobian_stereo_analytic_batch(mu_j_world,
                                           T_world_imu,
                                           R_imu_camL, p_imu_camL, K_l,
                                           R_imu_camR, p_imu_camR, K_r,
                                           eps_z=1e-12):
    """
    向量化版本：批量计算 (N,4,3) 的雅可比矩阵
    """
    R_world_imu = T_world_imu[:3, :3]
    p_world_imu = T_world_imu[:3, 3]
    mu_j_imu = (R_world_imu @ mu_j_world.T).T + p_world_imu  # (N,3)

    # 左相机
    mu_j_camL = (R_imu_camL @ mu_j_imu.T).T + p_imu_camL  # (N,3)
    xL = mu_j_camL[:, 0]
    yL = mu_j_camL[:, 1]
    zL = mu_j_camL[:, 2]
    zL_safe = np.where(np.abs(zL) < eps_z, eps_z, zL)
    fx_l, fy_l = K_l[0, 0], K_l[1, 1]
    # 构造每个路标的 2x3 雅可比（两行分别对应 u 和 v）
    d_uv_d_xyz_left_0 = np.stack([fx_l / zL_safe,
                                  np.zeros_like(zL_safe),
                                  -fx_l * xL / (zL_safe**2)], axis=1)  # (N,3)
    d_uv_d_xyz_left_1 = np.stack([np.zeros_like(zL_safe),
                                  fy_l / zL_safe,
                                  -fy_l * yL / (zL_safe**2)], axis=1)  # (N,3)
    d_uv_d_xyz_left = np.stack([d_uv_d_xyz_left_0, d_uv_d_xyz_left_1], axis=1)  # (N,2,3)
    d_xyzCamL_d_world = R_imu_camL @ R_world_imu  # (3,3)
    J_left = np.einsum('nij,jk->nik', d_uv_d_xyz_left, d_xyzCamL_d_world)  # (N,2,3)

    # 右相机
    mu_j_camR = (R_imu_camR @ mu_j_imu.T).T + p_imu_camR  # (N,3)
    xR = mu_j_camR[:, 0]
    yR = mu_j_camR[:, 1]
    zR = mu_j_camR[:, 2]
    zR_safe = np.where(np.abs(zR) < eps_z, eps_z, zR)
    fx_r, fy_r = K_r[0, 0], K_r[1, 1]
    d_uv_d_xyz_right_0 = np.stack([fx_r / zR_safe,
                                   np.zeros_like(zR_safe),
                                   -fx_r * xR / (zR_safe**2)], axis=1)
    d_uv_d_xyz_right_1 = np.stack([np.zeros_like(zR_safe),
                                   fy_r / zR_safe,
                                   -fy_r * yR / (zR_safe**2)], axis=1)
    d_uv_d_xyz_right = np.stack([d_uv_d_xyz_right_0, d_uv_d_xyz_right_1], axis=1)  # (N,2,3)
    d_xyzCamR_d_world = R_imu_camR @ R_world_imu  # (3,3)
    J_right = np.einsum('nij,jk->nik', d_uv_d_xyz_right, d_xyzCamR_d_world)  # (N,2,3)

    # 合并左右相机的雅可比，得到 (N,4,3)
    J_all = np.concatenate([J_left, J_right], axis=1)
    return J_all

# ================ 优化后的 EKF 更新（批量和并行化版本） ================
def ekf_update_stereo_sparse(mu, P_sparse,
                             z, R,
                             j_list,
                             T_imu_world,
                             R_imu_camL, p_imu_camL, K_l,
                             R_imu_camR, p_imu_camR, K_r,
                             pixel_threshold=30.0,
                             chi2_threshold=9.488):
    """
    使用向量化和并行化加速每帧中多个路标的更新：
      1. 将测量投影和雅可比计算批量处理
      2. 利用多线程对每个路标进行卡方检验（gating）
    """
    state_dim = mu.shape[0]
    T_world_imu = inversePose(T_imu_world)

    if len(j_list) == 0:
        return mu, P_sparse

    # 重构测量矩阵，z 原本是 (4*n, ) -> 转为 (n,4)
    n_meas = len(j_list)
    z_all = z.reshape(n_meas, 4)  # z是测量

    # 从状态向量中批量提取对应路标 (注意：mu 的形状为 (3M, )，reshape 成 (M,3))
    mu_landmarks = mu.reshape(-1, 3)[j_list, :]  # (n_meas, 3)

    # 向量化计算投影，得到 (n_meas,4)
    z_pred_all = project_landmarks_stereo(mu_landmarks, T_world_imu,
                                          R_imu_camL, p_imu_camL, K_l,
                                          R_imu_camR, p_imu_camR, K_r)
    # 计算残差
    r_all = z_all - z_pred_all
    res_norm = np.linalg.norm(r_all, axis=1)
    # 像素阈值 gating
    mask_pixel = res_norm <= pixel_threshold
    if np.sum(mask_pixel) == 0:
        return mu, P_sparse

    # 保留像素残差满足要求的测量
    valid_idx = np.nonzero(mask_pixel)[0]
    z_valid_all = z_all[valid_idx]
    z_pred_valid_all = z_pred_all[valid_idx]
    r_valid_all = r_all[valid_idx]
    valid_j_list = np.array(j_list)[valid_idx]
    mu_valid = mu_landmarks[valid_idx]  # (n_valid,3)

    # 向量化计算雅可比 (n_valid, 4, 3)
    H_all = compute_jacobian_stereo_analytic_batch(mu_valid, T_world_imu,
                                                   R_imu_camL, p_imu_camL, K_l,
                                                   R_imu_camR, p_imu_camR, K_r)

    n_valid = len(valid_j_list)
    # 并行化卡方检验
    def chi2_test(i, j, H_j, r_i):
        # 提取路标对应的 3x3 协方差块
        P_j = P_sparse[3*j:3*j+3, 3*j:3*j+3].toarray()
        # 这里假定测量噪声对所有路标相同，取 R[0:4,0:4] 作为 R_i
        R_i = R[0:4, 0:4]
        S_i = H_j @ P_j @ H_j.T + R_i
        d2 = r_i.T @ np.linalg.inv(S_i) @ r_i
        return d2

    valid_indices_final = []
    H_list = []
    R_blocks = []
    z_valid_list = []
    z_pred_list = []
    r_list = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_valid):
            futures.append(executor.submit(chi2_test, i, valid_j_list[i], H_all[i], r_valid_all[i]))
        for i, future in enumerate(futures):
            d2 = future.result()
            if d2 < chi2_threshold:
                valid_indices_final.append(valid_j_list[i])
                H_list.append(H_all[i])
                R_blocks.append(R[0:4, 0:4])  # 假定每个路标的 R 块相同
                z_valid_list.append(z_valid_all[i])
                z_pred_list.append(z_pred_valid_all[i])
                r_list.append(r_valid_all[i])

    m_inliers = len(valid_indices_final)
    if m_inliers == 0:
        return mu, P_sparse

    # 组装大矩阵
    z_big = np.hstack(z_valid_list)
    z_pred_big = np.hstack(z_pred_list)
    H_mat = sp.lil_matrix((4*m_inliers, state_dim), dtype=float)
    R_big = np.zeros((4*m_inliers, 4*m_inliers))
    for i in range(m_inliers):
        j = valid_indices_final[i]
        H_j = H_list[i]
        R_i = R_blocks[i]
        H_mat[4*i:4*i+4, 3*j:3*j+3] = H_j
        R_big[4*i:4*i+4, 4*i:4*i+4] = R_i

    r_big = z_big - z_pred_big
    H_mat = H_mat.tocsr()

    # 计算 S = H P H^T + R_big
    HP = H_mat @ P_sparse
    S = (HP @ H_mat.T).toarray() + R_big

    # 计算 Kalman 增益 K_gain = P H^T S^-1
    # 正确顺序：先计算 Y = P_sparse @ H_mat.T
    Y = (P_sparse @ H_mat.T).toarray()  # Y 的尺寸为 (state_dim, 4*m_inliers)

    # 求解 S X = Y.T, 得到 X，再转置回原形状
    X = np.linalg.solve(S, Y.T).T  # X 的尺寸为 (state_dim, 4*m_inliers)
    K_gain = X

    # 状态更新
    mu_upd = mu + K_gain.dot(r_big)

    I_sparse = sp.eye(state_dim, format='csr')
    KH = K_gain @ H_mat
    KH_sparse = sp.csr_matrix(KH)   # 强制转换成 csr_matrix
    P_upd_sparse = (I_sparse - KH_sparse) @ P_sparse

    return mu_upd, P_upd_sparse

# ============ 主流程：示例性“静态路标EKF”===============
def stereo_landmark_mapping_ekf(features,
                                T_imu_world_list,    # shape=(N,4,4)
                                K_l, K_r,
                                extL_T_camera_imu, extR_T_camera_imu,
                                min_obs=50,
                                sigma_pix=5.0,        # 测量噪声标准差（像素）
                                pixel_threshold=30.0, # 残差粗滤门限（像素）
                                chi2_threshold=9.488, # 卡方检验门限
                                # chi2_threshold= 13.277, # 卡方检验门限
                                init_landmark_cov=1e4 # 初始路标协方差
                               ):

    key_ids = select_key_landmarks(features, min_obs)
    print(f"Selected {len(key_ids)} key landmarks out of {features.shape[1]}")
    features_filtered = features[:, key_ids, :]
    M_new = len(key_ids)
    N = features_filtered.shape[2]

    def init_sparse_cov_custom(M, init_val):
        diag_vals = np.ones(3*M, dtype=float) * init_val
        return sp.diags(diag_vals, format='csr')

    mu = np.zeros(3*M_new)
    P = init_sparse_cov_custom(M_new, init_landmark_cov) # 路标噪声

    T_imu_camL = inversePose(extL_T_camera_imu)
    T_imu_camR = inversePose(extR_T_camera_imu)
    R_imu_camL = T_imu_camL[:3, :3]
    p_imu_camL = T_imu_camL[:3, 3]
    R_imu_camR = T_imu_camR[:3, :3]
    p_imu_camR = T_imu_camR[:3, 3]

    R_per_landmark = np.eye(4) * (sigma_pix**2) #测量噪声

    # 多帧三角化，得到路标初始位置
    landmarks_init = np.zeros((M_new, 3))
    for j in range(M_new):
        m_init = multi_view_triangulate_landmark(
            j, features_filtered,
            T_imu_world_list,
            T_imu_camL, K_l,
            T_imu_camR, K_r
        )
        if m_init is not None:
            mu[3*j:3*j+3] = m_init
            landmarks_init[j, :] = m_init

    # 逐帧 EKF update
    for t in tqdm(range(N)):
        T_imu_world = T_imu_world_list[t]
        j_list = []
        z_list = []
        for j_local in range(M_new):
            meas = features_filtered[:, j_local, t]
            if np.all(meas == -1):
                continue
            uL, vL, uR, vR = meas
            if (uL - uR) <= 0.5:
                continue
            j_list.append(j_local)
            z_list.append(meas)
        if len(j_list) == 0:
            continue

        z_big = np.hstack(z_list)
        R_big = block_diag(*[R_per_landmark]*len(j_list)) # 测量噪声

        mu, P = ekf_update_stereo_sparse(
            mu, P,
            z_big, R_big,
            j_list,
            T_imu_world,
            R_imu_camL, p_imu_camL, K_l,
            R_imu_camR, p_imu_camR, K_r,
            pixel_threshold=pixel_threshold,
            chi2_threshold=chi2_threshold
        )

    return mu, P, landmarks_init

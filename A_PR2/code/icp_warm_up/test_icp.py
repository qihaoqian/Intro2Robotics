import numpy as np
from scipy.spatial import KDTree
from utils import read_canonical_model, load_pc, visualize_icp_result


def transform_points(points, T):
    """对点云做齐次变换 T, points: (N,3)"""
    N = points.shape[0]
    ones = np.ones((N,1))
    pts_hom = np.hstack((points, ones))  # 变成 (N,4)
    pts_transformed = (T @ pts_hom.T).T  # (N,4)
    return pts_transformed[:, :3]

def compute_mean_error(src, tgt):
    """
    计算 src 和 tgt 点云之间的平均距离。
    - 简单做法: 用KDTree最近邻找距离的平均值
    - 或者做 src->tgt 与 tgt->src 双向的平均值
    """
    tree = KDTree(tgt)
    distances, _ = tree.query(src)
    mean_dist = np.mean(distances)
    return mean_dist

def make_z_axis_transform(angle_deg, src_centroid, tgt_centroid):
    """
    构造一个只绕Z轴旋转 angle_deg，再平移的齐次变换。
    这里演示：先绕Z，再把 src_centroid 对齐到 tgt_centroid。
    如果需要更复杂的策略，可自行修改。
    """
    theta = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    # 平移 t = tgt_centroid - R * src_centroid
    t = tgt_centroid - R @ src_centroid

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def global_alignment_z_axis(source, target, angle_step=15.0):
    """
    在0~360度内，每隔 angle_step 度枚举一次，找到使 source -> target 平均误差最小的变换T_init。
    返回 (T_init, best_error)
    """
    src_centroid = np.mean(source, axis=0)
    tgt_centroid = np.mean(target, axis=0)

    best_error = np.inf
    best_T = np.eye(4)

    angles = np.arange(0, 360, angle_step)
    for a in angles:
        # 构造一个仅绕Z轴旋转a度，并将centroid对齐的变换
        T_test = make_z_axis_transform(a, src_centroid, tgt_centroid)
        
        # 把 source 应用此变换
        src_transformed = transform_points(source, T_test)
        
        # 计算与 target 的平均距离
        error = compute_mean_error(src_transformed, target)

        if error < best_error:
            best_error = error
            best_T = T_test
    
    return best_T, best_error

def estimate_rotation_z_axis(source, target):
    '''
    Estimate rotation around z-axis using least squares
    '''
    x_s, y_s = source[:, 0], source[:, 1]
    x_t, y_t = target[:, 0], target[:, 1]

    # Calculate numerator and denominator for arctangent
    num = np.sum(x_s * y_t - y_s * x_t)
    den = np.sum(x_s * x_t + y_s * y_t)

    # Compute rotation angle
    theta = np.arctan2(num, den)

    # Construct rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    return R, theta

def icp_least_squares(source, target, max_iterations=50, tolerance=1e-6):
    '''
    ICP algorithm using least squares for z-axis rotation estimation
    '''
    src = np.copy(source)
    tgt = np.copy(target)
    T_total = np.eye(4)

    for i in range(max_iterations):
        # Find nearest neighbors
        tree = KDTree(tgt)
        distances, indices = tree.query(src)

        src_corr = src
        tgt_corr = tgt[indices]

        # Estimate rotation around z-axis
        R, theta = estimate_rotation_z_axis(src_corr, tgt_corr)

        # Compute centroids
        src_centroid = np.mean(src_corr, axis=0)
        tgt_centroid = np.mean(tgt_corr, axis=0)

        # Compute translation
        t = tgt_centroid - R @ src_centroid

        # Construct transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        # Apply transformation
        src = (R @ src.T).T + t

        # Update total transformation
        T_total = T @ T_total

        # Check for convergence
        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break

    return T_total

if __name__ == "__main__":
    obj_name = 'drill'  # Choose 'drill' or 'liq_container'
    num_pc = 4  # Number of point clouds

    # 1. Load canonical source model
    source_pc = read_canonical_model(obj_name)

    for i in range(num_pc):
        # 2. Load target point cloud
        target_pc = load_pc(obj_name, i)

        # ---- (A) 全局搜索: 获得初始变换 T_init ----
        T_init, init_error = global_alignment_z_axis(source_pc, target_pc, angle_step=15.0)
        print(f"Global alignment initial error = {init_error}")

        # ---- (B) 使用 T_init 变换 source，再进入 ICP 做 refine ----
        source_init = transform_points(source_pc, T_init)
        T_icp = icp_least_squares(source_init, target_pc)
        
        # 注意：最终从原始 source_pc 到 target_pc 的总变换:
        T_final = T_icp @ T_init

        # 3. Visualize the result (你已有的可视化函数)
        visualize_icp_result(source_pc, target_pc, T_final)




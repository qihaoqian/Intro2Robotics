import numpy as np

def segment_intersects_aabb(p0, p1, aabb):
    """
    判断线段 p0->p1 是否与 AABB 相交。
    aabb 格式为 [xmin, ymin, zmin, xmax, ymax, zmax]。
    返回 True 表示穿过／相交，False 表示不相交。
    """
    dir = p1 - p0
    t_min, t_max = 0.0, 1.0

    # 对 x,y,z 三个维度分别计算交点区间
    for i in range(3):
        if abs(dir[i]) < 1e-8:
            # 平行于这个轴，若起点不在 slab 内就无交点
            if p0[i] < aabb[i] or p0[i] > aabb[i+3]:
                return False
        else:
            # 计算进入／离开平面的参数 t
            t1 = (aabb[i]   - p0[i]) / dir[i]
            t2 = (aabb[i+3] - p0[i]) / dir[i]
            t_near = min(t1, t2)
            t_far  = max(t1, t2)
            # 更新全局的 t_min, t_max
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return False

    # 如果 [t_min, t_max] 与 [0,1] 有重叠，则相交
    return True


def is_segment_collision_free(p0, p1, blocks, boundary):
    """
    检查线段 p0->p1：
      1) 两端点都在 boundary 范围内
      2) 不与任何 blocks 中的 AABB 相交
    boundary 格式同上： [xmin, ymin, zmin, xmax, ymax, zmax]
    blocks 为形如 (n,6) 或 (n,9) 的数组，前 6 列即 AABB 坐标。
    """
    # 先 flatten 到 1D 长度为 6 的向量：[xmin,ymin,zmin,xmax,ymax,zmax]
    b = np.asarray(boundary).reshape(-1)
    # 1) 端点必须都在 boundary 内
    for P in (p0, p1):
        if not (b[0] <= P[0] <= b[3]
                and b[1] <= P[1] <= b[4]
                and b[2] <= P[2] <= b[5]):
            return False

    # 2) 与每个障碍块检测线段–AABB 相交
    #    假设 blocks[:, :6] 是 [xmin,ymin,zmin,xmax,ymax,zmax]
    for blk in blocks[:, :6]:
        if segment_intersects_aabb(p0, p1, blk):
            return False

    return True


def is_path_collision_free(path, blocks, boundary):
    """
    给定一系列离散路径点 path (shape=(m,3))，
    检查相邻两点连线是否都“无碰撞”。
    """
    for i in range(len(path) - 1):
        if not is_segment_collision_free(path[i], path[i+1], blocks, boundary):
            return False
    return True

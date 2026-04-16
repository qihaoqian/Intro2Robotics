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
      2) 不与任何 blocks 中的 AABB 相交（向量化 slab 检测，消除 Python 逐块循环）
    boundary 格式同上： [xmin, ymin, zmin, xmax, ymax, zmax]
    blocks 为形如 (n,6) 或 (n,9) 的数组，前 6 列即 AABB 坐标。
    """
    b = np.asarray(boundary).reshape(-1)
    for P in (p0, p1):
        if not (b[0] <= P[0] <= b[3]
                and b[1] <= P[1] <= b[4]
                and b[2] <= P[2] <= b[5]):
            return False

    if len(blocks) == 0:
        return True

    blks  = blocks[:, :6]   # (n, 6)
    d     = p1 - p0          # (3,)
    n     = len(blks)
    t_min = np.zeros(n)
    t_max = np.ones(n)

    for i in range(3):
        if abs(d[i]) < 1e-8:
            # 平行于轴 i：起点不在该轴的 slab 内 → 无交点
            outside = (p0[i] < blks[:, i]) | (p0[i] > blks[:, i + 3])
            t_min[outside] = 2.0   # 强制 t_min > t_max，标记无交点
        else:
            t1 = (blks[:, i]     - p0[i]) / d[i]
            t2 = (blks[:, i + 3] - p0[i]) / d[i]
            np.maximum(t_min, np.minimum(t1, t2), out=t_min)
            np.minimum(t_max, np.maximum(t1, t2), out=t_max)

    # 任意一个 block 满足 t_min <= t_max → 有碰撞
    return not np.any(t_min <= t_max)


def is_path_collision_free(path, blocks, boundary):
    """
    给定一系列离散路径点 path (shape=(m,3))，
    检查相邻两点连线是否都“无碰撞”。
    """
    for i in range(len(path) - 1):
        if not is_segment_collision_free(path[i], path[i+1], blocks, boundary):
            return False
    return True

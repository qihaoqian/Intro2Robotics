import numpy as np
import heapq
from collision_test import is_segment_collision_free


class MyPlanner:
    __slots__ = ['boundary', 'blocks', 'dR', 'weight', 'res']

    def __init__(self, boundary, blocks, weight=1.0, res=0.5):
        self.boundary = np.asarray(boundary, dtype=float)
        self.blocks   = np.asarray(blocks,    dtype=float)
        self.weight   = weight
        self.res      = res

        dX, dY, dZ = np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)          
        self.dR = dR / np.linalg.norm(dR, axis=0) * res

    def _key(self, p):
        return tuple(np.round(p / self.res).astype(int))

    def plan(self, start, goal, max_iter=100000):
        start, goal = map(np.asarray, (start, goal))
        start_key = self._key(start)
        goal_key  = self._key(goal)

        open_heap = [(self.weight * np.linalg.norm(start - goal), start_key)]
        g_score   = {start_key: 0.0}
        came_from = {}

        for it in range(max_iter):
            if not open_heap: break
            _, cur_key = heapq.heappop(open_heap)
            if cur_key == goal_key: break   

            cur = np.array(cur_key, dtype=float) * self.res

            for d in self.dR.T:
                nbr = cur + d
                nbr_key = self._key(nbr)
                if not is_segment_collision_free(cur, nbr, self.blocks, self.boundary):
                    continue
                tentative_g = g_score[cur_key] + np.linalg.norm(d)
                if tentative_g < g_score.get(nbr_key, np.inf):
                    came_from[nbr_key] = cur_key
                    g_score[nbr_key]   = tentative_g
                    f = tentative_g + self.weight * np.linalg.norm(nbr - goal)
                    heapq.heappush(open_heap, (f, nbr_key))

        else:
            return None  

        path = []
        node = goal_key
        while node != start_key:
            path.append(np.array(node) * self.res)
            node = came_from[node]
        path.append(start)
        return np.array(path[::-1])


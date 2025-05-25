import numpy as np
from collision_test import is_segment_collision_free


class MyPlanner:
    __slots__ = [
        "boundary",
        "blocks",
        "res",
        "step",
        "max_iter",
        "p_goal",
    ]

    def __init__(
        self,
        boundary,
        blocks,
        res=0.5,
        step=None,
        max_iter=500_000,
        p_goal=0.05,
    ):
        self.boundary = np.asarray(boundary, dtype=float).reshape(-1)
        self.blocks = np.asarray(blocks, dtype=float)
        self.res = float(res)
        self.step = float(step) if step is not None else float(res)
        self.max_iter = int(max_iter)
        self.p_goal = float(p_goal)

    def _inside(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.boundary[:3]) and np.all(p <= self.boundary[3:6]))

    def _sample(self, root: np.ndarray) -> np.ndarray:
        """Randomly sample a point inside the boundary."""
        if np.random.rand() < self.p_goal:
            # With small prob, sample the *root* of the opposite tree
            return root.copy()
        lo, hi = self.boundary[:3], self.boundary[3:6]
        return np.random.uniform(lo, hi)

    @staticmethod
    def _nearest(vertices: list[np.ndarray], q_rand: np.ndarray) -> int:
        """Return index of the vertex in vertices that is closest to q_rand."""
        dists = np.linalg.norm(np.asarray(vertices) - q_rand, axis=1)
        return int(dists.argmin())

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray | None:
        """Return a new configuration stepping from q_near toward q_rand.
        """
        v = q_rand - q_near
        d = np.linalg.norm(v)
        if d < 1e-12:
            return None
        return q_near + v * (min(self.step, d) / d)

    def _collision_free(self, p: np.ndarray, q: np.ndarray) -> bool:
        return is_segment_collision_free(p, q, self.blocks, self.boundary)

    def plan(self, start, goal):
        start, goal = map(lambda x: np.asarray(x, dtype=float), (start, goal))

        # ≡ Two RRT trees: A rooted at start, B rooted at goal -------------
        tree_a = {"V": [start], "P": [-1]}  # vertices, parent indices
        tree_b = {"V": [goal], "P": [-1]}

        for _ in range(self.max_iter):
            # Extend Tree A toward a random sample
            q_rand = self._sample(tree_b["V"][0])
            idx_near = self._nearest(tree_a["V"], q_rand)
            q_new = self._steer(tree_a["V"][idx_near], q_rand)
            if q_new is None or not self._inside(q_new):
                continue
            if not self._collision_free(tree_a["V"][idx_near], q_new):
                continue

            tree_a["V"].append(q_new)
            tree_a["P"].append(idx_near)
            idx_new_a = len(tree_a["V"]) - 1

            # Try to connect Tree B toward the new node
            idx_near_b = self._nearest(tree_b["V"], q_new)
            q_cur = tree_b["V"][idx_near_b]
            idx_parent = idx_near_b
            connected = False

            while True:
                q_next = self._steer(q_cur, q_new)
                if q_next is None:
                    break
                if not self._collision_free(q_cur, q_next):
                    break
                # Add q_next to Tree B
                tree_b["V"].append(q_next)
                tree_b["P"].append(idx_parent)
                idx_parent = len(tree_b["V"]) - 1
                q_cur = q_next
                # Trees meet?
                if np.linalg.norm(q_cur - q_new) < self.step * 0.5:
                    connected = True
                    idx_connect_b = idx_parent
                    break

            if connected:
                # A-side: Reverse the sequence after backtracking
                path_a = []
                idx = idx_new_a
                while idx != -1:
                    path_a.append(tree_a["V"][idx])    
                    idx = tree_a["P"][idx]
                path_a.reverse()                        

                # B-side: Keep the order of backtracking
                path_b = []
                idx = idx_connect_b
                while idx != -1:
                    path_b.append(tree_b["V"][idx])
                    idx = tree_b["P"][idx]

                # Concatenate
                return np.vstack((path_a + path_b))

        return None  # No path found

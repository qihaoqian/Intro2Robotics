import numpy as np
from collision_test import is_segment_collision_free


class MyPlanner:
    """
    step : float, optional
        If provided, overrides ``res`` as the extension length.

    p_goal : float, optional
        Probability of directly sampling the opposite tree’s root when
        drawing a random sample (goal‑biasing, default 0.05).
    """

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
        max_iter=20_000,
        p_goal=0.05,
    ):
        self.boundary = np.asarray(boundary, dtype=float).reshape(-1)
        self.blocks = np.asarray(blocks, dtype=float)
        self.res = float(res)
        self.step = float(step) if step is not None else float(res)
        self.max_iter = int(max_iter)
        self.p_goal = float(p_goal)

    def _inside(self, p: np.ndarray) -> bool:
        """Return True if point *p* lies strictly within the global boundary."""
        return bool(np.all(p >= self.boundary[:3]) and np.all(p <= self.boundary[3:6]))

    def _sample(self, root: np.ndarray) -> np.ndarray:
        """Randomly sample a point inside the boundary (goal‑biasing)."""
        if np.random.rand() < self.p_goal:
            # With small prob, sample the *root* of the opposite tree
            return root.copy()
        lo, hi = self.boundary[:3], self.boundary[3:6]
        return np.random.uniform(lo, hi)

    @staticmethod
    def _nearest(vertices: list[np.ndarray], q_rand: np.ndarray) -> int:
        """Return index of the vertex in *vertices* that is closest to *q_rand*."""
        dists = np.linalg.norm(np.asarray(vertices) - q_rand, axis=1)
        return int(dists.argmin())

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray | None:
        """Return a new configuration stepping from *q_near* toward *q_rand*.
        The step length is clipped to ``self.step``.  If *q_rand* coincides with
        *q_near*, returns None.
        """
        v = q_rand - q_near
        d = np.linalg.norm(v)
        if d < 1e-12:
            return None
        return q_near + v * (min(self.step, d) / d)

    def _collision_free(self, p: np.ndarray, q: np.ndarray) -> bool:
        return is_segment_collision_free(p, q, self.blocks, self.boundary)

    def plan(self, start, goal):
        """Plan a collision‑free path from *start* to *goal*.
        Returns
        -------
        numpy.ndarray | None
            ``(N, 3)`` sequence of way‑points including *start* and *goal*, or
            *None* if planning failed within ``max_iter``.
        """
        start, goal = map(lambda x: np.asarray(x, dtype=float), (start, goal))

        # ≡ Two RRT trees: A rooted at start, B rooted at goal -------------
        tree_a = {"V": [start], "P": [-1]}  # vertices, parent indices
        tree_b = {"V": [goal], "P": [-1]}

        for _ in range(self.max_iter):
            # 1) Extend Tree A toward a random sample
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

            # 2) Try to connect Tree B toward the new node
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
                # 3) A‑侧：反向回溯完后立刻反转
                path_a = []
                idx = idx_new_a
                while idx != -1:
                    path_a.append(tree_a["V"][idx])     # q_new → … → start
                    idx = tree_a["P"][idx]
                path_a.reverse()                        # start → … → q_new

                # 4) B‑侧：保持回溯得到的顺序（q_cur → … → goal），
                #    然后不反转，直接拼在后面
                path_b = []
                idx = idx_connect_b
                while idx != -1:
                    path_b.append(tree_b["V"][idx])
                    idx = tree_b["P"][idx]
                # path_b  = q_cur → … → goal

                # 5) 拼接
                return np.vstack((path_a + path_b))

        return None  # planning failed

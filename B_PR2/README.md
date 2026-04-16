# ECE276B PR2 Spring 2025

## How to Run the Motion Planner

This project implements three 3D motion planners: Bi-directional RRT, Weighted A*, and a naive greedy baseline.

### Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (numpy, matplotlib)
uv sync
```

### Usage

```bash
uv run python main.py [--planner PLANNER] [--weight WEIGHT] [--verbose]
```

### Arguments

| Argument    | Type   | Default  | Description                                                                    |
|-------------|--------|----------|--------------------------------------------------------------------------------|
| `--planner` | string | `bi_RRT` | Planner to use. Choices: `bi_RRT`, `weighted_Astar`.                          |
| `--weight`  | float  | `1.0`    | Heuristic weight for weighted A* (ignored for `bi_RRT`).                      |
| `--verbose` | flag   | `False`  | Display interactive 3D plots during planning.                                 |
| `--save`    | string | —        | Directory to save results. Creates `<DIR>/<planner>_<timestamp>/` containing `stats.csv`, per-map path PNGs, and a performance chart PNG. |

### Examples

```bash
# Run Bi-directional RRT (default)
uv run python main.py

# Run weighted A* with custom weight
uv run python main.py --planner weighted_Astar --weight 1.5

# Display interactive plots
uv run python main.py --verbose

# Save stats and figures (no display required)
uv run python main.py --planner weighted_Astar --save ./results

# Save and display
uv run python main.py --planner weighted_Astar --verbose --save ./results
```

### Saved output structure

```
results/
└── weighted_Astar_w1.0/
    ├── stats.csv            # per-map: success, path_length, planning_time_s
    ├── single_cube_path.png
    ├── maze_path.png
    ├── ...
    └── performance.png      # bar chart: path length vs planning time
```

## Planners

| Planner | Algorithm | Notes |
|---------|-----------|-------|
| `bi_RRT` | Bidirectional RRT | Probabilistic; trees swap each iteration for balanced growth |
| `weighted_Astar` | Weighted A* on discrete grid | Complete; weight > 1 trades optimality for speed |
| Naive (baseline) | Greedy 26-direction expansion | Fails on environments with local minima |

## Performance Results

See [compare.md](compare.md) for timing and path-length comparisons across all 7 test maps.

| Map          | Naive  | Weighted A* | Bi-RRT |
|--------------|--------|-------------|--------|
| single_cube  | pass   | pass        | pass   |
| maze         | fail   | pass        | fail   |
| flappy_bird  | fail   | pass        | fail   |
| pillars      | fail   | pass        | fail   |
| window       | fail   | pass        | fail   |
| tower        | pass   | pass        | pass   |
| room         | pass   | pass        | pass   |

## Optimizations

- **`Planner_weighted_Astar.py`** — inner-loop `np.linalg.norm(d)` replaced with `self.res` (direction vectors are pre-normalized to `res` length, so the norm is always constant).
- **`Planner_bi_RRT.py`** — trees swap roles each iteration so both trees grow actively (bidirectional); vertex arrays are pre-allocated as numpy arrays to eliminate repeated `list→ndarray` copies in the nearest-neighbor search.
- **`collision_test.py`** — `is_segment_collision_free` vectorized over all obstacle blocks simultaneously using the slab method, removing the Python per-block loop.
- **`Planner.py`** — inner loop variable renamed from `k` to `j` to eliminate shadowing of the outer loop's `k`.

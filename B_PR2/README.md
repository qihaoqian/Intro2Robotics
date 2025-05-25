# ECE276B PR2 Spring 2025

## 🚀 How to Run the Motion Planner

This script runs a motion planning algorithm (Bi-RRT or weighted A*) with optional parameters for weight and verbosity.

### 📄 Usage

```bash
python main.py [--planner PLANNER] [--weight WEIGHT] [--verbose]
```

### 🧾 Arguments

| Argument     | Type   | Default     | Description                                                                 |
|--------------|--------|-------------|-----------------------------------------------------------------------------|
| `--planner`  | string | `bi_RRT`    | The motion planner to use. Choices: `bi_RRT`, `weighted_Astar`.            |
| `--weight`   | float  | `1.0`       | Weight used in weighted A* planner (ignored if using `bi_RRT`).            |
| `--verbose`  | flag   | `False`     | If set, enables verbose output for debugging or detailed logging.          |

### 💡 Examples

- Run Bi-directional RRT (default):

```bash
python main.py
```

- Run weighted A* with custom weight:

```bash
python main.py --planner weighted_Astar --weight 1.5
```

- Enable verbose output:

```bash
python main.py --verbose
```







# UCSD ECE276B PR3

## Overview
In this assignment, you will implement a controller for a differential-drive car robot to track a trajectory.

## Dependencies
This starter code was tested with: python 3.10, matplotlib 3.9.0, and numpy 1.26.4.
MuJoCo and dm_control are required if you choose to use MuJoCo as your simulator.
CasADi (with IPOPT) is required for the CasADi-based CEC implementation (`cec.py`).
SciPy is required for the NumPy-based CEC implementation (`cec_numpy.py`).

## Starter code

### 1. main.py
Main entry point for the simulation. Initializes the CEC controller and runs a 120-second simulation loop on the Lissajous trajectory, printing per-step timing and cumulative translational/rotational errors. Set `use_mujoco = True` to use the MuJoCo physics engine instead of the built-in dynamics model.

### 2. utils.py
Contains shared configuration constants, the Lissajous reference trajectory generator, the differential-drive dynamics model (`car_next_state`), a simple P controller baseline, and the animated visualization function.

### 3. cec.py
CasADi-based CEC (Constrained Error Control / NMPC) implementation. Uses IPOPT to solve a finite-horizon optimal control problem at each time step. Falls back to the simple P controller if IPOPT fails to converge.

### 4. cec_numpy.py
NumPy/SciPy-based CEC implementation. Uses `scipy.optimize.minimize` with the SLSQP method. Does not require CasADi. Supports warm-starting from the previous solution to speed up convergence.

### 5. cec_original.py
Skeleton code provided as the assignment starting point for Part 1 (CEC). Contains `NotImplementedError` placeholders.

### 6. gpi.py
Skeleton code for the GPI (Generalized Policy Iteration) algorithm (Part 2 of the project). Implements a discrete-state/control-space policy iteration approach.

### 7. value_function.py
Skeleton code for the value function used by the GPI algorithm (Part 2 of the project). Supports both tabular (`GridValueFunction`) and function-approximation (`FeatureValueFunction`) approaches.

### 8. mujoco_car.py
Interface for the MuJoCo physics simulator. Provides `MujocoCarSim` which maps `[v, ω]` commands to the Ackermann steering model and steps the simulation.

## Running the simulation

### 1. Install dependencies

```bash
pip install numpy matplotlib scipy tqdm
# CasADi is only needed if using cec.py instead of cec_numpy.py
pip install casadi
# MuJoCo is only needed if use_mujoco = True
pip install mujoco dm_control
```

### 2. Choose a controller

Open `main.py` and select which CEC implementation to use (line 5):

```python
# Option A — NumPy/SciPy (no extra solver required, default)
from cec_numpy import CEC

# Option B — CasADi/IPOPT (faster, requires casadi installed)
from cec import CEC
```

To use the simple P controller baseline instead, comment out the `cec_planner` line and uncomment:

```python
control = utils.simple_controller(cur_state, cur_ref)
```

### 3. Choose a simulator

In `main.py` line 8, set:

```python
use_mujoco = False   # built-in noise model (default)
use_mujoco = True    # MuJoCo physics engine (requires mujoco + dm_control)
```

### 4. Run

```bash
cd starter_code
python main.py
```

The simulation runs for 120 seconds (240 steps at dt=0.5 s). Each step prints:

```
[v,w] [0.55 0.12]
42
0.034          ← step time (s)
[0.01 -0.02 0.03] 1.23 0.45   ← error, cumulative trans error, cumulative rot error
======================
```

At the end:

```
Total time:  8.2 s
Average iteration time:  34.1 ms
Final error_trains:  12.34
Final error_rot:  5.67
```

An animation GIF is saved to `./fig/animation<timestamp>.gif` (the `fig/` directory is created automatically) and the animation window opens.

### 5. Tune controller parameters

In `main.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `T` | `5` | Prediction horizon (steps). Larger = smoother but slower. |
| `Q` | `diag([1, 1])` | Position error weight. Increase to track position more tightly. |
| `R` | `diag([0.1, 0.05])` | Control effort weight. Increase to get smoother, smaller inputs. |
| `q` | `1` | Heading error weight. |
| `gamma` | `0.95` | Discount factor. Values closer to 1 weight future errors more. |

## Known issues fixed
- `utils.py`: noise model now correctly uses `sigma[0]`, `sigma[1]`, `sigma[2]` for x, y, θ independently.
- `cec.py`: solver failure no longer crashes the simulation; falls back to the P controller.
- `cec_numpy.py`: obstacle safety margin is consistent with `cec.py` (1e-3 added); warm-start initial guess no longer violates `v_min` bounds; constraint trajectory is computed once per optimizer call instead of twice.

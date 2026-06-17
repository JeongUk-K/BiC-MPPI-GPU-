# Quadrotor GPU New Behavior Review

This document explains the current behavior of the `quadrotor_gpu_new`
experiment so another GPT or reviewer can verify whether the implementation
matches the intended precision-landing comparison.

## Scope

Main runner:

- `run_quadrotor_gpu_new.sh`

Main source directory:

- `src/quadrotor/gpu new/`

Core shared experiment code:

- `src/quadrotor/gpu new/quadrotor_gpu_new_common.h`

Models and shared cost:

- `model/quadrotor.h`
- `model/quadrotor_precision_landing.h`
- `model/quadrotor_landing_cost.h`
- `mppi/cuda/cuda_legacy_model.cuh`

GPU solver implementations:

- `mppi/cuda/mppi_gpu.cu`
- `mppi/cuda/log_mppi_gpu.cu`
- `mppi/cuda/cluster_mppi_gpu.cu`
- `mppi/cuda/bi_mppi_gpu.cu`
- `mppi/cuda/svgd_mppi_gpu.cu`

## How To Run

Build and run all `quadrotor_gpu_new` solvers:

```bash
./run_quadrotor_gpu_new.sh --overwrite
```

Run a quick smoke test:

```bash
./run_quadrotor_gpu_new.sh --overwrite --smoke
```

Run one solver:

```bash
./run_quadrotor_gpu_new.sh --overwrite --solver mppi
./run_quadrotor_gpu_new.sh --overwrite --solver bi_mppi
```

Outputs are copied to:

```text
results/quadrotor_gpu_new/
```

Per-solver outputs:

```text
result_quadrotor_<solver>.csv
path_quadrotor_<solver>.csv
result_quadrotor_<solver>_summary.csv
result_quadrotor_all_summary.csv
```

## Compared Solvers

`run_quadrotor_gpu_new.sh` can run:

- `mppi`
- `log_mppi`
- `cluster_mppi`
- `bi_mppi`
- `svgd_mppi`

Entrypoints:

- `src/quadrotor/gpu new/mppi.cpp`
- `src/quadrotor/gpu new/log_mppi.cpp`
- `src/quadrotor/gpu new/cluster_mppi.cpp`
- `src/quadrotor/gpu new/bi_mppi.cpp`
- `src/quadrotor/gpu new/svgd_mppi.cpp`

One-directional solvers use `runQuadrotorGpuNewOneDirectional`.
Bidirectional solvers use `runQuadrotorGpuNewBidirectional` or
`runQuadrotorGpuNewSvgd`.

## Problem Definition

The quadrotor state is:

```text
x = [px, py, pz, vx, vy, vz]
```

The control is:

```text
u = [ax, ay, thrust]
```

Continuous dynamics from `model/quadrotor.h`:

```text
px_dot = vx
py_dot = vy
pz_dot = vz
vx_dot = ax
vy_dot = ay
vz_dot = thrust - g
g = 9.81
```

The experiment start and target are fixed in
`quadrotor_gpu_new_common.h`:

```text
start  = [1.5, 0.0, 5.0, 0.0, 0.0, 0.0]
target = [1.5, 5.0, 0.0, 0.0, 0.0, 0.0]
```

Interpretation:

- Start at height `z = 5.0`.
- Move from `y = 0.0` to `y = 5.0`.
- Land on the plane `z = 0.0`.
- Desired touchdown velocity is zero.

## Default Parameters

Defined in `QuadrotorGpuNewConfig`:

```text
T = 100
Tf = 50
Tb = 50
N = 10000
Nf = 10000
Nb = 10000
Nr = 5000
Ns = 10
istep = 5
maxiter = 200
map_begin = 299
num_maps = 300
dt = 0.1
gamma_u = 10.0
sigma_ax = 1.5
sigma_ay = 1.5
sigma_thrust = 1.5
eps_xy = 0.30
eps_vxy = 0.50
eps_vz = 1.00
dataset_dir = ../BARN_dataset/txt_files
```

Smoke mode overrides these with smaller values:

```text
T = 50
Tf = 25
Tb = 25
N = 256
Nf = 128
Nb = 128
Nr = 128
Ns = 4
istep = 2
maxiter = 10
num_maps = 3
```

## Map Handling And Collision

Each run loops through BARN maps from `map_begin` downward:

```text
map = map_begin, map_begin - 1, ...
```

By default this evaluates maps `299` down to `0`.

Maps are loaded from:

```text
<dataset_dir>/output_<map>.txt
```

with resolution `0.1`.

Collision checking uses the map value `10` as obstacle.

Important implementation details:

- Collision uses `x(0)` and `x(1)` only, i.e. `px`, `py`.
- Height `pz` is not used for map collision in this 2D map mode.
- If `px` is outside the map rows, collision is true.
- If `py` is outside map columns, the current checker returns false.

This last behavior may be intentional for open-ended forward progress, or it may
be a bug. It should be reviewed.

## Control Constraints

The quadrotor input projection is implemented both on CPU in
`model/quadrotor.h` and on CUDA in `mppi/cuda/cuda_legacy_model.cuh`.

The projection:

1. Computes `norm(u)`.
2. If `norm(u) >= 20.0`, multiplies the whole input by `20.0 / norm(u)`.
3. Computes:

```text
V = sqrt(ax^2 + ay^2)
S = thrust * tan(pi / 3)
```

4. If `V < -S`, sets the whole input to zero.
5. If `V > abs(S)`, projects the input toward the cone boundary:

```text
mul = 0.5 * (1.0 + S / V)
ax *= mul
ay *= mul
thrust = V * mul
```

Review note: step 2 caps the input vector norm at `20.0`; it does not normalize
the input to unit length.

## Warm Start

All quadrotor GPU-new solvers start with gravity compensation on the thrust
channel.

One-directional solvers:

```text
U_0[thrust, :] += g
```

Bidirectional solvers:

```text
U_f0[thrust, :] += g
U_b0[thrust, :] += g
dummy_u[thrust] += g
```

## Cost Function

The original `Quadrotor` model still uses position-only terminal cost:

```text
cost = || position - target_position ||
```

For `quadrotor_gpu_new`, the code now uses `QuadrotorPrecisionLanding`, which
overrides the terminal cost with a precision-landing cost from
`model/quadrotor_landing_cost.h`.

The shared host/device formula is:

```text
position_error = sqrt((px - tx)^2 + (py - ty)^2 + (pz - tz)^2)
vxy_error = sqrt((vx - tvx)^2 + (vy - tvy)^2)
vz_error = abs(vz - tvz)
landing_gate = 1 / (1 + position_error)

cost = position_error
     + landing_gate * ((eps_xy / eps_vxy) * vxy_error
                     + (eps_xy / eps_vz) * vz_error)
```

With current constants:

```text
eps_xy = 0.30
eps_vxy = 0.50
eps_vz = 1.00

vxy weight = 0.30 / 0.50 = 0.60
vz weight = 0.30 / 1.00 = 0.30
```

Interpretation:

- Far from the target, the solver mostly optimizes position.
- Near the landing target, velocity penalty becomes stronger.
- This is intended to align the optimization objective with the final precision
  landing test.

CUDA model dispatch:

- `Quadrotor` maps to `LEGACY_CUDA_QUADROTOR`, which keeps the old position-only
  cost.
- `QuadrotorPrecisionLanding` maps to
  `LEGACY_CUDA_QUADROTOR_PRECISION_LANDING`, which uses the precision-landing
  cost.

This keeps older quadrotor examples from unintentionally changing behavior.

## Rollout Cost Details

The GPU rollout kernel accumulates the terminal-style cost at each timestep and
once at the end:

```text
for t in horizon:
    cost += terminal_cost(x, target)
    x = x + dt * dynamics(x, u)
    if collision: cost = 1e8

if no collision:
    cost += terminal_cost(x, target)
```

This means the new velocity-aware cost is applied throughout the rollout, not
only at the final touchdown state.

Review note: because velocity cost is applied at every timestep, the
`landing_gate` is important. Without it, the solver may prefer becoming slow too
early rather than making progress toward the landing point.

## Closed-Loop Execution

All solvers use the same outer loop in `runQuadrotorGpuNewLoop`:

1. Save previous state `x_prev`.
2. Call `solver.solve()`.
3. Call `solver.move()` to execute the first control for one `dt`.
4. Save new state `x_next`.
5. Accumulate timing.
6. Check collision at `x_next`.
7. Check whether the trajectory crossed the landing plane `z = 0`.
8. If crossing occurred, interpolate touchdown state on `z = 0`.
9. Check collision at the touchdown state.
10. Evaluate landing metrics.
11. Stop.

If no collision and no ground crossing occurs, the loop continues until
`maxiter`.

## Landing And Termination Conditions

Collision failure:

```text
if collision(x_next):
    is_collision = true
    is_failed = true
    stop
```

Landing-plane crossing:

```text
crossed_ground = (x_prev.z > 0.0 && x_next.z <= 0.0)
```

If ground is crossed, touchdown is linearly interpolated:

```text
alpha = x_prev.z / (x_prev.z - x_next.z)
touchdown = x_prev + alpha * (x_next - x_prev)
touchdown.z = 0.0
```

Then precision landing metrics are computed:

```text
touchdown_xy_error = || touchdown.xy - target.xy ||
touchdown_vxy = || touchdown.vxvy ||
touchdown_vz_abs = abs(touchdown.vz)
touchdown_speed = || touchdown.vxyz ||
```

Precision landing success:

```text
is_precision_landed =
    touchdown_xy_error < eps_xy &&
    touchdown_vxy < eps_vxy &&
    touchdown_vz_abs < eps_vz
```

With defaults:

```text
touchdown_xy_error < 0.30 m
touchdown_vxy < 0.50 m/s
abs(touchdown_vz) < 1.00 m/s
```

Fields:

- `is_landed` means the vehicle crossed the landing plane without collision.
- `is_precision_landed` is the strict success condition.
- `is_failed = !is_precision_landed` after touchdown.
- If max iterations are exhausted without touchdown, `is_failed` remains true.

## Solver-Specific Behavior

### MPPI

Uses `MPPI_GPU`.

- Horizon: `T`
- Samples: `N`
- One-directional rollout from current state to target
- Uses weighted control sum over rollout costs

### Log-MPPI

Uses `LogMPPI_GPU`.

- Same solve structure as MPPI
- Noise is `Normal(0,1) * exp(Normal(0,1))`
- Reuses base MPPI rollout and cost

### Cluster-MPPI

Uses `ClusterMPPI_GPU`.

- GPU rollout from current state
- Copies rollout data to CPU
- Runs CPU DBSCAN over control deviation
- Computes representative control per cluster
- Chooses the cluster whose representative trajectory has lowest cost

Review note: Cluster-MPPI DBSCAN still uses `std::map` and `#pragma omp
critical`. This can become a CPU bottleneck.

### BiC-MPPI

Uses `BiMPPI_GPU`.

- Forward horizon: `Tf`
- Backward horizon: `Tb`
- Forward samples: `Nf`
- Backward samples: `Nb`
- Guide samples: `Nr`
- Runs backward rollout from target and forward rollout from current state
- Clusters forward and backward rollouts
- Selects forward/backward connection points
- Concatenates candidate paths
- Runs guide MPPI refinement

### SVGD-MPPI

Uses `SVGDMPPI_GPU`.

- Bidirectional variant with SVGD particles/surrogates
- Uses `Ns`, `istep`, `Nf`, `Nb`, `Nr`
- Also runs under `QuadrotorPrecisionLanding` in `quadrotor_gpu_new`

## Current Verification Performed

Build:

```bash
bash build_gpu.sh quadrotor_gpu_new
```

Smoke runs:

```bash
./run_quadrotor_gpu_new.sh --no-build --overwrite \
  --out results/quadrotor_gpu_new_velocity_cost_mppi_check \
  --solver mppi --smoke

./run_quadrotor_gpu_new.sh --no-build --overwrite \
  --out results/quadrotor_gpu_new_velocity_cost_bi_check \
  --solver bi_mppi --smoke
```

Both smoke runs completed and wrote CSV files.

Smoke mode only checks build/runtime correctness. It is not a meaningful
success-rate evaluation because `maxiter` is only `10`.

## Review Questions For GPT

1. Is the velocity-aware landing cost well scaled relative to the original
   position-only cost?
2. Should the velocity penalty use a smooth hinge around the thresholds instead
   of a linear velocity norm?
3. Should the `landing_gate = 1 / (1 + position_error)` be sharper, e.g. based
   on `exp(-k * position_error)`?
4. Should touchdown velocity be optimized only near `z = 0`, not just near the
   target position?
5. Should `eps_xy`, `eps_vxy`, and `eps_vz` be exposed as command-line options
   instead of compile-time constants?
6. Should map collision treat `py` out-of-bounds as collision instead of free
   space?
7. Should the closed-loop loop check initial-state collision before the first
   `solve()`?
9. Is applying terminal-style cost at every rollout step appropriate for this
   landing task, or should there be a separate stage/terminal cost split?
10. Should precision-landing success require `touchdown_speed` as well, or are
    `vxy` and `vz` sufficient?

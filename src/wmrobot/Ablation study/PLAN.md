# WMRobot BiC-MPPI Ablation Study Plan

## Goal

Create reproducible WMRobot ablation examples under `src/wmrobot/Ablation study`
and write CSV summaries with:

- `Success Rate`
- `Nfail`
- `Nbar_iter`
- `Tbar_elapsed`
- `dbar_goal`
- `dbar_conn`

## Variants

| Variant | Implementation meaning | Claim |
| --- | --- | --- |
| MPPI | baseline sampling | baseline sampling performance |
| Cluster-MPPI | clustering only | whether clustering alone is enough |
| BiC without Guide | forward/backward connection, no guide refinement | whether Guide MPPI is needed |
| BiC without Backward | forward clusters only | whether backward cost-to-go branches are needed |
| BiC without Clustering | best raw forward/backward sample connection | whether clustering gives representative branches |
| Full BiC-MPPI | full method | full structure effectiveness |

## Engineering Plan

1. Inspect existing `wmrobot` examples and solver APIs.
2. Add this resumable plan/progress log.
3. Implement a local ablation solver/harness in this folder, keeping existing solvers intact.
4. Add one executable example per variant plus a combined runner.
5. Emit per-run CSV and per-variant summary CSV.
6. Add CMake targets.
7. Build and run a small smoke test.
8. Record final commands and verification results here.

## Progress

- [x] Step 1: Existing `src/wmrobot` examples and solver boundaries inspected.
- [x] Step 2: This plan/progress log created.
- [x] Step 3: Ablation solver and experiment harness implemented.
- [x] Step 4: Six variant examples and combined runner implemented.
- [x] Step 5: CSV outputs verified.
- [x] Step 6: CMake targets added and build verified.
- [x] Step 7: Smoke test run and results recorded.

## Verification Log

- CMake configure succeeded with the space-containing folder path.
- Built targets:
  - `wmrobot_ablation_mppi`
  - `wmrobot_ablation_cluster_mppi`
  - `wmrobot_ablation_bic_without_guide`
  - `wmrobot_ablation_bic_without_backward`
  - `wmrobot_ablation_bic_without_clustering`
  - `wmrobot_ablation_full_bic_mppi`
  - `wmrobot_ablation_all`
- Smoke command:
  `./wmrobot_ablation_all --smoke --output-prefix smoke_wmrobot_ablation`
- Smoke outputs were generated in `build/`:
  - `smoke_wmrobot_ablation_summary.csv`
  - `smoke_wmrobot_ablation_<variant>_runs.csv`
  - `smoke_wmrobot_ablation_<variant>_summary.csv`
- Smoke aggregate summary header verified:
  `Variant,Success Rate,Nfail,Nbar_iter,Tbar_elapsed,dbar_goal,dbar_conn,num_runs,num_success`.
- Smoke scope is intentionally small (`num_maps=2`, `start_cases=1`,
  `maxiter=20`, low sample counts), so it validates execution and CSV shape, not
  final performance claims.

## Run Commands

From repository root:

```bash
cmake -S . -B build
cmake --build build --target wmrobot_ablation_all -j 4
cd build
./wmrobot_ablation_all --smoke --output-prefix smoke_wmrobot_ablation
```

For a larger sweep, increase the command-line knobs, for example:

```bash
./wmrobot_ablation_all \
  --num-maps 300 \
  --start-cases 2 \
  --maxiter 200 \
  --N 6000 \
  --Nf 6000 \
  --Nb 6000 \
  --Nr 3000 \
  --output-prefix wmrobot_ablation
```

Individual examples:

```bash
./wmrobot_ablation_mppi
./wmrobot_ablation_cluster_mppi
./wmrobot_ablation_bic_without_guide
./wmrobot_ablation_bic_without_backward
./wmrobot_ablation_bic_without_clustering
./wmrobot_ablation_full_bic_mppi
```

## Current Design Notes

- Existing `src/wmrobot` examples use GPU solvers.
- The ablation cuts require access to internal BiC stages, so this study will use
  a local CPU ablation solver in this folder instead of modifying GPU kernels.
- The local solver will reuse `WMRobotMap`, `CollisionChecker`, `MPPIParam`, and
  `BiMPPIParam`.
- Outputs:
  - per-run CSV: `wmrobot_ablation_<variant>_runs.csv`
  - per-variant summary CSV: `wmrobot_ablation_<variant>_summary.csv`
  - combined summary CSV from `wmrobot_ablation_all`: `wmrobot_ablation_summary.csv`
- Summary columns are:
  `Variant,Success Rate,Nfail,Nbar_iter,Tbar_elapsed,dbar_goal,dbar_conn,num_runs,num_success`.
- `dbar_conn` is `nan` for MPPI and Cluster-MPPI because they do not create
  branch connections.
- `BiC without Backward` uses the selected forward branch terminal-to-goal
  distance as its `d_conn` surrogate because no backward branch exists.
- Default run scope is moderate CPU size; `--smoke` is for quick verification and
  command-line arguments allow larger BARN sweeps.

## Resume Notes

If interrupted, continue from the first unchecked item in `Progress`.

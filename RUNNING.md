# Running Experiments

This repository now uses one runner per model group plus two study runners.
All runners build by default.  Use `--no-build` only when you want to reuse
existing `build/gpu/*` binaries.

## Quick Smoke Checks

Use these before a long run:

```bash
./run_wmrobot.sh --scenario basic --solver mppi --overwrite -- --smoke
./run_quadrotor.sh --scenario precision_landing -- --smoke
./run_ablation_study.sh --study wmrobot --smoke --overwrite
./run_waypoints.sh --scenario sequential --smoke --overwrite --no-viz
```

## General GPU Examples

WMRobot:

```bash
./run_wmrobot.sh --scenario basic --overwrite
./run_wmrobot.sh --scenario map_285 --overwrite
./run_wmrobot.sh --scenario basic --solver bi_mppi --overwrite
./run_wmrobot.sh --scenario basic --solver cluster_mppi --overwrite -- --smoke
```

For the basic WMRobot GPU examples, arguments after `--` are forwarded to the
solver executable.  Rollout logging is enabled by default:

```bash
./run_wmrobot.sh --scenario basic --solver cluster_mppi --overwrite -- --vis-every 5
./run_wmrobot.sh --scenario basic --solver all --overwrite -- --no-rollouts
```

Quadrotor:

```bash
./run_quadrotor.sh --scenario basic --overwrite
./run_quadrotor.sh --scenario precision_landing --overwrite -- --smoke
./run_quadrotor.sh --scenario basic --solver cluster_mppi --overwrite
```

Quadrotor rollout logging is enabled by default for both `basic` and
`precision_landing`:

```bash
./run_quadrotor.sh --scenario basic --solver cluster_mppi --overwrite -- --vis-every 5
./run_quadrotor.sh --scenario precision_landing --solver bi_mppi --overwrite -- --vis-every 5
./run_quadrotor.sh --scenario basic --solver all --overwrite -- --no-rollouts
```

Manipulator:

```bash
./run_manipulator.sh --overwrite
./run_manipulator.sh --solver bi_mppi --overwrite
./run_manipulator.sh --obstacle obstacles/cylinder_test.txt --overwrite
```

Velo:

```bash
./run_velo.sh --overwrite
./run_velo.sh --solver bi_mppi --overwrite
```

## Ablation Studies

Run one study:

```bash
./run_ablation_study.sh --study wmrobot --overwrite
./run_ablation_study.sh --study same_budget --overwrite
./run_ablation_study.sh --study clustering_parameter --overwrite
```

Run all ablation studies:

```bash
./run_ablation_study.sh --study all --overwrite
```

Forward experiment-specific options after `--`:

```bash
./run_ablation_study.sh --study same_budget --overwrite -- --budgets 1000 3000 6000
```

## Waypoint Experiments

Run one waypoint scenario:

```bash
./run_waypoints.sh --scenario parallel --overwrite
./run_waypoints.sh --scenario stitched --overwrite
./run_waypoints.sh --scenario sequential --overwrite
./run_waypoints.sh --scenario sequential_variants --overwrite
```

Run all waypoint scenarios:

```bash
./run_waypoints.sh --scenario all --overwrite
```

Useful waypoint options:

```bash
./run_waypoints.sh --scenario sequential --smoke --overwrite --no-viz
./run_waypoints.sh --scenario sequential_variants --overwrite --gif-fps 20
./run_waypoints.sh --scenario parallel --overwrite --no-sample-data
```

Forward solver options directly:

```bash
./run_waypoints.sh --scenario sequential --overwrite \
  --scenarios 1 \
  --maps-per-scenario 5 \
  --maxiter 60 \
  --N 256 \
  --Nf 256 \
  --Nb 256 \
  --Nr 128 \
  --vis-every 1
```

## Output Layout

Every runner writes a result directory under `results/` unless `--out DIR` is
provided.

Common files:

```text
run_manifest.txt
outputs_manifest.csv
aggregate_timing_success.csv
```

Raw CSV outputs are kept in the result directory, or under `raw_csv/` for
examples that originally write into `build/`.

Visualization data:

```text
vis_data/          rollout, cluster, and optimal path logger data
sample_vis_data/   sampled rollout data for waypoint-parallel
```

`--no-viz` skips PNG/GIF generation but keeps raw rollout data when the
executable emits it.

## Build Control

Default behavior builds first:

```bash
./run_wmrobot.sh
./run_waypoints.sh --scenario sequential_variants
```

Reuse existing binaries:

```bash
./run_wmrobot.sh --no-build
./run_waypoints.sh --scenario sequential_variants --no-build
```

If a CUDA source currently has a compile error, any runner that builds that
target will fail at the build step.  In that case either fix the compile error
or use `--no-build` with already-built binaries.

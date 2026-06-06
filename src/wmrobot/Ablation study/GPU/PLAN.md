# WMRobot GPU Ablation Study Plan

## Goal

Create GPU ablation examples under `src/wmrobot/Ablation study/GPU` using the
existing CUDA solvers and write the same CSV metrics as the CPU ablation study:

- `Success Rate`
- `Nfail`
- `Nbar_iter`
- `Tbar_elapsed`
- `dbar_goal`
- `dbar_conn`

## Variants

| Variant | GPU implementation path |
| --- | --- |
| MPPI | `MPPI_GPU` |
| Cluster-MPPI | `ClusterMPPI_GPU` |
| BiC without Guide | `BiMPPI_GPU` stages without `guideMPPI()` |
| BiC without Backward | `BiMPPI_GPU` forward rollout/clusters only |
| BiC without Clustering | GPU raw forward/backward rollouts, best raw connection |
| Full BiC-MPPI | `BiMPPI_GPU` full stage sequence |

## Progress

- [x] Step 1: Existing GPU solver and build script inspected.
- [x] Step 2: This GPU progress log created.
- [x] Step 3: GPU solver extension points added with minimal changes.
- [x] Step 4: GPU ablation harness implemented.
- [x] Step 5: GPU variant examples and combined runner implemented.
- [x] Step 6: `build_gpu.sh` target group added and build verified.
- [x] Step 7: GPU smoke run and CSV shape verified.
- [x] Step 8: Quick GPU ablation run completed and aggregate CSV verified.
- [x] Step 9: Default GPU ablation parameters aligned with `src/wmrobot/gpu`.
- [x] Step 10: Rebuilt after parameter alignment and smoke-verified execution.
- [x] Step 11: Unified `sigma_u=(0.6, 0.6)` across all GPU ablation solvers.
- [x] Step 12: Separate smoke/full all-run shell wrappers added.

## Verification Log

- Build command succeeded:
  `bash build_gpu.sh wmrobot_ablation`
- Built executables in `build/gpu/`:
  - `wmrobot_ablation_gpu_mppi`
  - `wmrobot_ablation_gpu_cluster_mppi`
  - `wmrobot_ablation_gpu_bic_without_guide`
  - `wmrobot_ablation_gpu_bic_without_backward`
  - `wmrobot_ablation_gpu_bic_without_clustering`
  - `wmrobot_ablation_gpu_full_bic_mppi`
  - `wmrobot_ablation_gpu_all`
- Smoke command succeeded from `build/`:
  `./gpu/wmrobot_ablation_gpu_all --smoke --output-prefix smoke_wmrobot_ablation_gpu`
- Smoke outputs were generated in `build/`:
  - `smoke_wmrobot_ablation_gpu_summary.csv`
  - `smoke_wmrobot_ablation_gpu_<variant>_runs.csv`
  - `smoke_wmrobot_ablation_gpu_<variant>_summary.csv`
- One-shot wrapper added and smoke-tested:
  `./run_wmrobot_ablation_gpu.sh --no-build --smoke --output-prefix smoke_script_wmrobot_ablation_gpu`
- The wrapper copies the aggregate summary to:
  `build/wmrobot_ablation_gpu_all_results.csv`
- Smoke aggregate summary header verified:
  `Variant,Success Rate,Nfail,Nbar_iter,Tbar_elapsed,dbar_goal,dbar_conn,num_runs,num_success`.
- Smoke scope is intentionally small and validates execution/CSV shape, not
  final performance claims.
- Quick verification command succeeded from repository root:
  `./run_wmrobot_ablation_gpu.sh --no-build --output-prefix wmrobot_ablation_gpu_default`
- Quick verification scope, before default parameter alignment:
  `num_maps=10`, `start_cases=2`, `maxiter=120`, `N=1000`,
  `Nf=1000`, `Nb=1000`, `Nr=700`, `raw_candidates=64`.
- Quick verification aggregate outputs were generated and verified:
  - `build/wmrobot_ablation_gpu_default_summary.csv`
  - `build/wmrobot_ablation_gpu_all_results.csv`
- Quick verification aggregate summary:

| Variant | Success Rate | Nfail | Nbar_iter | Tbar_elapsed | dbar_goal | dbar_conn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MPPI | 0.25 | 15 | 113.2 | 0.07500211975 | 0.4768488952 | nan |
| Cluster-MPPI | 0.35 | 13 | 112.6 | 0.4661542397 | 0.3462215472 | nan |
| BiC without Guide | 0.8 | 4 | 96.85 | 0.3677053784 | 0.1861852098 | 0.04195037222 |
| BiC without Backward | 0.65 | 7 | 91.55 | 0.1778986021 | 0.2500265743 | 1.569286958 |
| BiC without Clustering | 0.6 | 8 | 105.9 | 2.318019889 | 0.2906219768 | 0.002027958915 |
| Full BiC-MPPI | 0.8 | 4 | 103.4 | 1.194987983 | 0.2184143332 | 0.084797911 |
- Current defaults are now aligned with the ordinary GPU examples in
  `src/wmrobot/gpu`:
  - common sweep: `map_begin=299`, `num_maps=300`, `start_cases=2`,
    `maxiter=200`
  - MPPI / Cluster-MPPI: `T=100`, `N=10000`, `dt=0.1`, `gamma_u=10.0`,
    `sigma_u=(0.6, 0.6)`
  - BiC variants: `Tf=50`, `Tb=50`, `Nf=10000`, `Nb=10000`, `Nr=5000`,
    `dt=0.1`, `gamma_u=10.0`, `sigma_u=(0.6, 0.6)`,
    `deviation_mu=1.0`, `cost_mu=1.0`, `minpts=5`, `epsilon=0.01`,
    `psi=0.6`
  - `raw_candidates=64` remains an ablation-only cap for
    `BiC without Clustering`; it has no direct counterpart in
    `src/wmrobot/gpu`.
- Rebuild after alignment succeeded:
  `bash build_gpu.sh wmrobot_ablation`
- Smoke after alignment succeeded:
  `./run_wmrobot_ablation_gpu.sh --no-build --smoke --output-prefix smoke_aligned_wmrobot_ablation_gpu`
- The latest combined results copy currently points to this smoke verification:
  `build/wmrobot_ablation_gpu_all_results.csv`. Run the wrapper without
  `--smoke` to regenerate it with the full aligned defaults.
- All GPU ablation variants now share the same sigma path from
  `GpuAblationConfig::sigma_v/sigma_w`; Cluster-MPPI no longer has a separate
  sigma override.
- Separate all-run wrappers were added:
  - `./run_wmrobot_ablation_gpu_smoke_all.sh`
  - `./run_wmrobot_ablation_gpu_full_all.sh`
- Smoke all-run wrapper verified:
  `./run_wmrobot_ablation_gpu_smoke_all.sh --no-build`
- Full all-run wrapper path verified with a tiny non-smoke override:
  `./run_wmrobot_ablation_gpu_full_all.sh --no-build --num-maps 1 --start-cases 1 --maxiter 1 --N 128 --Nf 128 --Nb 128 --Nr 64 --raw-candidates 8 --output-prefix verify_full_wrapper_tiny`

## Run Commands

From repository root:

```bash
./run_wmrobot_ablation_gpu_smoke_all.sh
```

For the full default sweep:

```bash
./run_wmrobot_ablation_gpu_full_all.sh
```

The smoke wrapper writes `build/wmrobot_ablation_gpu_smoke_summary.csv` by
default. The full wrapper writes `build/wmrobot_ablation_gpu_full_summary.csv`
by default. Both wrappers build `wmrobot_ablation` first, run all GPU ablation
variants, and copy the aggregate summary to
`build/wmrobot_ablation_gpu_all_results.csv`. Pass `--no-build` to skip
rebuilding when binaries are already fresh.

The common runner remains available when custom arguments are needed:

```bash
./run_wmrobot_ablation_gpu.sh --raw-candidates 128 --output-prefix custom_prefix
```

Manual build/run remains available:

```bash
bash build_gpu.sh wmrobot_ablation
cd build
./gpu/wmrobot_ablation_gpu_all --smoke --output-prefix smoke_wmrobot_ablation_gpu
```

Individual examples:

```bash
./gpu/wmrobot_ablation_gpu_mppi
./gpu/wmrobot_ablation_gpu_cluster_mppi
./gpu/wmrobot_ablation_gpu_bic_without_guide
./gpu/wmrobot_ablation_gpu_bic_without_backward
./gpu/wmrobot_ablation_gpu_bic_without_clustering
./gpu/wmrobot_ablation_gpu_full_bic_mppi
```

## Design Notes

- GPU examples should be run from `build`, matching existing `run_gpu_all.sh`.
- Default dataset path is therefore `../BARN_dataset/txt_files`.
- `BiC without Clustering` needs raw perturbed controls from the GPU rollout
  buffers, so `BiMPPI_GPU` will expose protected raw rollout helpers while
  keeping public solver behavior unchanged.
- The raw no-clustering connection search uses the top feasible raw candidates
  by rollout cost. The cap is controlled by `--raw-candidates` to keep the
  pairwise connection search tractable in large sweeps.

## Implemented Files

- `wmrobot_ablation_gpu_common.h`
- `wmrobot_ablation_gpu_mppi.cpp`
- `wmrobot_ablation_gpu_cluster_mppi.cpp`
- `wmrobot_ablation_gpu_bic_without_guide.cpp`
- `wmrobot_ablation_gpu_bic_without_backward.cpp`
- `wmrobot_ablation_gpu_bic_without_clustering.cpp`
- `wmrobot_ablation_gpu_full_bic_mppi.cpp`
- `wmrobot_ablation_gpu_all.cpp`

## Resume Notes

If interrupted, continue from the first unchecked item in `Progress`.

# Manipulator BiC-MPPI with workspace link collision

This package extends the previous manipulator BiC-MPPI example from a q-space toy obstacle to a workspace link-collision benchmark.

## What changed

- `collision_checker.h`
  - Adds workspace AABB boxes `{xmin,xmax,ymin,ymax,zmin,zmax}`.
  - Adds CPU FK-based link collision for `x=[q;qdot]` or `x=q`.
  - Approximates each link by 5 sampled points along the segment between adjacent FK joint positions.

- `cuda_legacy_model.cuh`
  - Adds device-side DH forward kinematics.
  - Adds link sampled-point vs AABB distance checks.
  - Adds workspace obstacle soft cost for manipulator rollouts.

- `mppi_gpu.cuh`
  - Provides a workspace-aware `DEFINE_FORWARD_ROLLOUT_KERNEL` macro.

- `bi_mppi_gpu.cuh` / `src/bi_mppi_gpu.cu`
  - Adds GPU buffers for workspace boxes.
  - Uploads `CollisionChecker::workspace_boxes` to the device.
  - Passes workspace collision buffers to forward, backward, and guide kernels.

- `examples/manipulator_{mppi,logmppi,clustermppi,bicmppi}_workspace_example.cpp`
  - Register the same workspace box in both `ManipulatorDynamicsModel` and `CollisionChecker`.
  - Produce CSV logs with solver-specific prefixes.

- `examples/pinkNplace/manipulator_{mppi,logmppi,clustermppi,bicmppi}_pinkNplace_workspace_example.cpp`
  - Run the same sequential pick/place phase set for all four solvers.
  - Log phase name, gripper events, end-effector error, q error, solver time,
    and workspace-collision state with solver-specific prefixes.

## Collision model

For each state `x`, the GPU computes FK points:

```text
p_0 = base origin
p_i = FK_i(q), i=1..6
```

Each link segment `[p_{i-1}, p_i]` is sampled at 5 points. For each sampled point `p`, the collision distance is

```text
sd = dist_signed(p, AABB) - link_radius
```

A hard collision is triggered when

```text
sd < workspace_hard_margin
```

A soft cost is added when

```text
workspace_hard_margin <= sd < workspace_safe_margin
```

This is not a full capsule-AABB analytic distance, but it is fast, GPU-friendly, deterministic, and sufficient for a paper example if clearly described as sampled-link collision.

## Recommended paper wording

Use:

> We evaluate a 6-DOF second-order joint-space manipulator benchmark with workspace AABB obstacles. Each link is approximated by uniformly sampled points along the FK segment, and collision is evaluated by point-to-AABB distance with a link-radius margin.

Avoid:

> Full rigid-body manipulator collision checking.

unless you replace the sampled-link approximation with analytic capsule/mesh collision and use full rigid-body dynamics.

## Integration order

1. Back up your current repo.
2. Copy `include/collision_checker.h`, `include/cuda_legacy_model.cuh`, and `include/mppi_gpu.cuh` into your project include directory.
3. Patch or replace `bi_mppi_gpu.cuh` and `bi_mppi_gpu.cu` using the package versions.
4. Add the example target from `CMakeLists_fragment.txt`.
5. Build and run `manipulator_bicmppi_workspace_example`.

For this repository, the example is wired into the GPU build script:

```bash
bash build_gpu.sh manipulator_workspace_collision
cd build
./gpu/manipulator_mppi_workspace_example
./gpu/manipulator_logmppi_workspace_example
./gpu/manipulator_clustermppi_workspace_example
./gpu/manipulator_bicmppi_workspace_example
./gpu/manipulator_mppi_pinkNplace_workspace_example
./gpu/manipulator_logmppi_pinkNplace_workspace_example
./gpu/manipulator_clustermppi_pinkNplace_workspace_example
./gpu/manipulator_bicmppi_pinkNplace_workspace_example
```

## Expected outputs

```text
manipulator_workspace_mppi_summary.csv
manipulator_workspace_mppi_executed_x.csv
manipulator_workspace_mppi_executed_u.csv
manipulator_workspace_mppi_last_plan_x.csv
manipulator_workspace_logmppi_summary.csv
manipulator_workspace_clustermppi_summary.csv
manipulator_workspace_bicmppi_summary.csv
manipulator_workspace_bicmppi_executed_x.csv
manipulator_workspace_bicmppi_executed_u.csv
manipulator_workspace_bicmppi_last_plan_x.csv
manipulator_pinkNplace_mppi_summary.csv
manipulator_pinkNplace_mppi_executed_x.csv
manipulator_pinkNplace_logmppi_summary.csv
manipulator_pinkNplace_clustermppi_summary.csv
manipulator_pinkNplace_bicmppi_summary.csv
```

## Validation checklist

- Plot link positions for every executed state and verify no sampled point violates the AABB margin.
- Compare with q-space obstacle disabled; the trajectory should change when the workspace box is active.
- Run at least 50 random seeds for MPPI / Cluster-MPPI / BiC-MPPI before using the result as paper evidence.

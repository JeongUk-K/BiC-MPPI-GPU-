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

- `examples/manipulator_bicmppi_workspace_example.cpp`
  - Registers the same workspace box in both `ManipulatorDynamicsModel` and `CollisionChecker`.
  - Produces CSV logs with the prefix `manipulator_workspace_bicmppi_`.

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

## Expected outputs

```text
manipulator_workspace_bicmppi_summary.csv
manipulator_workspace_bicmppi_executed_x.csv
manipulator_workspace_bicmppi_executed_u.csv
manipulator_workspace_bicmppi_last_plan_x.csv
```

## Validation checklist

- Plot link positions for every executed state and verify no sampled point violates the AABB margin.
- Compare with q-space obstacle disabled; the trajectory should change when the workspace box is active.
- Run at least 50 random seeds for MPPI / Cluster-MPPI / BiC-MPPI before using the result as paper evidence.

## Pick-and-place workspace example

Additional executable:

```bash
./manipulator_bicmppi_pick_place_workspace_example
```

The pick-and-place example uses sequential target postures:

1. `home_to_pre_pick` with gripper open
2. `pre_pick_to_pick` and gripper close
3. `pick_to_lift`
4. `lift_to_transfer_high`
5. `transfer_to_pre_place`
6. `pre_place_to_place` and gripper release
7. `place_to_retreat` and gripper open

Workspace AABB obstacles are defined in `include/manipulator_pick_place_task.h`:

- central baffle between pick and place regions
- low walls around the pick bin
- low walls around the place bin

The gripper is represented as a discrete logged event only. Object grasping,
object dynamics, and gripper collision geometry are not included in the BiC-MPPI
state. This is intentional for the paper-level motion-planning benchmark: the
controlled plant remains the 6-DOF arm, while pick/release events segment the
motion into meaningful manipulation phases.

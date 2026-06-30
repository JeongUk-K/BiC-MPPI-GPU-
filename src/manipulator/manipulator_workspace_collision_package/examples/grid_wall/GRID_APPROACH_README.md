# Free-Opposite Grid Approach Scenario

This scenario replaces the earlier cell-to-cell initial condition.

## New task

```text
free_opposite -> free_aligned -> goal_retract -> goal_insert
```

The goal cell is sampled from the 3x3 grid.

## Files

```text
manipulator_grid_approach_task.h
manipulator_grid_approach_workspace_example.cpp
CODEX_GRID_APPROACH_SCENARIO_GUIDE.md
```

This uses the fixed `manipulator_grid_wall_task.h`.

## Run

Random goal:

```bash
./manipulator_grid_approach_workspace_example
```

Fixed goal:

```bash
./manipulator_grid_approach_workspace_example --goal 8
```

Fixed seed:

```bash
./manipulator_grid_approach_workspace_example --seed 7
```

## Validity checks

The following must be zero:

```text
free_opposite_coll
free_aligned_coll
goal_retract_coll
goal_insert_coll
```

# Manipulator rollout state archive

State archives contain every sampled rollout as
`[rollout, timestep, q1..q6, qdot1..qdot6]` in IEEE-754 little-endian
`float16`. The index records every iteration and branch, plus both compressed
and uncompressed byte offsets and sizes.

Each batch is an independent Zstandard frame, so a visualizer can seek to the
indexed compressed byte range and decompress only the requested iteration and
branch. `load_manipulator_rollout_state.py` implements this access pattern and
returns a float32 NumPy array.

The fixed random scenario selection generated on 2026-07-16 uses seed
`20260716` and scenario IDs:

```text
33,66,72,87,88,104,182,183,191,235
```


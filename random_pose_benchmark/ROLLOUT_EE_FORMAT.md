# Manipulator rollout EE archive

Each solver writes one compressed data stream and one CSV index per scenario:

```text
result/manipulator/rollout_ee/<solver>/
  scenario_00_ee_packed.bin.zst
  scenario_00_index.csv
```

After Zstandard decompression, the binary stream consists of consecutive
packed batches. The index gives the byte offset and size of every batch. For
each rollout, the first xyz point is three little-endian `int16` values (6
bytes), followed by three `int8` xyz deltas per remaining timestep. Apply a
time-axis cumulative sum, then multiply by the index's `coordinate_scale_m`
value to obtain metres. The generated archives use either 0.5 mm or 1 mm
resolution; consumers must read this field rather than hard-code a scale.

Forward solvers use the `forward` branch. BiC-MPPI records `forward`,
`backward`, and every guide batch as `guide_0`, `guide_1`, etc. Backward paths
are stored in chronological start-to-goal order.

`load_manipulator_rollout_ee.py` provides a NumPy loader for visualization.

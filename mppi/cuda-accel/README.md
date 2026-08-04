# Shared CUDA acceleration layer

`mppi/cuda-accel` is the active shared CUDA solver implementation used by
`build_gpu.sh`. Robot-specific programs should include the public solver
headers from this directory and link `libmppi_gpu.a`.

The BiC-MPPI K-means path keeps sampled controls and costs on the device:

1. forward/backward rollout kernels produce device controls and costs;
2. `gpu_kmeans.cuh` filters valid samples, builds features, clusters samples,
   and reduces every cluster to one weighted control sequence on the GPU;
3. only the small set of cluster representatives is copied to the CPU for
   connection construction;
4. guide rollout runs on the GPU and `gpu_control_reduction.cuh` reduces all
   guide samples to one control sequence before the device-to-host copy.

Connection search and representative trajectory validation remain on the CPU.
DBSCAN remains a compatibility path and therefore still copies its rollout
batch to the CPU.

The manipulator workspace-collision package retains robot-specific rollout
kernels, but consumes the same `gpu_kmeans.cuh` and
`gpu_control_reduction.cuh` device algorithms. The wmrobot executables consume
the complete shared `BiMPPI_GPU` implementation from this directory.

`mppi/cuda` is retained as the pre-acceleration compatibility implementation.

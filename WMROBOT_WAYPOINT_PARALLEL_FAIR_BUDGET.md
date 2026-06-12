# WMRobot Waypoint-Parallel BiC-MPPI Fair-Budget Experiment

이 문서는 `wmrobot_waypoint_parallel` 예제가 무엇을 검증하는지, waypoint 방식의 계산 구조가 어떻게 바뀌었는지, 그리고 fair sampling budget 조건에서 어떤 결과가 나왔는지 정리한다.

## 목적

BiC-MPPI는 start와 goal 사이의 양방향 rollout을 이용해 연결 가능한 경로를 찾고, 이후 guide MPPI로 경로를 다듬는다. 이 예제는 전체 맵을 한 번에 푸는 global BiC-MPPI 대신, 미리 주어진 waypoint 사이 구간들을 병렬로 탐색한 뒤 마지막 guide MPPI를 한 번 수행하는 방식이 더 빠르게 동작할 수 있는지 확인한다.

검증 대상은 다음 구조다.

```text
Global BiC-MPPI:
  whole map start -> goal
  forward rollout + backward rollout + guide MPPI

Waypoint-Parallel BiC:
  waypoint segment들을 동시에 탐색
  각 segment는 forward/backward connect까지만 수행
  전체 이어붙인 경로에 대해 final guide MPPI 1회 수행
```

## 실험 환경

예제 맵은 5개의 방을 이어 붙인 형태이며, 각 방 사이에는 portal waypoint가 있다.

```text
start -> wp1 -> wp2 -> wp3 -> wp4 -> goal
```

주요 파라미터는 다음과 같다.

```text
dt = 0.1
global_T = 225
global_N = 3000
segment_T = 45
segment_N = 3000
segment_Nr = 0
guide_N = 3000
num_segments = 5
goal_tolerance = 0.55
seed = 7
```

`segment_Nr=0`이 중요한 점이다. waypoint segment에서는 더 이상 guide rollout을 하지 않는다. 각 segment는 forward/backward rollout과 connection만 수행하고, `Nr` guide는 마지막 full-path guide에서 한 번만 수행한다.

## Fair Budget 정의

이전 구현에서는 각 waypoint segment가 `Tf=45`, `Tb=45`를 모두 사용했다. 그러면 segment 하나가 `90` horizon-step을 쓰고, 5개 segment는 `450` step을 사용한다. 여기에 final guide `225` step이 추가되어 global BiC보다 계산량이 커졌다.

현재 fair-budget 구현은 segment 하나의 forward/backward 합이 `segment_T`가 되도록 나눈다.

```text
segment_T = 45
Tf = segment_T / 2 = 22
Tb = segment_T - Tf = 23
Tf + Tb = 45
```

따라서 global BiC와 waypoint 방식의 configured sample-step이 같아진다.

```text
Global BiC-MPPI:
  forward  = 3000 * 112
  backward = 3000 * 113
  guide    = 3000 * 225
  total    = 1,350,000

Waypoint-Parallel BiC:
  segments = 5 * 3000 * (22 + 23)
  guide    = 3000 * 225
  total    = 1,350,000
```

즉 두 방식은 모두 `1,350,000` configured sample-step을 사용한다.

## 구현 위치

주요 구현 파일은 다음과 같다.

```text
src/wmrobot/gpu_waypoint/comparison.cpp
run_wmrobot_waypoint_parallel.sh
visualize_wmrobot_waypoint_parallel.py
animate_wmrobot_waypoint_sampling.py
```

핵심 구현은 `comparison.cpp`에 있다.

```text
segmentBiHorizons()
  segment_T를 Tf/Tb로 나눠 Tf + Tb = segment_T가 되게 한다.

ConnectOnlyBiMPPI
  segment에서 forward/backward rollout, connection, concatenate까지만 수행한다.
  segment별 guide MPPI는 수행하지 않는다.

ReferenceGuideBiMPPI
  이어붙인 전체 경로에 대해 final guide MPPI를 한 번 수행한다.
```

## 실행 방법

기본 실행:

```bash
./run_wmrobot_waypoint_parallel.sh --overwrite
```

sampling GIF와 timestep PNG까지 생성:

```bash
./run_wmrobot_waypoint_parallel.sh --overwrite --sample-gif
```

빌드 없이 다시 실행:

```bash
./run_wmrobot_waypoint_parallel.sh --no-build --overwrite
```

serial segment 비교:

```bash
./run_wmrobot_waypoint_parallel.sh --no-build --no-viz --overwrite --serial-segments --out ../results/wmrobot_waypoint_fair_budget_time_serial
```

parallel segment 비교:

```bash
./run_wmrobot_waypoint_parallel.sh --no-build --no-viz --overwrite --out ../results/wmrobot_waypoint_fair_budget_time_parallel
```

## 결과

시각화 로그를 끈 fair-budget 시간 비교 결과는 다음 파일에 저장되어 있다.

```text
results/wmrobot_waypoint_fair_budget_time_compare.csv
```

대표 결과:

| Method | Success | Configured sample-steps | Wall time (s) | Final error |
|---|---:|---:|---:|---:|
| MPPI | 0 | 675,000 | 0.002208616 | 6.902220 |
| Cluster-MPPI | 0 | 675,000 | 0.027893273 | 7.121280 |
| BiC-MPPI | 0 | 1,350,000 | 0.051108126 | 6.202752 |
| Waypoint-Parallel BiC | 1 | 1,350,000 | 0.014855365 | 0.157291 |

같은 configured sample-step 조건에서 waypoint 방식은 global BiC-MPPI보다 빠르게 성공했다.

```text
Global BiC-MPPI wall time        = 0.051108126 s
Waypoint-Parallel BiC wall time  = 0.014855365 s
Speedup                          = 3.44x

Waypoint effective parallel time = 0.012547723 s
Effective speedup                = 4.07x
```

serial segment와 parallel segment의 waypoint 결과:

```text
Waypoint serial wall time        = 0.016781916 s
Waypoint parallel wall time      = 0.014855365 s
Parallel speedup vs serial       = 1.13x
```

## 해석

Waypoint 방식이 빨라진 이유는 두 가지다.

1. 각 segment의 horizon이 짧다.
   - global BiC는 긴 전체 horizon에서 connection을 찾는다.
   - waypoint 방식은 waypoint 사이의 짧은 구간에서 connection을 찾는다.

2. segment 탐색이 병렬로 수행된다.
   - `std::async`로 5개 segment를 동시에 launch한다.
   - 각 segment는 guide 없이 connect-only BiC를 수행한다.
   - 마지막에 full-path guide MPPI를 한 번만 수행한다.

다만 parallel speedup이 이상적인 5배로 나오지는 않는다. 모든 segment가 같은 GPU를 사용하므로 CUDA kernel 실행과 synchronization에서 자원 경합이 발생한다. 따라서 이 예제는 multi-GPU 또는 CUDA stream/batched kernel 구조가 아니라, 현재 단일 GPU 코드 구조에서 가능한 병렬화 효과를 보여준다.

## 시각화 자료

기본 경로 시각화:

```text
results/wmrobot_waypoint_parallel/wmrobot_waypoint_paths.png
```

시간 비교:

```text
results/wmrobot_waypoint_parallel/wmrobot_waypoint_timing.png
results/wmrobot_waypoint_parallel/wmrobot_waypoint_runtime_breakdown.png
```

sampling 과정 GIF:

```text
results/wmrobot_waypoint_parallel/wmrobot_waypoint_sampling.gif
```

각 timestep PNG:

```text
results/wmrobot_waypoint_parallel/sampling_frames/frame_*.png
```

GIF는 variant별 2x2 panel로 구성된다.

```text
MPPI
Cluster-MPPI
BiC-MPPI
Waypoint-Parallel BiC
```

투명한 선은 sampled rollout 또는 candidate path이고, 굵은 선은 각 method가 선택한 plan prefix다.

## 주의점

시간 비교에는 `--sample-gif` 실행 결과를 사용하면 안 된다. 이 옵션은 solver 내부 trajectory를 host로 복사하고 파일로 저장하므로 측정 시간이 커진다. 시간 비교는 반드시 `--no-viz` 또는 기본 CSV 중 시각화 로그가 꺼진 결과를 사용해야 한다.

또한 결과 시간은 GPU 상태, driver scheduling, 동시 실행 부하에 따라 조금씩 달라질 수 있다. 이 문서의 결론은 절대 시간보다 같은 configured sample-step에서의 상대 비교에 초점을 둔다.

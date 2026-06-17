# WMRobot Waypoint Sequential Code Flow Review

이 문서는 GPT에게 현재 `gpu_waypoint_순차추종` 코드 흐름을 검증받기 위한 설명이다.

## 핵심 요약

현재 예제는 5개의 BARN map을 이어 붙인 stitched map에서 WMRobot이 waypoint를 따라 이동하는 실험이다.

비교 대상은 네 solver다.

```text
MPPI
Log-MPPI
Cluster-MPPI
BiC-MPPI
```

SVGD는 이 sequential waypoint 실험에서 제외되어 있다.

현재 파라미터 수정 위치는 하나다.

```text
src/wmrobot/gpu_waypoint_순차추종/stitched_barn_common.h
```

`stitched_barn_params.h`는 더 이상 사용하지 않는다.

## 실행 명령

기본 실행:

```bash
./run_wmrobot_waypoint_stitched_sequential.sh --overwrite
```

짧은 검증 실행:

```bash
./run_wmrobot_waypoint_stitched_sequential.sh \
  --no-build \
  --overwrite \
  --out results/wmrobot_waypoint_seq_check \
  --scenarios 1 \
  --maxiter 2 \
  --waypoint-count 4 \
  --N 32 \
  --Nf 32 \
  --Nb 32 \
  --Nr 16 \
  --global-T 20 \
  --bic-Tf 6 \
  --bic-Tb 6 \
  --vis-every 1 \
  --max-rollouts 8 \
  --gif-fps 2
```

## Shell 흐름

파일:

```text
run_wmrobot_waypoint_stitched_sequential.sh
```

동작 순서:

1. repository root로 이동.
2. 필요하면 `bash build_gpu.sh wmrobot_waypoint_stitched_sequential` 실행.
3. 아래 binary가 있는지 확인.

```text
build/gpu/wmrobot_waypoint_seq_mppi
build/gpu/wmrobot_waypoint_seq_log_mppi
build/gpu/wmrobot_waypoint_seq_cluster_mppi
build/gpu/wmrobot_waypoint_seq_bi_mppi
```

4. 기존 `build/vis_data/stitched_*_scenario_*` 삭제.
5. 네 solver binary를 순서대로 실행.
6. `build/vis_data`를 output directory 아래 `vis_data`로 복사.
7. `visualize_wmrobot_waypoint_stitched.py`를 실행해 CSV/PNG/GIF를 만든다.

기본 output root:

```text
results/wmrobot_waypoint_stitched_sequential
```

## 파라미터 구조

파일:

```text
src/wmrobot/gpu_waypoint_순차추종/stitched_barn_common.h
```

공통 파라미터는 `Config` struct에 있다.

```text
out_dir = ../results/wmrobot_waypoint_stitched_sequential
dataset_dir = ../BARN_dataset/txt_files
scenarios = 10
maps_per_scenario = 5
dataset_maps = 300
maxiter = 500
global_T = 500
waypoint_count = 6
bic_Tf = 50
bic_Tb = 50
N = 3000
Nf = 3000
Nb = 3000
Nr = 1500
vis_every = 1
dt = 0.1
resolution = 0.1
waypoint_tolerance = 0.45
start_x = 1.5
map_length = 5.0
seed = 7
```

solver별 파라미터는 같은 파일 바로 아래 함수에서 수정한다.

```cpp
makeMppiConfig()
makeLogMppiConfig()
makeClusterMppiConfig()
makeBiMppiConfig()
```

중요: `--waypoint-count`, `--global-T`, `--bic-Tf`, `--bic-Tb`, `--N`, `--Nf`, `--Nb`, `--Nr`, `--maxiter`, `--vis-every` 등 CLI 옵션은 C++ 기본값을 override한다.

## Entry Point

각 cpp 파일은 solver type과 config factory만 선택한다.

```text
mppi.cpp
  runMppiLikeExecutable<MPPI_GPU>(..., makeMppiConfig())

log_mppi.cpp
  runMppiLikeExecutable<LogMPPI_GPU>(..., makeLogMppiConfig())

cluster_mppi.cpp
  runMppiLikeExecutable<ClusterMPPI_GPU>(..., makeClusterMppiConfig())

bi_mppi.cpp
  runBiExecutable(..., makeBiMppiConfig())
```

모든 cpp는 다음 하나만 include한다.

```cpp
#include "stitched_barn_common.h"
```

## Scenario 생성

함수:

```cpp
makeScenario()
```

흐름:

1. `0..dataset_maps-1` map id 목록 생성.
2. `seed + scenario_index`로 shuffle.
3. 앞에서 `maps_per_scenario`개 선택.
4. `BARN_dataset/txt_files/output_<id>.txt` 로드.
5. 모든 map 크기가 같은지 검사.
6. column 방향으로 이어 붙임.

stitching code:

```cpp
scenario.cols = cols * maps.size();
scenario.map[r][m * cols + c] = maps[m][r][c];
```

waypoint path는 `y` 방향으로 증가한다. visualization에서는 map을 transpose해서 보여준다.

## Waypoint 생성

함수:

```cpp
makeWaypoints()
```

시작:

```text
x = start_x
y = 0
theta = pi / 2
```

목표:

```text
x = start_x
y = map_length * maps_per_scenario
theta = pi / 2
```

중간 waypoint:

```cpp
alpha = i / (waypoint_count - 1)
y = total_length * alpha
```

현재 기본 `waypoint_count`는 6이다. 55개 waypoint 실험이 목적이면 `Config::waypoint_count`를 바꾸거나 실행 시 `--waypoint-count 55`를 줘야 한다.

## MPPI / Log-MPPI / Cluster-MPPI 흐름

함수:

```cpp
runMppiLikeScenario()
```

각 closed-loop iteration에서:

1. 현재 state collision 검사.
2. rollout logging 여부 설정.
3. `solver.x_target = waypoints[target_index]`.
4. `solver.solve()`.
5. `solver.move()`.
6. elapsed time 누적.
7. `paths.csv`에 state 추가.
8. move 이후 collision 검사.
9. 현재 waypoint에 가까우면 `target_index++`.
10. waypoint가 바뀌면 `U_0` warm start를 zero reset.
11. 모든 waypoint에 도달하면 success.

즉 이 세 solver는 한 번에 final goal로 가지 않고, 현재 waypoint 하나를 순차 추종한다.

metadata:

```json
"target_mode": "sequential_waypoints",
"rollout_mode": "single_direction_parallel_gpu_rollouts"
```

## BiC-MPPI 흐름

함수:

```cpp
runBiScenario()
WaypointBatchBiMPPI_GPU::solveWaypointGraph()
```

BiC-MPPI는 MPPI 계열과 다르게 현재 waypoint 하나만 solve하지 않는다.

각 closed-loop iteration에서 남은 waypoint 전체를 anchor list로 만든다.

```cpp
anchors = {
  solver.x_init,
  waypoints[target_index],
  waypoints[target_index + 1],
  ...,
  waypoints.back()
};
```

그 다음:

```cpp
solver.solveWaypointGraph(anchors);
solver.move();
```

`solveWaypointGraph()` 내부 흐름:

1. 각 adjacent anchor pair를 segment로 본다.

```text
anchors[0] -> anchors[1]
anchors[1] -> anchors[2]
...
```

2. 각 segment마다:

```text
x_init = segment start
x_target = segment target
U_f0 = zero
U_b0 = zero
backwardRollout()
forwardRollout()
selectConnection()
concatenate()
best connected reference 선택
```

3. segment별 reference를 하나의 global reference로 이어 붙인다.

```text
Uref = [U_segment_0, U_segment_1, ...]
Xref = [X_segment_0, X_segment_1, ...]
```

4. 전체 reference에 대해 한 번 global guide를 수행한다.

```cpp
guideReference(Uref, Xref)
```

5. `partitioningControl()`로 guide 결과의 첫 구간을 다음 warm start로 반영한다.
6. `move()`가 첫 control을 적용해 robot state를 한 step 전진시킨다.

metadata:

```json
"target_mode": "multi_waypoint_bidirectional_connections",
"rollout_mode": "all_remaining_waypoint_segments_forward_backward_gpu_rollouts_then_global_guide"
```

## BiMPPI GPU Core 변경점

파일:

```text
mppi/cuda/bi_mppi_gpu.cuh
mppi/cuda/bi_mppi_gpu.cu
```

추가된 public API:

```cpp
void guideReference(const Eigen::MatrixXd &Uref,
                    const Eigen::MatrixXd &Xref);
```

추가/변경된 buffer 관리:

```cpp
void allocGuideFor(int Tr);
```

기존 guide buffer는 `Tf + Tb` 기준이었다. 현재는 waypoint segment들을 이어 붙인 긴 reference `Tr`에 맞춰 guide buffer를 재할당할 수 있다.

주의: waypoint segment들은 host loop에서 순차적으로 처리된다. 각 segment 내부 forward/backward rollout sample과 global guide rollout sample은 GPU 병렬이다.

## Success / Failure 판정

collision 판정:

```cpp
collision_checker.getCollisionGrid(solver.x_init)
```

검사 위치:

```text
closed-loop iteration 시작
solver.move() 이후
```

success 조건:

1. 모든 waypoint를 순차적으로 reach:

```cpp
target_index >= waypoints.size()
```

2. 또는 collision 없이 final goal 근처:

```cpp
final_error <= waypoint_tolerance
```

검토 포인트: strict waypoint following 성공 기준이 필요하면 final proximity fallback은 제거하거나 별도 metric으로 분리하는 것이 더 명확하다.

## 출력 파일

solver별 output:

```text
<out_dir>/mppi/
<out_dir>/log_mppi/
<out_dir>/cluster_mppi/
<out_dir>/bi_mppi/
```

각 solver directory:

```text
metadata.json
summary.csv
paths.csv
timings.csv
maps.csv
waypoints.csv
stitched_maps/scenario_XX.csv
```

global visualization output:

```text
<out_dir>/wmrobot_stitched_aggregate.csv
<out_dir>/wmrobot_stitched_timing.png
<out_dir>/wmrobot_stitched_paths_scenario_XX.png
<out_dir>/wmrobot_stitched_rollouts_scenario_XX.png
<out_dir>/rollout_gifs/wmrobot_stitched_<solver>_scenario_XX.gif
```

rollout log:

```text
<out_dir>/vis_data/stitched_<solver>_scenario_XX/step_YYYY/
```

step directory에는 solver에 따라 다음 파일이 생긴다.

```text
rollouts.bin
clusters.bin
forward_clusters.bin
backward_clusters.bin
guide_candidates.bin
optimal.bin
x_pos.bin
```

`timings.csv`는 폐루프 iteration마다 solver 내부 단계별 계산시간을 기록한다.

```text
scenario
solver
step
target_waypoint
reached_waypoints
total_waypoints
elapsed
elapsed_rollout
elapsed_clustering
elapsed_connection
elapsed_guide
cumulative_elapsed
cumulative_rollout
cumulative_clustering
cumulative_connection
cumulative_guide
x
y
theta
```

`step`은 `paths.csv`의 이동 후 state index와 맞춘다. 즉 `step=0`은 초기 상태이고,
`timings.csv`의 첫 row는 첫 번째 `solve() -> move()` 뒤의 `step=1`이다.

## GIF 시각화 흐름

파일:

```text
visualize_wmrobot_waypoint_stitched.py
```

GIF 생성 함수:

```cpp
make_solver_rollout_path_gif(...)
```

GIF는 solver별로 한 파일씩 생성한다. 기본 `scenarios = 10`이고 solver가 4개이므로
full visualization에서는 `10 * 4 = 40`개의 GIF가 생성된다. Static rollout PNG도
모든 scenario에 대해 `wmrobot_stitched_rollouts_scenario_XX.png`로 생성된다.

시각화는 stitched 진행 방향을 가로로 표시한다. plot 좌표는 `plot_x = state_y`,
`plot_y = state_x`로 변환되며, map도 같은 방향으로 눕혀서 표시한다.

표시 규칙:

```text
검은 점선/점: waypoints
진한 회색 선: rollout / guide candidate 경로
진한 오렌지 선: clustering으로 선별된 cluster 경로
진한 빨강 굵은 선: 해당 step의 optimal 탐색 경로
검은 선/점: 지금까지 실제 실행된 closed-loop 경로
```

`--max-rollouts`는 각 GIF frame에 표시하는 sampled trajectory 수를 제한한다. 실제 저장된 rollout이 많아도 stride sampling으로 줄여서 표시한다.

## GPT에게 확인받고 싶은 질문

1. MPPI/Log/Cluster는 current waypoint 하나만 target으로 두고, BiC-MPPI는 남은 waypoint segment 전체를 reference로 guide한다. 이 비교가 공정한가?
2. BiC-MPPI의 segment processing은 host에서 순차 dispatch된다. 진짜 "각 waypoint에서 양방향 rollout을 동시에"라는 실험 의도라면 segment별 CUDA stream 병렬화가 필요한가?
3. BiC-MPPI에서 각 segment마다 `U_f0`, `U_b0`를 zero reset하는 것이 적절한가, 아니면 segment 간 warm start를 전달해야 하는가?
4. success 조건에 final goal proximity fallback이 있는 것이 waypoint sequential tracking 평가에 적절한가?
5. 기본 `waypoint_count = 6`이 실험 목적에 맞는가? 원래 의도가 55 waypoint라면 기본값을 55로 바꿔야 하는가?
6. `vis_every = 1`은 GIF에는 좋지만 full experiment 저장량이 크다. 기본값으로 적절한가?
7. BiC-MPPI의 global guide horizon은 이어 붙인 segment reference 길이이다. MPPI 계열의 `global_T = 500`과 계산 budget 비교가 타당한가?
8. DBSCAN clustering이 CPU에서 실행되므로 BiC timing의 bottleneck이 될 수 있다. GPU rollout 병렬성만으로 충분한 비교인가?

## 현재 검증된 항목

다음 명령은 통과했다.

```bash
bash -n run_wmrobot_waypoint_stitched_sequential.sh
bash build_gpu.sh wmrobot_waypoint_stitched_sequential
./run_wmrobot_waypoint_stitched_sequential.sh \
  --no-build \
  --no-viz \
  --overwrite \
  --out results/wmrobot_waypoint_seq_single_header_check \
  --scenarios 1 \
  --maxiter 1 \
  --waypoint-count 4 \
  --N 16 \
  --Nf 16 \
  --Nb 16 \
  --Nr 8 \
  --global-T 10 \
  --bic-Tf 4 \
  --bic-Tb 4 \
  --no-rollouts
```

GIF 생성도 별도 짧은 실행에서 확인됐다.

```text
results/wmrobot_waypoint_seq_gif_check/wmrobot_stitched_rollouts_scenario_00.gif
```

# WMRobot Waypoint Sequential Current Code Analysis

작성 기준: 현재 repository의 `gpu_waypoint_순차추종` 구현.

이 문서는 `run_wmrobot_waypoint_stitched_sequential.sh`를 실행했을 때 어떤 코드 경로가 동작하고, 결과/경로/GIF가 어떻게 생성되는지 설명한다.

## 대상 파일

```text
run_wmrobot_waypoint_stitched_sequential.sh
build_gpu.sh
src/wmrobot/gpu_waypoint_순차추종/stitched_barn_common.h
src/wmrobot/gpu_waypoint_순차추종/mppi.cpp
src/wmrobot/gpu_waypoint_순차추종/log_mppi.cpp
src/wmrobot/gpu_waypoint_순차추종/cluster_mppi.cpp
src/wmrobot/gpu_waypoint_순차추종/bi_mppi.cpp
mppi/cuda/bi_mppi_gpu.cuh
mppi/cuda/bi_mppi_gpu.cu
visualize_wmrobot_waypoint_stitched.py
```

## 실행 방법

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

`--no-viz`를 주지 않으면 실행 후 자동으로 PNG와 GIF를 생성한다.

## Shell 동작 흐름

`run_wmrobot_waypoint_stitched_sequential.sh`는 다음 순서로 동작한다.

1. 필요하면 `bash build_gpu.sh wmrobot_waypoint_stitched_sequential` 실행.
2. 아래 binary 존재 여부 확인:

```text
build/gpu/wmrobot_waypoint_seq_mppi
build/gpu/wmrobot_waypoint_seq_log_mppi
build/gpu/wmrobot_waypoint_seq_cluster_mppi
build/gpu/wmrobot_waypoint_seq_bi_mppi
```

3. 기존 `build/vis_data/stitched_*_scenario_*` 삭제.
4. 네 solver binary를 차례대로 실행.
5. `build/vis_data`의 rollout log를 output directory의 `vis_data`로 복사.
6. `visualize_wmrobot_waypoint_stitched.py` 실행.
7. aggregate CSV, PNG, GIF 경로를 출력.

## 주요 기본 파라미터

기본값은 `src/wmrobot/gpu_waypoint_순차추종/stitched_barn_common.h`에서 관리한다.

```text
scenarios = 10
maps_per_scenario = 5
dataset_maps = 300
seed = 7
maxiter = 500
dt = 0.1
resolution = 0.1
start_x = 1.5
map_length = 5.0
waypoint_count = 6
waypoint_tolerance = 0.45
save_rollouts = true
vis_every = 1
```

MPPI 계열:

```text
global_T = 500
N = 3000
```

BiC-MPPI:

```text
bic_Tf = 50
bic_Tb = 50
Nf = 3000
Nb = 3000
Nr = 1500
```

CLI에서 `--waypoint-count`를 주면 C++ 기본값이 다시 override된다.

## Map Stitching

각 scenario마다 BARN map 300개 중 `maps_per_scenario`개를 random selection한다.

```cpp
std::mt19937 rng(seed + scenario_index);
std::shuffle(ids.begin(), ids.end(), rng);
ids.resize(maps_per_scenario);
```

선택된 map들은 `BARN_dataset/txt_files/output_<id>.txt`에서 읽는다.

stitching은 map column 방향으로 붙인다.

```cpp
scenario.cols = cols * maps.size();
scenario.map[r][m * cols + c] = maps[m][r][c];
```

시각화에서는 map을 transpose해서 그리므로, 결과적으로 robot이 `y`축 방향으로 길어진 stitched map을 진행하는 형태가 된다.

## Waypoint 생성

waypoint는 `makeWaypoints()`에서 만든다.

시작점:

```text
x = start_x
y = 0
theta = pi / 2
```

최종 목표:

```text
x = start_x
y = map_length * maps_per_scenario
theta = pi / 2
```

중간 waypoint는 시작점과 최종 목표 사이에 균등 배치한다.

```cpp
alpha = i / (waypoint_count - 1)
y = total_length * alpha
```

## MPPI / Log-MPPI / Cluster-MPPI 동작

세 solver는 모두 `runMppiLikeScenario()`를 사용한다.

closed-loop iteration마다:

1. 현재 상태 collision 검사.
2. `solver.x_target = waypoints[target_index]`.
3. `solver.solve()`.
4. `solver.move()`.
5. 누적 time 기록.
6. `paths.csv`에 현재 state 추가.
7. 다시 collision 검사.
8. 현재 waypoint와의 거리가 `waypoint_tolerance` 이하이면 `target_index++`.
9. waypoint가 바뀌면 `U_0` warm start를 zero로 reset.
10. 모든 waypoint에 도달하면 success.

즉 MPPI, Log-MPPI, Cluster-MPPI는 각 step에서 "현재 waypoint 하나"만 target으로 두고 순차 추종한다.

metadata:

```json
"target_mode": "sequential_waypoints",
"rollout_mode": "single_direction_parallel_gpu_rollouts"
```

## BiC-MPPI 현재 동작

이전 구현에서는 BiC-MPPI도 `waypoints[target_index]` 하나만 target으로 두고 순차 추종했다.

현재 구현은 `WaypointBatchBiMPPI_GPU`를 사용한다. closed-loop iteration마다 현재 상태와 남은 waypoint 전체를 anchor list로 만든다.

```cpp
anchors = {
  solver.x_init,
  waypoints[target_index],
  waypoints[target_index + 1],
  ...,
  waypoints.back()
};
solver.solveWaypointGraph(anchors);
solver.move();
```

`solveWaypointGraph()` 내부에서는 각 adjacent anchor pair를 하나의 segment로 본다.

```text
anchors[0] -> anchors[1]
anchors[1] -> anchors[2]
...
anchors[n-2] -> anchors[n-1]
```

각 segment마다 수행하는 작업:

1. `x_init = segment_start`
2. `x_target = segment_goal`
3. `U_f0`, `U_b0` zero reset
4. `backwardRollout()`
5. `forwardRollout()`
6. clustering
7. `selectConnection()`
8. `concatenate()`
9. feasible connected reference 중 가장 좋은 reference 선택

그 후 segment별 reference들을 하나의 full-route reference로 이어 붙인다.

```text
Uref = [U_segment_0, U_segment_1, ...]
Xref = [X_segment_0, X_segment_1, ...]
```

마지막으로 `BiMPPI_GPU::guideReference(Uref, Xref)`가 전체 남은 waypoint route를 한 번에 guide한다.

metadata:

```json
"target_mode": "multi_waypoint_bidirectional_connections",
"rollout_mode": "all_remaining_waypoint_segments_forward_backward_gpu_rollouts_then_global_guide"
```

## BiC 병렬성 범위

현재 BiC-MPPI의 병렬성은 다음과 같다.

GPU 병렬:

```text
각 segment의 forward rollout samples
각 segment의 backward rollout samples
global guide rollout samples
```

CPU/host 순차:

```text
남은 waypoint segment들을 하나씩 순회
segment별 rollout kernel dispatch
DBSCAN clustering
connection selection
segment reference concatenation
```

즉 "각 waypoint segment가 동시에 하나의 kernel batch로 전부 실행"되는 구조는 아니다. 다만 한 closed-loop solve 안에서 모든 남은 waypoint segment를 처리하고, 이어 붙인 full reference를 global guide한다.

## BiMPPI Core 변경

`mppi/cuda/bi_mppi_gpu.cuh`에 public API가 추가됐다.

```cpp
void guideReference(const Eigen::MatrixXd &Uref,
                    const Eigen::MatrixXd &Xref);
```

`mppi/cuda/bi_mppi_gpu.cu`에서는 guide buffer가 기존 `Tf + Tb` 고정 길이에서 reference 길이 `Tr`에 맞춰 확장되도록 변경됐다.

```cpp
void BiMPPI_GPU::allocGuideFor(int Tr)
```

이 변경 때문에 BiC waypoint batch에서 `Tf + Tb`보다 긴 full-route reference를 guide할 수 있다.

## Collision / Success 조건

collision은 closed-loop iteration 시작과 `move()` 후에 검사한다.

```cpp
collision_checker.getCollisionGrid(solver.x_init)
```

success는 두 방식 중 하나로 설정된다.

1. `target_index >= waypoints.size()`
2. collision 없이 final state가 final goal에서 `waypoint_tolerance` 이내

```cpp
stats.success = stats.success ||
                (!stats.is_collision &&
                 stats.final_error <= cfg.waypoint_tolerance);
```

이 때문에 중간 waypoint 도달 여부와 별개로 final goal 근처에 있으면 success가 될 수 있다.

## Output 구조

기본 output root:

```text
results/wmrobot_waypoint_stitched_sequential
```

solver별 output:

```text
mppi/
log_mppi/
cluster_mppi/
bi_mppi/
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

visualization output:

```text
vis_data/
wmrobot_stitched_aggregate.csv
wmrobot_stitched_timing.png
wmrobot_stitched_paths_scenario_XX.png
wmrobot_stitched_rollouts_scenario_XX.png
rollout_gifs/wmrobot_stitched_<solver>_scenario_XX.gif
```

## CSV 의미

`summary.csv`:

```text
solver
solver_label
scenario
map_ids
success
is_collision
iter
reached_waypoints
total_waypoints
elapsed
elapsed_rollout
elapsed_clustering
elapsed_connection
elapsed_guide
final_error
path_points
```

`paths.csv`:

```text
scenario
solver
step
x
y
theta
```

`timings.csv`:

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

`elapsed_*`는 해당 폐루프 iteration의 solver 내부 단계별 시간이고,
`cumulative_*`는 같은 scenario 안에서 그 시점까지 누적된 시간이다.
`step=1`은 첫 번째 `solve() -> move()` 이후의 state이며 `paths.csv`의 `step=1`과 대응된다.

`maps.csv`:

```text
scenario
slot
map_id
col_begin
col_end
```

`waypoints.csv`:

```text
scenario
index
x
y
theta
```

## Rollout Logging

`vis_every = 1`이므로 기본적으로 모든 closed-loop iteration에서 rollout log가 저장된다.

저장 위치:

```text
build/vis_data/stitched_<solver>_scenario_XX/step_YYYY/
```

runner가 이를 실행 후 output directory로 복사한다.

```text
<out_dir>/vis_data/stitched_<solver>_scenario_XX/step_YYYY/
```

step directory에는 solver에 따라 다음 binary 파일들이 생긴다.

```text
rollouts.bin
clusters.bin
forward_clusters.bin
backward_clusters.bin
guide_candidates.bin
optimal.bin
x_pos.bin
```

모든 solver가 모든 파일을 쓰는 것은 아니다. visualization script는 존재하는 파일만 읽는다.

## GIF 시각화

`visualize_wmrobot_waypoint_stitched.py`는 solver별/scenario별 GIF를 생성한다.

```text
rollout_gifs/wmrobot_stitched_<solver>_scenario_XX.gif
```

기본 `scenarios = 10`에서 solver가 4개이므로 full visualization에서는
`10 * 4 = 40`개 GIF가 생성된다. Static rollout PNG도 모든 scenario에 대해
`wmrobot_stitched_rollouts_scenario_XX.png`로 생성된다.

그림과 GIF는 stitched 진행 방향이 화면 가로축이 되도록 그린다. 즉 실제 state의
`y` 좌표를 plot `x`축으로, 실제 state의 `x` 좌표를 plot `y`축으로 변환한다.

표시 방식:

```text
검은 점선/점: waypoint
진한 회색 선: rollout / guide candidate 경로
진한 오렌지 선: clustering으로 선별된 cluster 경로
진한 빨강 굵은 선: 해당 step의 optimal 탐색 경로
검은 선/점: 지금까지 실제 실행된 closed-loop 경로
```

GIF 관련 shell option:

```text
--rollout-scenario N
--max-rollouts N
--gif-fps N
```

`--rollout-scenario`는 호환성을 위해 남겨둔 옵션이다. 현재 static rollout PNG와 GIF는 모든 scenario에 대해 생성된다.
`--max-rollouts`는 각 GIF frame에 표시할 sampled trajectory 개수를 제한한다. 실제 saved rollout 수가 많아도 GIF에서는 stride sampling으로 줄여서 그린다.

## 성능 및 저장 용량 주의점

`vis_every = 1`은 모든 step의 rollout을 저장하므로 full experiment에서는 파일 수와 용량이 커질 수 있다.

저장량을 줄이려면:

```bash
--vis-every 10
```

GIF 표시량을 줄이려면:

```bash
--max-rollouts 8 --gif-fps 3
```

완전히 시각화를 끄려면:

```bash
--no-viz
```

rollout 저장 자체를 끄려면:

```bash
--no-rollouts
```

## 현재 구조에서 중요한 검토 포인트

1. `waypoint_count`의 실제 기본값은 `6`이다. 실험 의도가 55 waypoint라면 `stitched_barn_common.h`를 바꾸거나 실행 시 `--waypoint-count 55`를 줘야 한다.
2. BiC-MPPI는 모든 남은 waypoint segment를 한 closed-loop solve에서 처리하지만, segment dispatch 자체는 host에서 순차 수행된다.
3. BiC-MPPI의 DBSCAN clustering과 connection selection은 CPU에서 수행된다.
4. MPPI/Log/Cluster는 current waypoint 하나만 target으로 두는 반면, BiC는 남은 route 전체를 reference로 guide한다. 비교 공정성은 별도 판단이 필요하다.
5. success 조건은 final goal proximity로도 true가 될 수 있으므로, "모든 waypoint 순차 통과"만 성공으로 보고 싶다면 success 조건을 더 엄격하게 바꿔야 한다.
6. full run에서 GIF까지 만들면 시간이 늘어난다. 대규모 실험에서는 `--no-viz`로 solver 결과만 만들고 나중에 visualization을 따로 실행하는 방식이 더 낫다.

## 검증된 명령

아래 명령으로 문법, build, 짧은 실행, GIF 생성이 확인됐다.

```bash
python3 -m py_compile visualize_wmrobot_waypoint_stitched.py
bash -n run_wmrobot_waypoint_stitched_sequential.sh
bash build_gpu.sh wmrobot_waypoint_stitched_sequential
./run_wmrobot_waypoint_stitched_sequential.sh \
  --no-build \
  --overwrite \
  --out results/wmrobot_waypoint_seq_gif_check \
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

검증 결과 GIF:

```text
results/wmrobot_waypoint_seq_gif_check/rollout_gifs/wmrobot_stitched_mppi_scenario_00.gif
```

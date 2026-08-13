# WMRobot Stitched Sequential Waypoint 실험 방법과 해석

## 1. 문서의 대상

이 문서는 다음 결과의 생성 방법과 올바른 해석을 설명한다.

```text
results/wmrobot_waypoint_stitched_sequential/
```

대응하는 구현은 다음 디렉터리에 있다.

```text
src/wmrobot/gpu_waypoint_순차추종/
```

이 실험은 이름이 비슷한
`wmrobot_waypoint_stitched_sequential_variants` 실험과는 별개다.

비교 대상은 다음 네 controller다.

- MPPI
- Log-MPPI
- Cluster-MPPI
- BiC-MPPI

네 방법 모두 closed-loop receding-horizon 방식이다. 한 번 계산한 제어열
전체를 실행하지 않고, 매 제어 주기마다 첫 번째 제어만 적용한 뒤 현재
상태에서 다시 계획한다.

---

## 2. 문제 구성

### 2.1 Stitched BARN map

각 시나리오는 BARN dataset의 맵 150개 중 5개를 선택하여 y축 방향으로
연결한다.

- 시나리오 수: 10
- 시나리오당 맵 수: 5
- 전체 dataset map 수: 150
- map resolution: 0.1 m
- map 하나의 진행 방향 길이: 5 m
- 연결된 전체 진행 방향 길이: 25 m
- map 선택 seed: `7 + scenario_index`

동일한 scenario index에는 모든 solver가 동일한 stitched map을 사용한다.
맵 선택과 연결은
[`stitched_barn_common.h`](../src/wmrobot/gpu_waypoint_순차추종/stitched_barn_common.h)의
`selectMapIds()`와 `makeScenario()`에서 수행한다.

### 2.2 상태와 입력

WMRobot의 상태와 입력은 다음과 같다.

```text
x = [px, py, theta]
u = [v, omega]
```

연속시간 운동 모델은 다음과 같다.

```text
px_dot    = v cos(theta)
py_dot    = v sin(theta)
theta_dot = omega
```

입력 제한은 다음과 같다.

```text
0 <= v <= 1
-pi/2 <= omega <= pi/2
```

모델과 비용 함수는 [`model/wmrobot_map.h`](../model/wmrobot_map.h)에
정의되어 있다.

### 2.3 Waypoint

25 m 길이의 stitched map에 6개의 waypoint를 균등 배치한다.

| index | x [m] | y [m] | theta [rad] |
|---:|---:|---:|---:|
| 0 | 1.5 | 0 | π/2 |
| 1 | 1.5 | 5 | π/2 |
| 2 | 1.5 | 10 | π/2 |
| 3 | 1.5 | 15 | π/2 |
| 4 | 1.5 | 20 | π/2 |
| 5 | 1.5 | 25 | π/2 |

Waypoint 0은 초기 상태이고, 실제 추종은 waypoint 1부터 시작한다.

Waypoint 도달 여부는 방향을 제외한 위치 거리로 판정한다.

```text
sqrt((px - px_wp)^2 + (py - py_wp)^2) <= 0.45 m
```

반면 rollout의 목표 비용은 `(px, py, theta)` 전체 상태와 목표 상태 사이의
norm을 사용한다. 따라서 도달 판정은 위치만 사용하지만, 최적화 과정에서는
목표 방향 `theta = π/2`도 고려한다.

---

## 3. MPPI, Log-MPPI, Cluster-MPPI의 waypoint 추종

세 방법의 sampling 및 제어열 선택 방식은 서로 다르지만 waypoint를
제공하는 상위 closed-loop 구조는 동일하다.

```text
현재 상태 → waypoint 1
waypoint 1 도달 → waypoint 2
waypoint 2 도달 → waypoint 3
...
waypoint 5 도달 → 성공
```

매 제어 반복에서 다음 작업을 수행한다.

1. 아직 도달하지 않은 가장 가까운 순번의 waypoint 하나를 `x_target`으로
   설정한다.
2. 현재 상태에서 해당 waypoint를 목표로 rollout한다.
3. 선택된 최적 제어열의 첫 제어 `u0`만 0.1초 동안 적용한다.
4. 이동 후 현재 위치가 목표 waypoint의 0.45 m 이내인지 검사한다.
5. 도달했다면 target index를 증가시키고 다음 waypoint로 변경한다.
6. waypoint가 변경될 때 이전 목표에 대한 warm-start 제어열을 0으로
   초기화한다.

즉, 세 방법은 최종 waypoint나 모든 waypoint를 한 번에 직접 추종하지
않는다. 매 순간 **다음 waypoint 하나**를 추종한다.

### 3.1 Solver별 차이

| Solver | Waypoint target | rollout 처리의 핵심 |
|---|---|---|
| MPPI | 다음 waypoint 하나 | 전체 rollout을 비용 기반으로 가중 결합 |
| Log-MPPI | 다음 waypoint 하나 | MPPI와 같은 Gaussian rollout을 사용하고 log-sum-exp로 sampling weight 정규화 |
| Cluster-MPPI | 다음 waypoint 하나 | rollout을 군집화하고 cluster 대표 제어열 선택 |

Waypoint 처리 로직은 `runMppiLikeScenario()`에 공통으로 구현되어 있다.

---

## 4. BiC-MPPI의 multi-waypoint 계획

BiC-MPPI도 waypoint 도달 판정과 target index 증가는 동일하게 사용한다.
그러나 매 제어 주기에 계획하는 범위가 다른 세 방법보다 넓다.

### 4.1 Anchor 구성

BiC-MPPI는 현재 상태와 아직 통과하지 않은 모든 waypoint를 anchor로
구성한다.

초기 상태에서는:

```text
[현재 상태, wp1, wp2, wp3, wp4, wp5]
```

wp1 통과 후에는:

```text
[현재 상태, wp2, wp3, wp4, wp5]
```

따라서 “모든 waypoint에서 rollout한다”는 표현보다 다음 표현이 더
정확하다.

> BiC-MPPI는 현재 상태부터 남은 waypoint들 사이의 모든 연속 구간에
> 대해 양방향 rollout을 수행한다.

### 4.2 구간별 양방향 연결

각 인접 anchor 구간에 대해 다음 계산을 수행한다.

```text
구간 시작 anchor ── forward rollout ──▶
                                      연결
구간 종료 anchor ◀─ backward rollout ─
```

각 구간에서:

- 시작 anchor에서 `Nf=3000`개의 forward rollout
- 종료 anchor에서 `Nb=3000`개의 backward rollout
- forward horizon `Tf=50`
- backward horizon `Tb=50`
- rollout들을 clustering
- forward/backward cluster trajectory 사이의 가장 가까운 시점 쌍 선택
- 선택한 두 trajectory를 연결하여 구간 reference 생성

이 작업을 남은 모든 구간에 수행한다.

### 4.3 Global guide

구간별로 선택한 reference를 순서대로 연결한다.

```text
현재→wp1 reference
 + wp1→wp2 reference
 + wp2→wp3 reference
 + ...
 = 남은 전체 경로 reference
```

이 전체 reference 주위에서 `Nr=1500`개의 guide rollout을 추가로
생성한다. Guide MPPI가 선택한 최종 trajectory가 해당 제어 주기의
BiC-MPPI 최적 예측 경로 `Xo`다.

따라서 BiC-MPPI는 단순히 다음 waypoint만 목표로 하지 않는다.
다음 waypoint를 통과해야 한다는 순차 규칙은 유지하면서, 실제 계획에는
남은 waypoint 전체로 구성한 global guide를 사용한다.

구현은 `WaypointBatchBiMPPI_GPU::solveWaypointGraph()`와
`BiMPPI_GPU::guideReference()`에서 확인할 수 있다.

---

## 5. Horizon의 정확한 의미

공통 integration timestep은 다음과 같다.

```text
dt = 0.1 s
```

### 5.1 MPPI, Log-MPPI, Cluster-MPPI

세 방법은 다음 설정을 사용한다.

```text
T = 500
N = 3000
```

따라서 한 번의 solve에서 예측하는 시간은:

```text
500 step × 0.1 s = 50 s
```

이 horizon은 다음 waypoint 하나를 목표로 하는 receding horizon이다.
로봇은 500 step 전체를 실행하지 않고 첫 1 step만 실행한다.

### 5.2 BiC-MPPI 구간 horizon

BiC-MPPI의 기본 양방향 horizon은 다음과 같다.

```text
Tf = 50 step = 5 s
Tb = 50 step = 5 s
```

그러나 forward 50 step과 backward 50 step을 항상 단순히 이어 붙여
100 step으로 만드는 것은 아니다. 선택된 연결 시점을 `df`, `db`라고
하면 구간 reference의 길이는 다음과 같이 결정된다.

```text
Tsegment = max(Tf, df + (Tb - db))
```

따라서 현재 설정에서 구간 하나의 horizon은 50~100 step, 즉 5~10초
범위에서 동적으로 결정된다.

### 5.3 BiC-MPPI global guide horizon

전체 guide horizon `Tr`은 남은 모든 구간 reference 길이의 합이다.

```text
Tr = Σ Tsegment
```

시작 시에는 남은 구간이 5개이므로:

```text
250 <= Tr <= 500 step
25 s <= 예측 시간 <= 50 s
```

실제 저장 결과에서 첫 제어 반복의 `Tr`은 다음과 같았다.

| scenario | first-step Tr |
|---:|---:|
| 0 | 496 |
| 1 | 497 |
| 2 | 491 |
| 3 | 494 |
| 4 | 493 |
| 5 | 487 |
| 6 | 491 |
| 7 | 496 |
| 8 | 497 |
| 9 | 489 |

Waypoint를 통과하면 남은 구간 수가 줄어들기 때문에 `Tr`도 일반적으로
짧아진다.

BiC-MPPI의 metadata에도 `global_T`가 기록되지만, BiC planning에 직접
사용되는 고정 global horizon은 아니다. 실제 horizon은 `Tf`, `Tb`,
연결 시점 및 남은 구간 수로 만들어지는 동적 `Tr`이다.

---

## 6. Rollout GIF 해석

GIF의 각 프레임은 한 번의 closed-loop 제어 반복에 대응한다.

| 색상 | 의미 |
|---|---|
| 회색 | sampled rollout 또는 guide candidate |
| 주황색 | clustering으로 얻은 대표 trajectory |
| 빨간색 | 현재 제어 주기에서 선택된 최적 예측 경로 `Xo` |
| 검은색 | 현재까지 실제로 실행된 closed-loop 경로 |
| 검은 점 | 제어 적용 후의 현재 로봇 위치 |

### 6.1 빨간 경로

빨간 경로는 로봇이 앞으로 그대로 실행할 확정 경로가 아니다.

```text
현재 상태에서 후보 rollout 생성
→ 최적 예측 경로 Xo 선택
→ Xo의 첫 제어 u0만 실행
→ 다음 프레임에서 현재 상태 기준으로 다시 계획
```

그러므로 다음 프레임에서 빨간 경로가 크게 바뀌는 것은 정상이다.
실제로 실행된 부분은 검은 경로에 누적된다.

### 6.2 Solver별 빨간 경로

- MPPI, Log-MPPI, Cluster-MPPI:
  현재 상태에서 **다음 waypoint 하나**를 향한 500-step 최적 예측 경로
- BiC-MPPI:
  현재 상태부터 **남은 모든 waypoint 구간**을 연결한 reference에 guide
  MPPI를 적용한 동적 길이의 전역 최적 예측 경로

BiC-MPPI GIF에서 주황색 경로는 각 구간의 forward/backward cluster
trajectory이고, 회색에는 sampled rollout과 global guide candidate가
표시된다. 빨간색은 이 계산을 거쳐 최종 선택된 `Xo`다.

색상과 trajectory file의 매핑은
[`visualize_wmrobot_waypoint_stitched.py`](../visualize_wmrobot_waypoint_stitched.py)의
`gif_trajectory_groups()`에 정의되어 있다.

---

## 7. 성공, 충돌 및 결과 해석

한 scenario는 다음 조건 중 하나로 종료된다.

- 마지막 waypoint까지 도달: 성공
- 이동 전후 상태에서 map collision 검출: 충돌 실패
- `maxiter=1000`까지 마지막 waypoint에 도달하지 못함: 실패

`results/wmrobot_waypoint_stitched_sequential/wmrobot_stitched_aggregate.csv`
기준 결과는 다음과 같다.

| Solver | 성공률 | 평균 도달 waypoint 수 |
|---|---:|---:|
| MPPI | 0.0 | 1.1 |
| Log-MPPI | 0.0 | 1.1 |
| Cluster-MPPI | 0.0 | 1.1 |
| BiC-MPPI | 0.5 | 4.3 |

여기서 `reached_waypoints`는 초기 waypoint 0을 포함한 index 진행 상태다.
예를 들어 값이 1이면 초기 waypoint만 완료되어 있으며 아직 waypoint 1을
통과하지 못한 상태를 의미한다.

이 결과만으로 “BiC-MPPI는 항상 우수하다”는 일반적 결론을 내릴 수는
없다. 이 실험에서는 세 baseline이 다음 waypoint 하나를 직접 목표로
계획하는 반면, BiC-MPPI는 남은 전체 구간을 이용한 global guide를
사용한다. 따라서 algorithm 자체뿐 아니라 제공되는 planning scope도
다르다는 점을 함께 명시해야 한다.

---

## 8. 결과 파일

각 solver 디렉터리에 다음 파일이 생성된다.

| 파일 | 내용 |
|---|---|
| `metadata.json` | 실제 사용된 실험 및 solver 설정 |
| `summary.csv` | scenario별 성공, 충돌, 반복 수, 시간, 최종 오차 |
| `paths.csv` | 실제 실행된 closed-loop 상태 경로 |
| `timings.csv` | 제어 반복별 rollout, clustering, connection, guide 시간 |
| `maps.csv` | stitched map을 구성한 원본 map ID |
| `waypoints.csv` | waypoint 상태 |
| `stitched_maps/` | scenario별 stitched occupancy map |
| `vis_data/` | rollout, cluster, guide candidate, optimal trajectory |
| `rollout_gifs/` | solver/scenario별 rollout 애니메이션 |

---

## 9. 현재 runner를 이용한 재현

현재 repository에서는 다음 명령으로 같은 종류의 실험을 실행할 수 있다.

```bash
./run_waypoints.sh \
  --scenario sequential \
  --out results/wmrobot_waypoint_stitched_sequential \
  --overwrite
```

명시적으로 주요 설정을 지정하려면 다음과 같이 실행한다.

```bash
./run_waypoints.sh \
  --scenario sequential \
  --out results/wmrobot_waypoint_stitched_sequential \
  --overwrite \
  --scenarios 10 \
  --maps-per-scenario 5 \
  --maxiter 1000 \
  --waypoint-count 6 \
  --global-T 500 \
  --bic-Tf 50 \
  --bic-Tb 50 \
  --N 3000 \
  --Nf 3000 \
  --Nb 3000 \
  --Nr 1500 \
  --seed 7 \
  --vis-every 1
```

Runner는 다음 순서로 solver를 실행한 뒤 시각화를 생성한다.

```text
MPPI
→ Log-MPPI
→ Cluster-MPPI
→ BiC-MPPI
→ CSV aggregate, PNG, rollout GIF 생성
```

기존 `wmrobot_waypoint_stitched_sequential` 결과에는 최신 runner가 만드는
`run_manifest.txt`가 없다. 따라서 기존 raw 결과는 unified runner 도입
전에 각 solver binary를 순서대로 직접 실행하여 생성했을 가능성이 높다.
다만 각 solver의 `metadata.json`에 기록된 설정은 위에 정리한 설정과
일치한다.

---

## 10. 핵심 요약

```text
MPPI / Log-MPPI / Cluster-MPPI
    다음 waypoint 하나를 추종
    0.45 m 이내 도달 시 다음 waypoint로 변경
    고정 horizon T=500

BiC-MPPI
    현재 상태와 남은 모든 waypoint를 anchor로 구성
    각 연속 구간에서 forward/backward rollout 및 연결
    모든 구간을 합쳐 global reference 생성
    reference 주위에서 guide MPPI 수행
    waypoint 통과 판정은 동일하게 0.45 m
    global guide horizon은 남은 구간에 따라 동적으로 변함

Rollout GIF
    빨간색 = 현재 시점의 최적 예측 경로
    검은색 = 실제로 실행된 누적 경로
    매 주기 빨간 경로의 첫 제어만 실행
```

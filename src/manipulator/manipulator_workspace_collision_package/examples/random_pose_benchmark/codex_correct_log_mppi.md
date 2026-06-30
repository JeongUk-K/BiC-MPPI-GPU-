# Codex Task: Correct Log-MPPI Implementation

## Objective

Update the current `Log-MPPI` solver so that it implements **log-domain MPPI weight normalization**, not log-normal input-noise reshaping.

The current implementation appears to modify the control perturbation as something like

\[
\epsilon' = \epsilon \exp(z), \qquad z \sim \mathcal{N}(\mu_{\log}, \sigma_{\log}^2),
\]

which is a **log-normal noise reshaping variant**, not a log-domain MPPI weighting method.

The corrected `Log-MPPI` baseline should use the **same Gaussian control sampling distribution as standard MPPI**, and only compute the MPPI exponential weights in the log domain using a numerically stable log-sum-exp normalization.

---

## High-Level Definition

Standard MPPI samples control sequences as

\[
U^k = \bar U + \epsilon^k,
\qquad
\epsilon^k \sim \mathcal{N}(0,\Sigma_u),
\]

rolls out each sampled sequence, evaluates its cost \(J^k\), and computes

\[
w^k =
\frac{\exp(-\gamma_u J^k)}
{\sum_j \exp(-\gamma_u J^j)}.
\]

The corrected `Log-MPPI` should compute the same mathematical weights, but in a numerically stable log-domain form:

\[
\log w^k_{\rm unnorm} = -\gamma_u J^k,
\]

\[
\alpha = \max_k \log w^k_{\rm unnorm},
\]

\[
\tilde w^k = \exp(\log w^k_{\rm unnorm} - \alpha),
\]

\[
w^k = \frac{\tilde w^k}{\sum_j \tilde w^j}.
\]

Equivalently,

\[
\log Z = \alpha + \log\sum_j \exp(\log w^j_{\rm unnorm}-\alpha),
\]

\[
w^k = \exp(\log w^k_{\rm unnorm}-\log Z).
\]

The final control update remains

\[
U^* = \sum_k w^k U^k.
\]

---

## Files to Inspect

Inspect the current solver package and locate the relevant files. Likely paths include:

```text
src/manipulator/manipulator_workspace_collision_package/include/mppi_gpu.cuh
src/manipulator/manipulator_workspace_collision_package/include/mppi_param.h
src/manipulator/manipulator_workspace_collision_package/src/mppi_gpu.cu
src/manipulator/manipulator_workspace_collision_package/src/log_mppi_gpu.cu
src/manipulator/manipulator_workspace_collision_package/src/cluster_mppi_gpu.cu
src/manipulator/manipulator_workspace_collision_package/src/bi_mppi_gpu.cu
src/manipulator/manipulator_workspace_collision_package/examples/random_pose_benchmark/
```

Also inspect any `log_mppi_gpu.cuh`, `log_mppi_param`, or benchmark registration files if they exist.

---

## Required Behavioral Change

### 1. Remove log-normal perturbation from Log-MPPI

Search for logic similar to:

```cpp
z = mu_log + sigma_log * normal();
scale = exp(z);
noise = sigma * normal() * scale;
```

or:

```cpp
eps *= exp(z);
```

or any use of parameters such as:

```cpp
mu_log
sigma_log
log_mu
log_std
```

inside the **sampling perturbation** of `Log-MPPI`.

For corrected `Log-MPPI`, the sampled control should be identical to standard MPPI:

```cpp
noise_j = sigma_j * normal_j;
u_sample_j = u_nominal_j + noise_j;
u_sample_j = clamp(u_sample_j, u_min_j, u_max_j);
```

Do not apply `log`, `exp`, or log-normal scaling to the input or perturbation.

---

### 2. Implement log-sum-exp weight normalization

Replace direct exponential weight normalization with a numerically stable log-domain computation.

Do **not** compute:

```cpp
weight[k] = exp(-gamma_u * cost[k]);
sum += weight[k];
weight[k] /= sum;
```

without stabilization.

Use this structure:

```cpp
log_w[k] = -gamma_u * cost[k];
log_w_max = max_k(log_w[k]);
sum_exp = sum_k(exp(log_w[k] - log_w_max));
log_sum_w = log_w_max + log(sum_exp);
weight[k] = exp(log_w[k] - log_sum_w);
```

If the existing MPPI solver already subtracts the minimum cost,

```cpp
weight[k] = exp(-gamma_u * (cost[k] - cost_min));
```

that is a partially stabilized form. However, for the `Log-MPPI` baseline, implement the explicit log-sum-exp form so the method is clearly distinguishable and accurately named.

---

### 3. Handle invalid or collision-penalized samples

The code uses a large collision penalty such as:

```cpp
J_col = 1e8;
```

For log-domain weighting, samples with collision-scale costs should either be excluded from the softmax or assigned zero probability.

Recommended behavior:

```cpp
if (!isfinite(cost[k]) || cost[k] >= J_col) {
    log_w[k] = -INFINITY;
} else {
    log_w[k] = -gamma_u * cost[k];
}
```

Then compute `log_w_max`.

If all samples are invalid, use a deterministic fallback:

```cpp
U_out = U_argmin_cost;
```

or preserve the existing least-violating fallback behavior used elsewhere in the package.

Do not allow `NaN`, `Inf`, or zero-sum weights to propagate to the output control sequence.

---

### 4. Keep the output update identical to MPPI

After normalized weights are computed, update the control sequence as usual:

```cpp
U_out[t, j] = sum_k weight[k] * U_sample[k, t, j];
```

Then apply the same admissible input projection or clamping used by standard MPPI.

Do not change:

- dynamics rollout,
- cost function,
- collision checking,
- warm-start shifting,
- benchmark success criteria,
- random scenario generation.

The only intended difference between `MPPI` and corrected `Log-MPPI` is the **weight normalization implementation**.

---

## CUDA Implementation Guidance

If the solver is implemented on GPU, organize the computation as follows.

### Kernel or step 1: rollout and cost

For each sample \(k\):

```cpp
sample Gaussian perturbation
construct U_sample[k]
roll out dynamics
evaluate cost[k]
store U_sample[k] and cost[k]
```

Sampling must match standard MPPI.

### Kernel or step 2: compute log weights

For each sample \(k\):

```cpp
if invalid:
    log_w[k] = -INFINITY;
else:
    log_w[k] = -gamma_u * cost[k];
```

### Kernel or reduction step 3: max log weight

Compute

```cpp
log_w_max = max_k(log_w[k]);
```

### Kernel or reduction step 4: sum exponentials

Compute

```cpp
sum_exp = sum_k(exp(log_w[k] - log_w_max));
```

Then:

```cpp
log_sum_w = log_w_max + log(sum_exp);
```

### Kernel or step 5: normalize and update

For each sample:

```cpp
weight[k] = exp(log_w[k] - log_sum_w);
```

Then reduce:

```cpp
U_out = sum_k weight[k] * U_sample[k];
```

---

## CPU Reference Implementation

Add a CPU-side helper if useful for testing:

```cpp
std::vector<double> computeLogSoftmaxWeights(
    const std::vector<double>& costs,
    double gamma_u,
    double collision_cost
) {
    const int N = static_cast<int>(costs.size());
    std::vector<double> log_w(N, -std::numeric_limits<double>::infinity());
    std::vector<double> w(N, 0.0);

    for (int k = 0; k < N; ++k) {
        if (std::isfinite(costs[k]) && costs[k] < collision_cost) {
            log_w[k] = -gamma_u * costs[k];
        }
    }

    double log_w_max = -std::numeric_limits<double>::infinity();
    for (double v : log_w) {
        log_w_max = std::max(log_w_max, v);
    }

    if (!std::isfinite(log_w_max)) {
        // Caller should handle fallback, e.g., choose argmin cost.
        return w;
    }

    double sum_exp = 0.0;
    for (int k = 0; k < N; ++k) {
        if (std::isfinite(log_w[k])) {
            sum_exp += std::exp(log_w[k] - log_w_max);
        }
    }

    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        return w;
    }

    const double log_sum_w = log_w_max + std::log(sum_exp);

    for (int k = 0; k < N; ++k) {
        if (std::isfinite(log_w[k])) {
            w[k] = std::exp(log_w[k] - log_sum_w);
        }
    }

    return w;
}
```

---

## Testing Requirements

### 1. Unit-level weight test

Create a small deterministic test for weight normalization.

Example:

```cpp
costs = {1000.0, 1001.0, 1002.0};
gamma_u = 10.0;
```

Direct exponentiation may underflow. Log-sum-exp should return finite normalized weights with

```cpp
sum(weights) ~= 1.0
all weights finite
weights[0] > weights[1] > weights[2]
```

### 2. Equivalence test at moderate cost scale

For moderate costs:

```cpp
costs = {1.0, 2.0, 3.0};
gamma_u = 0.5;
```

Compare:

```cpp
direct softmax over -gamma_u * cost
```

and:

```cpp
log-sum-exp softmax
```

They should match within numerical tolerance.

### 3. Invalid sample test

Use:

```cpp
costs = {1e8, 12.0, 15.0, INFINITY};
```

Expected:

```cpp
weight[0] = 0
weight[3] = 0
weight[1] > weight[2]
sum(weights) ~= 1
```

### 4. All-invalid fallback test

Use:

```cpp
costs = {1e8, 1e8, INFINITY};
```

Expected:

```cpp
sum(weights) == 0
fallback branch chooses argmin finite/least-violating sample
no NaN output
```

### 5. Regression benchmark

Run the existing random-pose benchmark and ensure:

- `MPPI` results are unchanged.
- corrected `Log-MPPI` runs without NaNs.
- `Log-MPPI` uses the same sample count and horizon as MPPI.
- `Log-MPPI` no longer uses `mu_log` or `sigma_log` for perturbation scaling.
- Output CSV labels remain clear.

---

## Benchmark Reporting Update

After modifying the solver, update any experiment description or paper text.

### Old inaccurate description

Avoid:

```text
Log-MPPI modifies Gaussian control noise by a log-normal multiplier.
```

Avoid:

```text
Log-MPPI applies log to the input.
```

### Correct description

Use:

```text
Log-MPPI uses the same Gaussian rollout distribution as MPPI but computes the exponential trajectory weights using log-sum-exp normalization for numerical stability.
```

Or in LaTeX:

```latex
Log-MPPI uses the same Gaussian rollout distribution as MPPI, but evaluates the exponential trajectory weights in the log domain using log-sum-exp normalization.
```

---

## Expected Final State

After this task:

1. `MPPI` remains the standard baseline.
2. `Log-MPPI` uses identical Gaussian sampling to MPPI.
3. `Log-MPPI` differs only in numerically stable log-domain weight normalization.
4. Log-normal perturbation code is removed from the `Log-MPPI` solver.
5. `mu_log`, `sigma_log`, or equivalent parameters are removed from active Log-MPPI sampling logic, or clearly deprecated if kept for backward compatibility.
6. No NaNs or zero-weight failures occur when cost magnitudes are large.
7. Paper/README descriptions match the corrected implementation.

---

## Important Scope Boundary

Do not modify BiC-MPPI, Cluster-MPPI, scenario generation, manipulator dynamics, or cost functions unless required to share a common utility for log-sum-exp weighting.

This task is specifically about correcting the `Log-MPPI` solver semantics.

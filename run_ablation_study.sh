#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/experiment_common.sh"
ORIGINAL_ARGS=("$@")

STUDY="wmrobot"
OUT_DIR=""
DO_BUILD=1
OVERWRITE=0
SMOKE=0
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_ablation_study.sh [options] [-- experiment args]

Ablation-study runner.  Each study stores original CSV outputs and the top-level
directory gets an aggregate_timing_success.csv with success rate and separated
rollout / clustering / connection / guide timing columns where the executable
emits them.

Options:
  --study NAME     wmrobot, same_budget, clustering_parameter, or all.
                   Default: wmrobot
  --out DIR        Output directory. Default: results/ablation_study/<study>
  --overwrite      Replace output directory and pass overwrite to supporting binaries.
  --no-build       Use existing build/gpu binaries.
  --smoke          Forward a short validation run to supporting binaries.
  -h, --help       Show this help.

Examples:
  ./run_ablation_study.sh --study wmrobot --smoke --overwrite
  ./run_ablation_study.sh --study same_budget -- --budgets 1000 3000 6000
  ./run_ablation_study.sh --study all --smoke --overwrite

Arguments after `--` are forwarded to the wmrobot ablation and same-budget
executables.  The clustering-parameter binaries do not accept runtime options.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --study)
            STUDY="$2"; shift 2 ;;
        --out)
            OUT_DIR="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
        --no-build)
            DO_BUILD=0; shift ;;
        --smoke)
            SMOKE=1; shift ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; RUN_ARGS+=("$@"); break ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    if [[ "$STUDY" == "all" ]]; then
        OUT_DIR="results/ablation_study"
    else
        OUT_DIR="results/ablation_study/$STUDY"
    fi
fi

ensure_executable() {
    local exe="$1"
    [[ -x "$GPU_DIR/$exe" ]] || {
        echo "ERROR: missing executable: $GPU_DIR/$exe" >&2
        exit 1
    }
}

run_wmrobot_ablation() {
    local out="$1"
    mkdir -p "$out"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot_ablation
    reset_build_outputs
    ensure_executable wmrobot_ablation_gpu_all

    local args=("--output-prefix" "$out/wmrobot_ablation_gpu")
    [[ "$SMOKE" -eq 1 ]] && args+=("--smoke")
    args+=("${RUN_ARGS[@]}")

    echo "[run] wmrobot_ablation_gpu_all"
    (cd "$BUILD_DIR" && ./gpu/wmrobot_ablation_gpu_all "${args[@]}")
    copy_build_vis_data "$out"
}

run_same_budget() {
    local out="$1"
    mkdir -p "$out"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot_same_budget
    reset_build_outputs
    ensure_executable wmrobot_same_budget_all

    local git_commit
    git_commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    local args=("--out" "$out" "--git-commit" "$git_commit")
    [[ "$OVERWRITE" -eq 1 ]] && args+=("--overwrite")
    [[ "$SMOKE" -eq 1 ]] && args+=("--smoke")
    args+=("${RUN_ARGS[@]}")

    echo "[run] wmrobot_same_budget_all"
    (cd "$BUILD_DIR" && ./gpu/wmrobot_same_budget_all "${args[@]}")
    copy_build_vis_data "$out"
}

run_clustering_parameter() {
    local out="$1"
    mkdir -p "$out/raw_csv"
    if [[ "$SMOKE" -eq 1 ]]; then
        echo "[note] clustering_parameter binaries do not support --smoke; running the fixed binaries."
    fi
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_clustering.sh"
    reset_build_outputs

    local solvers=(
        bi_mppi_0.001
        bi_mppi_0.005
        bi_mppi_0.01
        bi_mppi_0.05
        bi_mppi_0.1
    )

    for exe in "${solvers[@]}"; do
        ensure_executable "$exe"
    done

    for exe in "${solvers[@]}"; do
        local epsilon="${exe#bi_mppi_}"
        local csv="result_bi_mppi_clustering_${epsilon}.csv"
        rm -f "$BUILD_DIR/$csv"
        echo "[run] $exe"
        (cd "$BUILD_DIR" && "./gpu/$exe")
        if [[ -f "$BUILD_DIR/$csv" ]]; then
            cp -f "$BUILD_DIR/$csv" "$out/raw_csv/"
        else
            echo "[warn] expected CSV was not generated: $BUILD_DIR/$csv" >&2
        fi
    done
    copy_build_vis_data "$out"
}

OUT_ABS="$(repo_out_path "$OUT_DIR")"
prepare_output_dir "$OUT_ABS" "$OVERWRITE"
write_run_manifest "$OUT_ABS" "$0" "${ORIGINAL_ARGS[@]}"

scenario_out() {
    local name="$1"
    if [[ "$STUDY" == "all" ]]; then
        printf '%s/%s\n' "$OUT_ABS" "$name"
    else
        printf '%s\n' "$OUT_ABS"
    fi
}

case "$STUDY" in
    wmrobot)
        run_wmrobot_ablation "$(scenario_out wmrobot)" ;;
    same_budget)
        run_same_budget "$(scenario_out same_budget)" ;;
    clustering_parameter)
        run_clustering_parameter "$(scenario_out clustering_parameter)" ;;
    all)
        run_wmrobot_ablation "$(scenario_out wmrobot)"
        run_same_budget "$(scenario_out same_budget)"
        run_clustering_parameter "$(scenario_out clustering_parameter)" ;;
    *)
        echo "ERROR: unsupported ablation study: $STUDY" >&2
        usage >&2
        exit 1 ;;
esac

aggregate_timing_success_csv "$OUT_ABS" "$OUT_ABS/aggregate_timing_success.csv"
write_outputs_manifest "$OUT_ABS"

echo ""
echo "Outputs:"
echo "  $OUT_ABS"
echo "  $OUT_ABS/aggregate_timing_success.csv"

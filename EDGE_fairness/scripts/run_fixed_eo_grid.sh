#!/usr/bin/env bash
# Fixed sampling-time EO guidance, using the existing SP grid/evaluator.
# Usage: bash scripts/run_fixed_eo_grid.sh {cora|citeseer} PHYSICAL_GPU [--dry_run]
# Set EDGE_RUN_DIR to a prepared stage-1 run; EDGE_CHECKPOINT defaults to 10000.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 {cora|citeseer} PHYSICAL_GPU [--dry_run]" >&2
    exit 2
fi
dataset="$1"
gpu="$2"
dry_run=false
if [[ $# -eq 3 ]]; then
    [[ "$3" == "--dry_run" ]] || { echo "Unknown option: $3" >&2; exit 2; }
    dry_run=true
fi
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "PHYSICAL_GPU must be one GPU index" >&2; exit 2; }

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_exec="${EDGE_PYTHON:-python}"
case "$dataset" in
    cora|citeseer) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
command -v "$python_exec" >/dev/null || { echo "Python executable not found: $python_exec" >&2; exit 2; }
run_dir="${EDGE_RUN_DIR:?Set EDGE_RUN_DIR to the trained run directory containing args.pickle and check/}"
run_name="${EDGE_RUN_NAME:-$(basename -- "$run_dir")}"
checkpoint="${EDGE_CHECKPOINT:-10000}"
[[ "$checkpoint" =~ ^[1-9][0-9]*$ ]] || { echo "EDGE_CHECKPOINT must be a positive integer" >&2; exit 2; }
cpu_threads="${EDGE_CPU_THREADS:-1}"
[[ "$cpu_threads" =~ ^[1-9][0-9]*$ ]] || { echo "EDGE_CPU_THREADS must be a positive integer" >&2; exit 2; }
export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads" TF_NUM_INTRAOP_THREADS="$cpu_threads" TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES="$gpu"
out_dir="${EDGE_OUT_DIR:-$repo_dir/results/fixed_eo_grid/$dataset}"

command=(
    "$python_exec" fair_grid_eval_generated_graphs.py
    --repo_dir "$repo_dir" --dataset "$dataset"
    --run_name "$run_name" --run_dir "$run_dir" --checkpoint "$checkpoint"
    --fair_score_metric eo --fair_score_guidance_normalize True
    --eta_values 0.0005 0.001 0.0025 0.005 0.01 0.02
    --k_values 0.1 0.3 0.5 0.7
    --include_baseline --baseline_k 1
    --num_samples 8 --seeds 0 1 2
    --gen_device cuda:0 --lp_device cuda:0 --lp_model gcn
    --fair_sensitive_attr y --largest_cc False --graph_variant full
    --auc_candidates lp/auc_mean --eo_candidates lp/eo_abs_gap_mean
    --out_dir "$out_dir"
)
selection_command=(
    "$python_exec" scripts/select_fixed_eo_operating_points.py
    --dataset "$dataset"
    --summary_csv "$out_dir/eo/summary_long_${dataset}.csv"
    --auc_max_drop 0.01
    --out_csv "$out_dir/eo/operating_points_${dataset}.csv"
)

if [[ "$dry_run" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    printf '%q ' "${selection_command[@]}"
    printf '\n'
    "$python_exec" - "${command[@]:2}" <<'PY'
from pathlib import Path
from fair_grid_eval_generated_graphs import make_combos, parse_args, validate_inputs
args = parse_args()
validate_inputs(args, Path(args.repo_dir))
combos = make_combos(args.eta_values, args.k_values, args.pair_mode)
assert len(set(combos)) == 24 and all(eta > 0 for eta, _ in combos)
print("Inputs validated: 24 guided settings + 1 uncontrolled, 3 seeds, 8 graphs/seed; no generation or evaluation launched.")
PY
    exit 0
fi

"${command[@]}"
"${selection_command[@]}"

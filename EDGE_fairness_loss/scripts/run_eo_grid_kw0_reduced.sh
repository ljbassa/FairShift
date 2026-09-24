#!/usr/bin/env bash
# Usage: bash scripts/run_eo_grid_kw0_reduced.sh {cora|citeseer} PHYSICAL_GPU [--dry_run]
# Derived from run_eo_grid_v5.sh and the completed v5 LP summaries.
# Only the tracking weight and searched combinations change; k candidates stay
# as in the original weak (.3/.5) and strong (.3/.5/.7) blocks.
# Pruning is empirical: old tracking=.1 rankings need not hold at tracking=0.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 {cora|citeseer} PHYSICAL_GPU [--dry_run]" >&2
    exit 2
fi
dataset="$1"
gpu="$2"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "PHYSICAL_GPU must be one GPU index" >&2
    exit 2
fi
dry_args=()
if [[ $# -eq 3 ]]; then
    if [[ "$3" != "--dry_run" ]]; then
        echo "Only --dry_run is accepted as the third argument" >&2
        exit 2
    fi
    dry_args=(--dry_run)
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_exec="$(command -v "${EDGE_PYTHON:-python}")"
case "$dataset" in
    cora)
        stage1="${STAGE1_CKPT:-$repo_dir/../EDGE_fairness/wandb/$dataset/multinomial_diffusion/multistep/REPLACE_WITH_STAGE1_RUN/check/checkpoint_9999.pt}"
        diffusion_steps=256
        batch_size=2
        run_count=28
        ;;
    citeseer)
        stage1="${STAGE1_CKPT:-$repo_dir/../EDGE_fairness/wandb/$dataset/multinomial_diffusion/multistep/REPLACE_WITH_STAGE1_RUN/check/checkpoint_9999.pt}"
        diffusion_steps=128
        batch_size=4
        run_count=16
        ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
test -x "$python_exec"
if [[ ${#dry_args[@]} -eq 0 && ! -f "$stage1" ]]; then
    echo "Set STAGE1_CKPT to a compatible $dataset Stage-1 checkpoint: $stage1" >&2
    exit 2
fi

# Restrict every child process, including evaluators and data-loader workers.
cpu_cores="${EDGE_CPU_CORES:-1}"
cpu_threads="${EDGE_CPU_THREADS:-1}"
if [[ ! "$cpu_cores" =~ ^[1-9][0-9]*$ || ! "$cpu_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "EDGE_CPU_CORES and EDGE_CPU_THREADS must be positive integers" >&2
    exit 2
fi
cpu_list="${EDGE_CPU_LIST:-}"
if [[ -z "$cpu_list" ]]; then
    allowed_cpus="$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)"
    IFS=, read -r -a cpu_ranges <<< "$allowed_cpus"
    selected_cpus=()
    for cpu_range in "${cpu_ranges[@]}"; do
        IFS=- read -r first_cpu last_cpu <<< "$cpu_range"
        last_cpu="${last_cpu:-$first_cpu}"
        for ((cpu=first_cpu; cpu<=last_cpu && ${#selected_cpus[@]}<cpu_cores; cpu++)); do
            selected_cpus+=("$cpu")
        done
    done
    cpu_list="$(IFS=,; echo "${selected_cpus[*]}")"
fi
taskset -pc "$cpu_list" "$$" >/dev/null
export OMP_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads" VECLIB_MAXIMUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads" BLIS_NUM_THREADS="$cpu_threads"
export CUDA_VISIBLE_DEVICES="$gpu"
echo "[cpu] allowed CPUs: $cpu_list; requested library threads: $cpu_threads"

prefix="stage2_${dataset}_eo_controller_v5_kw0_reduced"
root="$repo_dir/wandb/$dataset/multinomial_diffusion/controller/eo"
common=(
    --repo_dir "$repo_dir" --python_exec "$python_exec"
    --stage1_ckpt "$stage1" --name_prefix "$prefix"
    --dataset "$dataset" --device cuda:0 --generated_eval_device cuda:0
    --fair_score_metric eo --fair_score_guidance_normalize False
    --diffusion_dim 128 --diffusion_steps "$diffusion_steps" --batch_size "$batch_size"
    --edge_dropout 0 --noise_schedule linear
    --controller_epochs 600 --controller_lrs 0.001
    --controller_replay_num_samples 2 --controller_replay_refresh 100
    --k_tracking_weights 0 --eval_every 300 --check_every 300
    --num_generation 8 --sample_batch_size 1
    --generation_seed 0 --generated_eval_seed 0
    --run_generated_eval --skip_existing --fail_fast
    --pareto_label_points none
)
if [[ ${#dry_args[@]} -gt 0 ]]; then
    # Keep command validation out of the actual experiment manifest.
    dry_dir="$(mktemp -d "${TMPDIR:-/tmp}/edge-eo-kw0-dry.XXXXXXXX")"
    trap 'rm -rf -- "$dry_dir"' EXIT
    common+=(--manifest "$dry_dir/manifest.jsonl")
fi

run_grid() {
    "$python_exec" scripts/run_controller_grid.py \
        "${common[@]}" "${dry_args[@]}" "$@" -- --seed 0
}

if [[ "$dataset" == cora ]]; then
    # AUC-preserving neighborhood: includes the v5 AUC-retained winner (8).
    run_grid --fair_score_k_values 0.3 0.5 \
        --eta_values 100 --fair_weights 1000 10000 --utility_weights 0.3 3 \
        --include_uncontrolled --skip_generated_pareto
    # Preserve an intermediate historical AUC/EO tradeoff (2).
    run_grid --fair_score_k_values 0.3 0.5 \
        --eta_values 1500 --fair_weights 1000 --utility_weights 0.3 \
        --skip_generated_pareto
    # EO-focused neighborhood: includes the v5 minimum-EO winner (18).
    run_grid --fair_score_k_values 0.3 0.5 0.7 \
        --eta_values 2500 --fair_weights 1000 10000 --utility_weights 0.05 0.1 0.3
else
    # Preserve all six historical CiteSeer Pareto parameter triples, expanding
    # each over its original k candidates: 2 + 2 + 3 + 9 = 16 runs.
    run_grid --fair_score_k_values 0.3 0.5 \
        --eta_values 100 --fair_weights 1000 --utility_weights 3 \
        --include_uncontrolled --skip_generated_pareto
    run_grid --fair_score_k_values 0.3 0.5 \
        --eta_values 2500 --fair_weights 1000 --utility_weights 1 \
        --skip_generated_pareto
    run_grid --fair_score_k_values 0.3 0.5 0.7 \
        --eta_values 5000 --fair_weights 1000 --utility_weights 0.3 \
        --skip_generated_pareto
    run_grid --fair_score_k_values 0.3 0.5 0.7 \
        --eta_values 10000 --fair_weights 1000 10000 100000 --utility_weights 0.1
fi

if [[ ${#dry_args[@]} -gt 0 ]]; then
    echo "[dry_run] Planned: one uncontrolled baseline + $run_count controller runs; no GPU work launched."
    exit 0
fi

"$python_exec" scripts/select_controller_operating_points.py \
    --dataset "$dataset" \
    --summary_csv "$root/${prefix}_summary.csv" \
    --uncontrolled_summary_csv "$root/uncontrolled_${prefix}/generated_samples/controller_best.pyg_full.overlap_lp_gae_summary.csv" \
    --auc_max_drop 0.01 \
    --out_csv "$repo_dir/results/eo_grid_kw0_reduced/${dataset}_operating_points.csv" \
    --plot_path "$repo_dir/results/eo_grid_kw0_reduced/${dataset}_auc_eo.png"

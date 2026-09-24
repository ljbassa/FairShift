# EDGE Fairness Loss

This repository trains a Stage-2 fairness controller on top of a Stage-1 EDGE denoising model trained in
`EDGE_fairness`. The Stage-2 controller learns per-step fairness-guidance parameters, exports generated graphs, and
supports SP and EO objectives with LP AUC vs fairness Pareto summaries.

Use this repository after you already have a compatible Stage-1 checkpoint. This FairShift copy contains code,
configuration examples, and tests. Supply graph data and checkpoints separately; experiment outputs are not shipped.

## What You Need

Required inputs:

- A Stage-1 checkpoint trained in `../EDGE_fairness`, for example
  `../EDGE_fairness/wandb/cora/multinomial_diffusion/multistep/<EDGE_STAGE1_RUN_NAME>/check/checkpoint_9999.pt`.
- A matching graph pickle in this repository, for example `graphs/cora_feat.pkl`.
- The same model-shape arguments used by Stage-1: `--dataset`, `--diffusion_dim`, `--diffusion_steps`,
  `--edge_dropout`, `--use_node_feat`, `--degree`, `--noise_schedule`, `--loss_type`, `--parametrization`, and
  `--num_heads`.
- Python dependencies from `requirements.txt` or `requirements.rest.txt`, depending on whether Torch is already
  installed in your environment.

`train_controller.py` loads only the Stage-1 model weights from `--controller_pretrained_ckpt`. It does not read the
Stage-1 `args.pickle`, so pass the matching architecture flags manually.

## Run Order

1. Prepare graph pickle files.
2. Train a Stage-1 base model in `EDGE_fairness`.
3. Train one Stage-2 controller with `train_controller.py`.
4. Evaluate the controller-generated graphs with `evaluate_generated_graphs.py`.
5. Run a controller grid with `scripts/run_controller_grid.py`.
6. Summarize the grid and draw the Pareto curve.

The grid script can perform steps 5 and 6 automatically when `--run_generated_eval` is enabled.

## Installation

```bash
cd EDGE_fairness_loss
pip install -r requirements.txt
```

If your Torch/CUDA stack is already installed manually:

```bash
pip install -r requirements.rest.txt
```

## Prepare Graph Pickles

The training and evaluation code expects NetworkX pickles under `graphs/`. Each pickle should include node attributes
`x`, `y`, and `orig_id`.

Create them in this repository:

```bash
cd EDGE_fairness_loss
mkdir -p graphs data

python datasets/make_planetoid_pickle.py \
  --dataset cora \
  --root data \
  --out graphs/cora_feat.pkl

python datasets/make_planetoid_pickle.py \
  --dataset citeseer \
  --root data \
  --out graphs/citeseer_feat.pkl

python datasets/make_planetoid_pickle.py \
  --dataset amazon_photo \
  --root data \
  --out graphs/amazon_photo_feat.pkl
```

You can also copy the same pickle files from `EDGE_fairness`:

```bash
cp ../EDGE_fairness/graphs/cora_feat.pkl graphs/cora_feat.pkl
cp ../EDGE_fairness/graphs/citeseer_feat.pkl graphs/citeseer_feat.pkl
cp ../EDGE_fairness/graphs/amazon_photo_feat.pkl graphs/amazon_photo_feat.pkl
```

## Obtain the Stage-1 Checkpoint

Train the base model in `EDGE_fairness`. The Cora example below matches the controller examples in this README.

```bash
cd EDGE_fairness
export EDGE_STAGE1_RUN=replace_with_edge_stage1_run_name

python train.py \
  --name "$EDGE_STAGE1_RUN" \
  --epochs 10000 \
  --num_generation 8 \
  --num_iter 32 \
  --diffusion_dim 128 \
  --diffusion_steps 256 \
  --edge_dropout 0.05 \
  --device cuda:0 \
  --dataset cora \
  --batch_size 2 \
  --clip_value 1 \
  --lr 5e-4 \
  --optimizer adam \
  --final_prob_edge 1 0 \
  --sample_time_method importance \
  --check_every 1000 \
  --eval_every 1000 \
  --noise_schedule linear \
  --dp_rate 0.0 \
  --loss_type vb_ce_xt_prescribred_st \
  --parametrization xt_prescribed_st \
  --empty_graph_sampler empirical \
  --degree \
  --num_heads 8 8 8 8 1 \
  --use_node_feat \
  --log_wandb False
```

The checkpoint used by Stage-2 is:

```text
../EDGE_fairness/wandb/cora/multinomial_diffusion/multistep/<EDGE_STAGE1_RUN_NAME>/check/checkpoint_9999.pt
```

Checkpoint filenames are zero-indexed: `--checkpoint 10000` corresponds to `checkpoint_9999.pt`.

Set a shell variable before running Stage-2:

```bash
cd EDGE_fairness_loss
export EDGE_STAGE1_RUN=replace_with_edge_stage1_run_name
export STAGE1_CKPT=../EDGE_fairness/wandb/cora/multinomial_diffusion/multistep/${EDGE_STAGE1_RUN}/check/checkpoint_9999.pt
```

For Citeseer, use `--dataset citeseer`, `--diffusion_steps 128`, `--batch_size 4`, `--edge_dropout 0.0`, and the
Citeseer Stage-1 checkpoint. For Amazon Photo, use smaller `--batch_size` and `--sample_batch_size` values to avoid GPU
OOM.

## Train One Controller

Run `train_controller.py` from this repository. This freezes the Stage-1 denoiser and trains only the per-step
fairness-controller parameters. The following examples use the default `--fair_score_metric sp`; the EO workflow
below uses separate `controller/eo/` directories.

```bash
cd EDGE_fairness_loss
export EDGE_CONTROLLER_RUN=replace_with_edge_controller_run_name

python train_controller.py \
  --name "$EDGE_CONTROLLER_RUN" \
  --controller_pretrained_ckpt "$STAGE1_CKPT" \
  --controller_epochs 500 \
  --controller_lr 1e-3 \
  --controller_replay_num_samples 2 \
  --controller_replay_refresh 10 \
  --num_generation 8 \
  --sample_batch_size 1 \
  --eval_every 100 \
  --check_every 100 \
  --diffusion_dim 128 \
  --diffusion_steps 256 \
  --edge_dropout 0.05 \
  --device cuda:0 \
  --dataset cora \
  --batch_size 2 \
  --clip_value 1 \
  --lr 5e-4 \
  --optimizer adam \
  --final_prob_edge 1 0 \
  --sample_time_method importance \
  --noise_schedule linear \
  --loss_type vb_ce_xt_prescribred_st \
  --parametrization xt_prescribed_st \
  --degree \
  --num_heads 8 8 8 8 1 \
  --use_node_feat \
  --fair_label_attr y \
  --fair_score_k 0.5 \
  --fair_score_eta 1e-6 \
  --fair_score_eta_scale 1.0 \
  --fair_score_fair_loss_weight 1.0 \
  --fair_score_k_tracking_loss_weight 1.0 \
  --fair_score_utility_loss_weight 0.1 \
  --fair_score_guidance_normalize True
```

Outputs are written to:

```text
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_RUN_NAME>/
```

Important output files:

```text
controller_metrics.jsonl
check/controller_last.pt
check/controller_best.pt
check/controller_final.pt
check/full_model_best.pt
generated_samples/controller_best.pyg_full.pt
generated_samples/controller_best.meta.json
```

## Evaluate One Controller Run

`train_controller.py` exports `generated_samples/controller_best.pyg_full.pt`. Evaluate it with the sample.py-compatible
GAE LP evaluator:

```bash
python evaluate_generated_graphs.py \
  --graph_path wandb/cora/multinomial_diffusion/controller/sp/${EDGE_CONTROLLER_RUN}/generated_samples/controller_best.pyg_full.pt \
  --dataset cora \
  --label_attr y \
  --sensitive_attr y \
  --device cuda:0
```

Default outputs are saved next to the graph file:

```text
controller_best.pyg_full.overlap_lp_gae_per_graph.csv
controller_best.pyg_full.overlap_lp_gae_summary.csv
```

The columns used for Pareto plots are usually:

```text
lp/auc_mean
lp/score_sp_abs_gap_mean
aggregate_lp/auc
aggregate_lp/score_sp_abs_gap
```

## Grid Search Controllers

Use `scripts/run_controller_grid.py` to sweep controller hyperparameters. With `--run_generated_eval`, the script runs
each controller, evaluates `controller_best.pyg_full.pt`, writes a grid summary CSV, and draws a Pareto curve.

```bash
export EDGE_CONTROLLER_GRID_PREFIX=replace_with_edge_controller_grid_prefix

python scripts/run_controller_grid.py \
  --repo_dir . \
  --stage1_ckpt "$STAGE1_CKPT" \
  --name_prefix "$EDGE_CONTROLLER_GRID_PREFIX" \
  --dataset cora \
  --device cuda:0 \
  --generated_eval_device cuda:0 \
  --diffusion_dim 128 \
  --diffusion_steps 256 \
  --edge_dropout 0.05 \
  --batch_size 2 \
  --num_generation 8 \
  --sample_batch_size 1 \
  --controller_epochs 500 \
  --controller_replay_num_samples 2 \
  --controller_replay_refresh 10 \
  --eval_every 100 \
  --check_every 100 \
  --fair_score_guidance_normalize True \
  --fair_score_k_values 0.3 0.5 0.7 \
  --eta_values 1e-6 3e-6 \
  --controller_lrs 1e-3 \
  --fair_weights 0.5 1.0 \
  --utility_weights 0.05 0.1 \
  --k_tracking_weights 1.0 \
  --run_generated_eval \
  --pareto_x_metric lp/score_sp_abs_gap_mean \
  --pareto_y_metric lp/auc_mean \
  --pareto_label_points front
```

Expected grid outputs:

```text
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_GRID_PREFIX>_manifest.jsonl
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_GRID_PREFIX>_summary.csv
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.jpg
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.front.csv
```

Each individual grid run is stored under:

```text
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_GRID_PREFIX>_*/
```

The per-run directory name includes an automatic `norm` or `raw` tag after the prefix.

Useful grid flags:

- `--dry_run`: print all commands without running them.
- `--skip_existing`: skip training when `check/controller_final.pt` exists; with `--run_generated_eval`,
  still evaluate missing LP summaries. A missing generated graph is reported as an evaluation failure.
- `--max_runs N`: run only the first `N` combinations for smoke tests.
- `--fail_fast`: stop at the first controller or generated-evaluation failure.
- `--force_generated_eval`: rerun generated-graph LP evaluation even if the summary CSV already exists.

## Summarize and Plot Manually

If you ran the grid without `--run_generated_eval`, first evaluate the generated graphs for each run. For one run:

```bash
export EDGE_CONTROLLER_GRID_RUN_DIR=replace_with_edge_controller_grid_run_dir

python evaluate_generated_graphs.py \
  --graph_path "${EDGE_CONTROLLER_GRID_RUN_DIR}/generated_samples/controller_best.pyg_full.pt" \
  --dataset cora \
  --label_attr y \
  --sensitive_attr y \
  --device cuda:0
```

Then summarize all controller runs with the same prefix:

```bash
python scripts/summarize_controller_grid.py \
  --controller_root wandb/cora/multinomial_diffusion/controller/sp \
  --fair_score_metric sp \
  --prefix "$EDGE_CONTROLLER_GRID_PREFIX" \
  --sort_by lp/score_sp_abs_gap_mean \
  --out_csv wandb/cora/multinomial_diffusion/controller/sp/${EDGE_CONTROLLER_GRID_PREFIX}_summary.csv
```

Draw the LP AUC vs score-SP Pareto curve:

```bash
python scripts/plot_controller_grid_pareto.py \
  --summary_csv wandb/cora/multinomial_diffusion/controller/sp/${EDGE_CONTROLLER_GRID_PREFIX}_summary.csv \
  --out_path wandb/cora/multinomial_diffusion/controller/sp/${EDGE_CONTROLLER_GRID_PREFIX}_pareto_lp_auc_vs_score_sp.jpg \
  --x_metric lp/score_sp_abs_gap_mean \
  --y_metric lp/auc_mean \
  --label_points front \
  --title "cora: Controller LP Pareto"
```

The plot script also writes the Pareto front rows to:

```text
wandb/cora/multinomial_diffusion/controller/sp/<EDGE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.front.csv
```

## Notes

- By default, `fair_score_k_raw` and `fair_score_eta_raw` are per-step vectors of length `diffusion_steps`.
- `--fair_score_k` and `--fair_score_eta` initialize those values. In the current replay objective, k learns
  through the tracking loss; setting its weight to zero leaves k effectively fixed at initialization.
- `--fair_score_eta_mode shared` learns one eta across steps; `--fair_score_k_mode fixed_one` fixes k exactly to 1.
- When `--fair_score_guidance_normalize True` is used, small eta initializations such as `1e-6` to `3e-6` are reasonable
  starting points for compact searches.
- The LP Pareto curve minimizes `lp/score_sp_abs_gap_mean` on the x-axis and maximizes `lp/auc_mean` on the y-axis.

## EO Controller and Evaluation

Choose `--fair_score_metric eo` for equal opportunity, or `sp` for statistical parity. Both compare pairs whose
endpoint labels are equal against pairs whose endpoint labels differ (`--fair_label_attr y` by default).
The EO training surrogate is the signed difference between `sum(w*q)/sum(w)` in these two groups. Here `q` is
the controller's running edge probability and `w` is a separate, detached unguided denoiser probability used as a
soft positive condition. It is not replaced with the guided probability. The fairness loss is half the squared
gap, averaged over valid graphs, alongside the configured utility and k-tracking losses.

`--fair_score_eo_min_mass` defaults to `1e-6`. Both groups must contain pairs and have soft-positive mass greater
than this threshold. Invalid graphs contribute zero fairness gap/gradient and are counted in
`fair_controller_invalid_graph_fraction`; inspect that value along with `fair_controller_valid_graphs` and the
`fair_controller_eo_positive_mass_*` fields. A zero masked training gap is not evidence of downstream fairness.

The downstream evaluator measures the difference in mean GAE scores on actual positive held-out pairs of each
generated graph. Negative pairs do not enter EO. This score-based equal-opportunity measure differs from both
the soft training surrogate and equalized odds. Missing positive pairs in either group produce an undefined
EO (`NaN`, `eo_defined=0`). Evaluation reports SP and EO for either controller objective; no extra evaluator
objective flag is needed.

| Summary column | Meaning |
| --- | --- |
| `lp/auc_mean` | Mean generated-graph held-out LP AUC |
| `lp/eo_abs_gap_mean` | Mean absolute EO score gap; lower is better |
| `lp/eo_signed_gap_mean` | Mean signed same-group minus different-group score gap |
| `lp/eo_defined_mean` | Fraction of graphs with positive pairs in both groups |
| `aggregate_lp/eo_abs_gap` | Absolute gap after pooling evaluated pairs across graphs |

Always inspect EO validity; the mean over finite gaps alone can hide undefined graphs. `aggregate_lp/*` and
per-graph means use different aggregation. The GAE evaluator uses its fixed 80/10/10 split and existing
validation-AUC-selected hyperparameter grid; legacy `--lp_*` options do not replace that protocol.

All commands below run from `FairShift/EDGE_fairness_loss`. Activate an environment with the repository
dependencies and set the paths for your data/checkpoint. This setup limits the current shell and its children
to one allowed CPU; the launchers also expose their own CPU limits.

```bash
export EDGE_PYTHON="$(command -v python)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export EDGE_CPU_CORES=1 EDGE_CPU_THREADS=1
EDGE_CPU_FIRST=$(awk '/Cpus_allowed_list/ {split($2, ranges, /[,-]/); print ranges[1]}' /proc/self/status)
taskset -pc "$EDGE_CPU_FIRST" "$$"
export CUDA_VISIBLE_DEVICES=0
export STAGE1_CKPT=/absolute/path/to/your/cora_stage1_checkpoint.pt
```

`CUDA_VISIBLE_DEVICES` selects the physical GPU. The GAE evaluator internally uses logical `cuda:0` whenever a
GPU is visible, so set visibility explicitly even when passing `--device`. Set `CUDA_VISIBLE_DEVICES=''` and
use `--device cpu` for a CPU run. The example architecture below matches the Cora Stage-1 example above;
change all relevant architecture options together when using another checkpoint.

Train one EO controller:

```bash
export EDGE_EO_RUN=cora_eo_example
"$EDGE_PYTHON" train_controller.py \
  --name "$EDGE_EO_RUN" --controller_pretrained_ckpt "$STAGE1_CKPT" \
  --dataset cora --device cuda:0 --seed 0 --generation_seed 0 \
  --controller_epochs 500 --controller_lr 1e-3 \
  --controller_replay_num_samples 2 --controller_replay_refresh 10 \
  --num_generation 8 --sample_batch_size 1 --eval_every 100 --check_every 100 \
  --diffusion_dim 128 --diffusion_steps 256 --edge_dropout 0.05 --batch_size 2 \
  --clip_value 1 --lr 5e-4 --optimizer adam \
  --final_prob_edge 1 0 --sample_time_method importance --noise_schedule linear \
  --loss_type vb_ce_xt_prescribred_st --parametrization xt_prescribed_st \
  --degree --num_heads 8 8 8 8 1 --use_node_feat --fair_label_attr y \
  --fair_score_metric eo --fair_score_eo_min_mass 1e-6 \
  --fair_score_k 0.5 --fair_score_eta 1e-6 --fair_score_eta_scale 1 \
  --fair_score_fair_loss_weight 1 --fair_score_k_tracking_loss_weight 1 \
  --fair_score_utility_loss_weight 0.1 --fair_score_guidance_normalize True

"$EDGE_PYTHON" evaluate_generated_graphs.py \
  --graph_path "wandb/cora/multinomial_diffusion/controller/eo/${EDGE_EO_RUN}/generated_samples/controller_best.pyg_full.pt" \
  --dataset cora --label_attr y --sensitive_attr y --device cuda:0 --seed 0
```

The full path is `wandb/<dataset>/multinomial_diffusion/controller/<sp|eo>/<name>/`. With
`--controller_root PATH`, it becomes `PATH/<sp|eo>/<name>/`. Checkpoints and generated graphs record the chosen
metric. `evaluate.py` restores the saved metric; old checkpoints without that metadata default to SP.

## EO Grid, Uncontrolled Baseline, and Operating Points

This complete grid example uses the same Cora architecture, evaluates generated graphs, and exports a separate
uncontrolled Stage-1 baseline. The small normalized-guidance grid is an example to adapt, not a selected result.
Add `--dry_run` before the final `--` to inspect commands without training or evaluation. A grid dry run writes
its command manifest; use a distinct prefix or `--manifest` path when keeping experiment records.

```bash
export EDGE_EO_PREFIX=cora_eo_grid_example
"$EDGE_PYTHON" scripts/run_controller_grid.py \
  --repo_dir . --python_exec "$EDGE_PYTHON" \
  --stage1_ckpt "$STAGE1_CKPT" --name_prefix "$EDGE_EO_PREFIX" \
  --dataset cora --device cuda:0 --generated_eval_device cuda:0 \
  --diffusion_dim 128 --diffusion_steps 256 --edge_dropout 0.05 --batch_size 2 \
  --noise_schedule linear --loss_type vb_ce_xt_prescribred_st \
  --parametrization xt_prescribed_st --num_heads 8 8 8 8 1 \
  --num_generation 8 --sample_batch_size 1 \
  --controller_epochs 500 --controller_replay_num_samples 2 --controller_replay_refresh 10 \
  --eval_every 100 --check_every 100 --clip_value 1 --lr 5e-4 --optimizer adam \
  --fair_score_metric eo --fair_score_eo_min_mass 1e-6 --fair_score_guidance_normalize True \
  --fair_score_k_values 0.3 0.5 --eta_values 1e-6 3e-6 --controller_lrs 1e-3 \
  --fair_weights 1 --utility_weights 0.1 --k_tracking_weights 1 \
  --include_uncontrolled --generation_seed 0 --generated_eval_seed 0 \
  --run_generated_eval --skip_existing --fail_fast --pareto_label_points front \
  -- --seed 0 --fair_label_attr y
```

The grid adds the Stage-1 `--degree`, `--use_node_feat`, final edge probabilities, and importance-sampling flags
shown in the single-controller example. EO defaults to `lp/eo_abs_gap_mean` on the Pareto x-axis when
`--pareto_x_metric` is omitted; do not carry over an explicitly selected SP axis. Summary/plot scripts also
accept `--fair_score_metric eo`.

The EO directory contains `${EDGE_EO_PREFIX}_manifest.jsonl`, `${EDGE_EO_PREFIX}_summary.csv`, and
`${EDGE_EO_PREFIX}_pareto_lp_auc_vs_eo.jpg`. Candidate directories start with the prefix; the baseline is stored
separately as `uncontrolled_${EDGE_EO_PREFIX}/`. It loads the same Stage-1 checkpoint and disables both controller
training and guidance. `--generation_seed` resets the RNG immediately before graph export for every arm;
`--generated_eval_seed` controls the downstream evaluator, independently of the controller training seed.
Matching seeds do not make the generated graphs identical. Omitting `--generation_seed` preserves the previous
RNG behavior. On resume, existing uncontrolled graph metadata must match the requested checkpoint, graph count,
generation seed, and disabled-guidance state.

Select and plot operating points against that separately evaluated baseline:

```bash
EDGE_EO_ROOT=wandb/cora/multinomial_diffusion/controller/eo
"$EDGE_PYTHON" scripts/select_controller_operating_points.py \
  --dataset cora \
  --summary_csv "$EDGE_EO_ROOT/${EDGE_EO_PREFIX}_summary.csv" \
  --uncontrolled_summary_csv "$EDGE_EO_ROOT/uncontrolled_${EDGE_EO_PREFIX}/generated_samples/controller_best.pyg_full.overlap_lp_gae_summary.csv" \
  --auc_max_drop 0.01 \
  --out_csv "results/${EDGE_EO_PREFIX}_operating_points.csv" \
  --plot_path "results/${EDGE_EO_PREFIX}_operating_points.png"
```

`auc_retained` minimizes EO among runs that improve baseline EO while losing at most 0.01 absolute AUC.
`fairness_oriented` minimizes EO among improving runs, with no AUC floor unless `--fairness_max_auc_drop` is set.
Rows with non-finite metrics or reported undefined EO are excluded; an empty eligible set yields
`no_qualifying_run`. These are descriptive selections from the supplied evaluation summaries. Use independent
validation data to select settings before reporting performance on a separate final test set; the selector
does not create an independent split or guarantee an EO improvement.

## Supplied EO Launchers

The shell launchers package larger existing search configurations. They accept a physical GPU index and optional
`--dry_run`, use `EDGE_PYTHON` (default: Python on `PATH`), and require an explicit `STAGE1_CKPT` for a real run.
`EDGE_CPU_CORES` and `EDGE_CPU_THREADS` default to 1; `EDGE_CPU_LIST` can select a specific allowed CPU set.
Each child process inherits the launcher's CPU affinity and numeric-library thread limits.

| Launcher | Arguments | Search |
| --- | --- | --- |
| `scripts/run_eo_grid_v5.sh` | `{cora|citeseer} PHYSICAL_GPU [--dry_run]` | EO controller grid with tracking weight 0.1 |
| `scripts/run_eo_grid_kw0_reduced.sh` | `{cora|citeseer} PHYSICAL_GPU [--dry_run]` | Reduced grid with tracking weight 0 |
| `scripts/run_cora_eo_kw0_refined.sh` | `PHYSICAL_GPU [--dry_run]` | Cora refinement with tracking weight 0 |
| `scripts/run_citeseer_eo_kw0_refined.sh` | `PHYSICAL_GPU [--dry_run]` | Citeseer refinement with tracking weight 0 |

```bash
EDGE_PYTHON="$(command -v python)" EDGE_CPU_CORES=1 EDGE_CPU_THREADS=1 \
  bash scripts/run_eo_grid_v5.sh cora 0 --dry_run

STAGE1_CKPT=/absolute/path/to/compatible/cora_checkpoint.pt \
  EDGE_PYTHON="$(command -v python)" EDGE_CPU_CORES=1 EDGE_CPU_THREADS=1 \
  bash scripts/run_cora_eo_kw0_refined.sh 0
```

Dry runs need neither checkpoint files nor graph data and launch no training/evaluation. Real runs also require
the matching `graphs/<dataset>_feat.pkl`. These launchers use **unnormalized** guidance, `edge_dropout=0`, Cora
256/Citeseer 128 diffusion steps, and their encoded search ranges; verify them against the checkpoint before
running. Do not substitute their eta ranges into the normalized example above. Output paths remain under
`wandb/` and `results/`; historical scores and selected operating points are not included in this repository.

## Direct-Gap Diagnostics, Ablations, and Proxy Runner

`run_direct_gap.py` observes the sampler's actual pair cache and final committed probabilities, then joins pair
IDs to the downstream evaluator. It decomposes signed SP gaps into support/score terms, and EO gaps into
support/conditioning/score terms when the actual detached conditioning weights were recorded. The observer is
off by default, adds no training loss, and does not reconstruct missing EO weights from `q`.

The synthetic CPU smoke requires no external checkpoint or dataset. Use fresh output directories:

```bash
CUDA_VISIBLE_DEVICES='' "$EDGE_PYTHON" -B run_direct_gap.py toy-smoke \
  --cpu-threads 1 --output results/direct_gap_toy_NEW
CUDA_VISIBLE_DEVICES='' "$EDGE_PYTHON" -B run_direct_gap.py analyze \
  --cpu-threads 1 --manifest results/direct_gap_toy_NEW/offline_manifest.json \
  --output results/direct_gap_offline_NEW
```

For real data, adapt `configs/direct_gap_cora_sp.json` with matching graph, Stage-1 arguments/weights, controller
provenance, seeds, and device. Its unresolved controller entry must be replaced with a verified compatible
controller. Run `preflight --manifest configs/YOUR_MANIFEST.json --output results/preflight_NEW` first;
`observe` generates and evaluates one graph with one evaluator seed; `batch` executes all listed configurations
and seeds. The example manifest is not a runnable
experiment result. Existing learned-k pilot weights are not automatically compatible with fixed-k diagnostics.

Controller ablations reuse the exact settings of an existing run, with shared eta and/or k fixed to one:

```bash
"$EDGE_PYTHON" scripts/run_controller_ablations.py \
  --baseline_run "wandb/cora/multinomial_diffusion/controller/eo/${EDGE_EO_RUN}" \
  --output_root results/controller_ablations_NEW \
  --variants shared_eta k_one shared_eta_k_one --seeds 0 \
  --generation_seed 0 --device cuda:0 --run_generated_eval --dry_run
```

The baseline needs its own `args.pickle`; remove `--dry_run` to execute after supplying the corresponding assets.
For matched fixed-k diagnostics, `configs/direct_gap_k_ablation.example.json` provides the separate calibration
schema: use `run_direct_gap.py prepare-k-ablation --manifest ... --output ...` to prepare it. The corresponding
`calibrate-k-ablation` command performs training and requires `--allow-calibration`.

`run_proxy_minimal.py` is a separate Cora/EDGE terminal-GCN protocol, with explicit controller/GCN selection
provenance and an optional prespecified pilot. Its metrics must not be pooled with the default GAE-grid results.
Supply your own JSON spec; the historical default `results/proxy_minimal/run_spec.json` is not shipped:

```bash
CUDA_VISIBLE_DEVICES='' "$EDGE_PYTHON" -B run_proxy_minimal.py \
  --spec configs/YOUR_PROXY_SPEC.json --dry-run
```

The spec must provide `dataset`, graph and backbone paths, saved backbone arguments, SP/EO controller checkpoints
and arguments, root seeds, and the GCN configuration with selection evidence. Paths must resolve inside this
directory or sibling `EDGE_fairness`; FairWire assets are outside this runner's scope. The default mode performs
CPU preflight and writes its audit under `results/proxy_minimal/`. `--execute` runs the configured 30-graph
experiment; the prespecified pilot additionally uses `--smoke` and checks its recorded preparation. Existing
artifact directories cannot be overwritten or resumed by this runner. No prior specs, weights, audits, or
experiment outputs are bundled here.

The historical proxy runner retains a physical-GPU-4 contract: its spec must use `"device": "cuda:4"` even
for CPU preflight. Actual `--execute`/`--smoke` requires `CUDA_VISIBLE_DEVICES` to be **unset**, CUDA availability,
and at least five physical GPUs. The prespecified controller-training path has the same addressing restriction.
The generic EO setup above with `CUDA_VISIBLE_DEVICES=0` cannot be reused for these historical execution modes;
CPU preflight does not itself launch GPU work.

Existing CPU tests exercise EO conditioning, interfaces, baseline exports, operating-point selection, and the
diagnostic pipelines without external assets:

```bash
CUDA_VISIBLE_DEVICES='' "$EDGE_PYTHON" -B -m unittest discover -s tests -v
```

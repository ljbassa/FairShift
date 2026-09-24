# EDGE Fairness

PyTorch implementation based on
["Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling"](https://arxiv.org/pdf/2305.04111.pdf).
This repository extends EDGE with fairness-guided graph generation and link-prediction fairness evaluation.

The code is developed from https://github.com/ehoogeboom/multinomial_diffusion and uses evaluation modules from
https://github.com/uoguelph-mlrg/GGM-metrics and https://github.com/hheidrich/CELL.

## Installation

Install dependencies from the requirement files. `requirements.txt` includes the pinned CUDA/Torch stack used in this
repository, while `requirements.rest.txt` keeps the non-Torch dependencies separate for environments where Torch is
installed manually.

```bash
pip install -r requirements.txt
```

## Data

The training commands below expect NetworkX pickle files under `graphs/`. Each pickle stores an undirected graph with
node attributes used by training and fairness evaluation:

- `x`: node feature vector
- `y`: node label used as the default sensitive/group attribute
- `orig_id`: original PyG node id

Create the graph pickle files with `datasets/make_planetoid_pickle.py`. The script uses PyTorch Geometric datasets, so
the first run downloads raw data into `data/`.

```bash
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

Expected outputs:

```text
graphs/cora_feat.pkl
graphs/citeseer_feat.pkl
graphs/amazon_photo_feat.pkl
```

For quick smoke tests, add `--max_nodes <N>` to save an induced subgraph with the first `N` nodes. The same script also
supports `pubmed`, `cornell`, `texas`, `wisconsin`, `amazon_computer`, and `amazon_computers`.

## Training

Change `--device` as needed for your machine. Each command writes checkpoints to
`wandb/<dataset>/multinomial_diffusion/multistep/<name>/check/`.
Replace placeholder run names such as `<EDGE_STAGE1_RUN_NAME>` before running the commands.

`--checkpoint 10000` in evaluation loads `check/checkpoint_9999.pt`, because checkpoint filenames are zero-indexed.

### Cora

```bash
export EDGE_STAGE1_RUN=replace_with_edge_stage1_run_name

python train.py \
  --name "$EDGE_STAGE1_RUN" \
  --epochs 10000 \
  --num_generation 8 \
  --num_iter 32 \
  --diffusion_dim 128 \
  --diffusion_steps 256 \
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
  --log_wandb False
```

### Citeseer

```bash
export EDGE_STAGE1_RUN=replace_with_edge_stage1_run_name

python train.py \
  --name "$EDGE_STAGE1_RUN" \
  --epochs 10000 \
  --num_generation 8 \
  --num_iter 32 \
  --diffusion_dim 128 \
  --diffusion_steps 128 \
  --device cuda:0 \
  --dataset citeseer \
  --batch_size 4 \
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
  --log_wandb False
```

### Amazon Photo

```bash
export EDGE_STAGE1_RUN=replace_with_edge_stage1_run_name

python train.py \
  --name "$EDGE_STAGE1_RUN" \
  --epochs 10000 \
  --num_generation 4 \
  --eval_num_generation 2 \
  --test_num_generation 4 \
  --sample_batch_size 1 \
  --num_iter 16 \
  --diffusion_dim 128 \
  --diffusion_steps 128 \
  --device cuda:0 \
  --dataset amazon_photo \
  --batch_size 1 \
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
  --log_wandb False
```

## LP AUC vs Score-SP Pareto Evaluation

The example below uses the trained Cora run and enables normalized score-SP guidance with
`--fair_score_guidance_normalize True`. It evaluates a grid of `(eta, k)` values, writes a long summary CSV, and draws
the Pareto curve using `lp/auc_mean` as the score to maximize and `lp/score_sp_abs_gap_mean` as the fairness gap to
minimize.

```bash
python fair_grid_eval_generated_graphs.py \
  --repo_dir . \
  --dataset cora \
  --run_name "$EDGE_STAGE1_RUN" \
  --checkpoint 10000 \
  --num_samples 8 \
  --eta_values 0.005 0.01 0.015 0.02 \
  --k_values 0.3 0.5 0.7 \
  --seeds 0 1 2 \
  --fair_score_guidance_normalize True \
  --auc_candidates lp/auc_mean aggregate_lp/auc \
  --sp_candidates lp/score_sp_abs_gap_mean aggregate_lp/score_sp_abs_gap \
  --gen_device cuda:0 \
  --lp_device cuda:0 \
  --fair_sensitive_attr y \
  --fair_edge_sensitive_mode either \
  --largest_cc False \
  --lp_model gcn \
  --lp_epochs 200 \
  --out_dir fair_grid_generated_lp_norm_cora
```

Expected outputs:

```text
fair_grid_generated_lp_norm_cora/sp/summary_long.csv
fair_grid_generated_lp_norm_cora/sp/summary_long_cora.csv
fair_grid_generated_lp_norm_cora/sp/pareto_curve_cora.jpg
```

The CSV files used for the LP AUC vs score-SP table are produced by the command above. The per-run CSV files are under
`fair_grid_generated_lp_norm_cora/sp/evaluated_graphs/`, and the aggregate CSVs are the `summary_long*.csv`
files in the `sp/` directory.

To keep a short generic filename as well:

```bash
cp fair_grid_generated_lp_norm_cora/sp/summary_long_cora.csv \
  fair_grid_generated_lp_norm_cora/sp/summary.csv
```

You can redraw only the Pareto curve from an existing summary CSV without regenerating graphs:

```bash
python fair_grid_eval_generated_graphs.py \
  --repo_dir . \
  --dataset cora \
  --summary_csv fair_grid_generated_lp_norm_cora/sp/summary_long_cora.csv \
  --auc_candidates lp/auc_mean aggregate_lp/auc \
  --sp_candidates lp/score_sp_abs_gap_mean aggregate_lp/score_sp_abs_gap \
  --out_dir fair_grid_generated_lp_norm_cora
```

## Results

Training outputs are stored under:

```text
wandb/<dataset>/multinomial_diffusion/multistep/<run_name>/
```

## Equal opportunity (EO) guidance

This folder implements fixed sampling-time guidance (EDGE-cond FairShift-F). SP remains the default.
`evaluate.py`, `fair_grid_eval.py`, and `fair_grid_eval_generated_graphs.py` accept
`--fair_score_metric sp` or `--fair_score_metric eo`. For direct generation with `evaluate.py`, enable the
existing guidance switch `--fair_score_sp` together with `--fair_score_metric eo`; the switch name is retained
for compatibility with older run arguments.

The differentiable EO surrogate compares edges joining nodes with the same label against edges joining nodes
with different labels. For either group, its conditional score is `sum(w * q) / sum(w)`, where `q` is the guided
running edge probability and `w` is the detached running probability from the unguided denoiser logits.
The guidance uses the signed difference between the two conditional scores. This soft-positive condition is
separate from the held-out link-prediction EO metric reported by `evaluate_generated_graphs.py`.

`--fair_score_eo_min_mass` defaults to `1e-6`. A graph receives no EO correction when either group is empty or
either group's positive mass is at or below that threshold. No pseudo-count is added to make an unsupported
group appear valid. `--fair_score_guidance_normalize True` normalizes the guidance gradient, while
`--fair_score_eta` and `--fair_score_k` control guidance strength and the running-logit update.

### EO grid and uncontrolled baseline

Run from this folder after creating `graphs/cora_feat.pkl` and preparing a stage-1 EDGE run. Set
`EDGE_STAGE1_DIR` to the directory containing `args.pickle` and `check/`; it may be outside this checkout.
The example below requires `check/checkpoint_9999.pt` because `--checkpoint 10000` uses a zero-indexed filename.
Choose an allowed CPU core for `EDGE_CPU` and use the Python environment with the dependencies above installed.

```bash
export EDGE_STAGE1_DIR=/absolute/path/to/edge_stage1_run
export EDGE_CPU=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

taskset -c "$EDGE_CPU" python fair_grid_eval_generated_graphs.py \
  --repo_dir . \
  --dataset cora \
  --run_dir "$EDGE_STAGE1_DIR" \
  --checkpoint 10000 \
  --fair_score_metric eo \
  --fair_score_eo_min_mass 1e-6 \
  --fair_score_guidance_normalize True \
  --eta_values 0.0005 0.001 0.0025 0.005 0.01 0.02 \
  --k_values 0.1 0.3 0.5 0.7 \
  --include_baseline --baseline_k 1 \
  --num_samples 8 --seeds 0 1 2 \
  --gen_device cuda:0 --lp_device cuda:0 \
  --fair_sensitive_attr y --largest_cc False --graph_variant full \
  --lp_model gcn --lp_epochs 200 \
  --auc_candidates lp/auc_mean --eo_candidates lp/eo_abs_gap_mean \
  --out_dir results/fixed_eo_grid/cora
```

This evaluates 24 guided settings plus one uncontrolled setting for each seed. At `eta=0`, the grid explicitly
sets `fair_score_apply_sample=False`; nonzero settings explicitly enable it. The grid also overrides saved
normalization settings, so an older checkpoint's arguments do not silently change the requested experiment.

Both grid drivers separate outputs into `<out_dir>/sp/` and `<out_dir>/eo/`. For this example, the EO summary is
`results/fixed_eo_grid/cora/eo/summary_long_cora.csv` and the plot is
`results/fixed_eo_grid/cora/eo/pareto_curve_cora.jpg`. The grid passes generated graphs to the evaluator in memory.
Direct generation with sample saving enabled writes to `<save_dir or run_dir/generated_samples>/<sp|eo>/`.
Generated data, checkpoints, CSVs, figures, and logs are excluded from this aggregate repository.

### Select EO operating points

The selector compares matched seed sets and graph counts against the uncontrolled `(eta=0, k=1)` baseline.
It rejects failed, incomplete, non-finite, or explicitly undefined EO measurements. `auc_retained` selects the
lowest EO gap among improvements with at most `--auc_max_drop` loss in AUC; `fairness_oriented` selects the lowest
EO gap among improvements without that AUC constraint. If no run qualifies, the corresponding row records
`no_qualifying_run`. Means and population standard deviations summarize per-seed graph means.

```bash
taskset -c "$EDGE_CPU" python scripts/select_fixed_eo_operating_points.py \
  --dataset cora \
  --summary_csv results/fixed_eo_grid/cora/eo/summary_long_cora.csv \
  --auc_max_drop 0.01 \
  --out_csv results/fixed_eo_grid/cora/eo/operating_points_cora.csv
```

`scripts/run_fixed_eo_grid.sh` combines the same grid and selector for Cora or Citeseer. It uses `python` from
the active environment by default; set `EDGE_PYTHON` to override it. Set `EDGE_RUN_DIR` to the prepared stage-1
directory and `EDGE_CHECKPOINT` to the checkpoint argument (default `10000`). `EDGE_OUT_DIR` overrides the result
directory, and `EDGE_CPU_THREADS` defaults to `1`. The GPU argument selects the physical device, exposed as
`cuda:0` inside the process. The dry run validates input paths and grid settings without sampling or LP training.

```bash
export EDGE_RUN_DIR="$EDGE_STAGE1_DIR"
export EDGE_CHECKPOINT=10000
taskset -c "$EDGE_CPU" bash scripts/run_fixed_eo_grid.sh cora 0 --dry_run
# Remove --dry_run to generate graphs, evaluate them, and select operating points.
```

### EO regression checks

The tests cover the weighted surrogate and its derivative, invalid-group behavior, SP compatibility,
checkpoint argument overrides, and operating-point selection. They do not run full graph generation or training.

```bash
taskset -c "$EDGE_CPU" python -m unittest discover -s tests -v
```

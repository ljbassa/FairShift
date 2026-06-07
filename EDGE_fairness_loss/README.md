# EDGE Fairness Loss

This repository trains a Stage-2 fairness controller on top of a Stage-1 EDGE denoising model trained in
`EDGE_fairness`. The Stage-2 controller learns per-step fairness-guidance parameters, exports generated graphs, and
supports grid search with LP AUC vs score-SP Pareto summaries.

Use this repository after you already have a compatible Stage-1 checkpoint.

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
fairness-controller parameters.

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
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_RUN_NAME>/
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
  --graph_path wandb/cora/multinomial_diffusion/controller/${EDGE_CONTROLLER_RUN}/generated_samples/controller_best.pyg_full.pt \
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
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_GRID_PREFIX>_manifest.jsonl
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_GRID_PREFIX>_summary.csv
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.jpg
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.front.csv
```

Each individual grid run is stored under:

```text
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_GRID_PREFIX>_*/
```

The per-run directory name includes an automatic `norm` or `raw` tag after the prefix.

Useful grid flags:

- `--dry_run`: print all commands without running them.
- `--skip_existing`: skip runs that already have `check/controller_final.pt`.
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
  --controller_root wandb/cora/multinomial_diffusion/controller \
  --prefix "$EDGE_CONTROLLER_GRID_PREFIX" \
  --sort_by lp/score_sp_abs_gap_mean \
  --out_csv wandb/cora/multinomial_diffusion/controller/${EDGE_CONTROLLER_GRID_PREFIX}_summary.csv
```

Draw the LP AUC vs score-SP Pareto curve:

```bash
python scripts/plot_controller_grid_pareto.py \
  --summary_csv wandb/cora/multinomial_diffusion/controller/${EDGE_CONTROLLER_GRID_PREFIX}_summary.csv \
  --out_path wandb/cora/multinomial_diffusion/controller/${EDGE_CONTROLLER_GRID_PREFIX}_pareto_lp_auc_vs_score_sp.jpg \
  --x_metric lp/score_sp_abs_gap_mean \
  --y_metric lp/auc_mean \
  --label_points front \
  --title "cora: Controller LP Pareto"
```

The plot script also writes the Pareto front rows to:

```text
wandb/cora/multinomial_diffusion/controller/<EDGE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.front.csv
```

## Notes

- `fair_score_k_raw` and `fair_score_eta_raw` are learned per-step vectors of length `diffusion_steps`.
- `--fair_score_k` and `--fair_score_eta` initialize the controller values; they are not fixed global values.
- When `--fair_score_guidance_normalize True` is used, small eta initializations such as `1e-6` to `3e-6` are reasonable
  starting points for compact searches.
- The LP Pareto curve minimizes `lp/score_sp_abs_gap_mean` on the x-axis and maximizes `lp/auc_mean` on the y-axis.

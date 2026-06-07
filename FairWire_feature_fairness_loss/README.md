# FairWire Feature Fairness Loss Experiments

This folder trains and evaluates the Stage-2 fairness controller for the
node-conditioned FairWire feature model. Dependency details are kept in
`requirements.txt`.

```bash
pip install -r requirements.txt
```

Run the commands below from the `FairWire_feature_fairness_loss` directory.

## 1. Required Inputs

Before training a controller, prepare:

- A Stage-1 node-conditioned FairWire checkpoint, usually trained with
  `aA=0.0` and `aX=0.0`.
- A reference graph pickle at `graphs/<dataset>_feat.pkl` if you want automatic
  LP AUC and score-SP evaluation.

Create the reference graphs with:

```bash
mkdir -p graphs

python make_reference_graph.py --dataset cora --out_path graphs/cora_feat.pkl
python make_reference_graph.py --dataset citeseer --out_path graphs/citeseer_feat.pkl
python make_reference_graph.py --dataset amazon_photo --out_path graphs/amazon_photo_feat.pkl
```

## 2. Bring In a FairWire_feature Checkpoint

The Stage-1 model can be trained in `FairWire_feature` and reused here. Copy or
symlink the whole checkpoint directory so the checkpoint path stays easy to
read.

Copy example:

```bash
cp -a ../FairWire_feature/cora_0.0_0.0_cpts ./
cp -a ../FairWire_feature/citeseer_0.0_0.0_cpts ./
cp -a ../FairWire_feature/amazon_photo_0.0_0.0_cpts ./
```

Symlink example:

```bash
ln -s ../FairWire_feature/cora_0.0_0.0_cpts cora_0.0_0.0_cpts
```

Then point the controller to the copied or linked checkpoint file:

```bash
export FEATURE_STAGE1_AA0_CKPT=cora_0.0_0.0_cpts/Sync_T3.pth
```

If the Stage-1 model was trained with a different diffusion step count, set the
matching file name instead:

```bash
export FEATURE_STAGE1_AA0_CKPT=cora_0.0_0.0_cpts/Sync_T8.pth
```

The checkpoint dataset must match the evaluation dataset. For example, use a
Cora checkpoint with Cora reference graphs.

## 3. Train One Controller and Evaluate LP Metrics

The example below starts from the `aA=0, aX=0` checkpoint, trains the controller
with normalized guidance, generates graphs from `controller_best`, and runs LP
evaluation automatically.

```bash
export FW_FEATURE_CONTROLLER_RUN=replace_with_fw_feature_controller_run_name

python train_controller.py \
  --controller_pretrained_ckpt "$FEATURE_STAGE1_AA0_CKPT" \
  --name "$FW_FEATURE_CONTROLLER_RUN" \
  --log_home ./wandb \
  --device cuda:0 \
  --seed 0 \
  --controller_epochs 1000 \
  --controller_lr 1e-3 \
  --controller_replay_num_samples 1 \
  --controller_replay_refresh 100 \
  --num_generation 64 \
  --sample_batch_size 32768 \
  --fair_label_attr y \
  --fair_score_eta 0.005 \
  --fair_score_k 0.15 \
  --fair_score_learn_k False \
  --fair_score_learn_eta True \
  --fair_score_guidance_normalize True \
  --fair_score_fair_loss_weight 1e5 \
  --fair_score_k_tracking_loss_weight 0.0 \
  --fair_score_utility_loss_weight 0.1 \
  --run_generated_eval \
  --generated_eval_max_graphs 64 \
  --generated_eval_lp_epochs 1000 \
  --generated_eval_lp_patience 5 \
  --generated_eval_lp_batch_size 16384
```

Typical outputs are written under:

```text
wandb/cora/Sync/controller/<FW_FEATURE_CONTROLLER_RUN_NAME>/
```

Important files include:

- `controller_metrics.jsonl`
- `check/controller_best.pt`
- `check/full_model_best.pt`
- `generated_samples/controller_best.pyg_full.pt`
- `generated_samples/controller_best.pyg_full.overlap_lp_gae_summary.csv`

The generated LP summary contains the main metrics:

- `lp/auc_mean`
- `lp/score_sp_abs_gap_mean`
- `aggregate_lp/auc`
- `aggregate_lp/score_sp_abs_gap`

## 4. Controller Grid Search

Use `scripts/run_controller_grid.py` to sweep controller hyperparameters. With
`--run_generated_eval`, the script trains every controller, evaluates generated
graphs, writes a summary CSV, and plots the LP AUC versus score-SP Pareto curve.

```bash
export FW_FEATURE_CONTROLLER_GRID_PREFIX=replace_with_fw_feature_controller_grid_prefix

python scripts/run_controller_grid.py \
  --repo_dir . \
  --stage1_ckpt "$FEATURE_STAGE1_AA0_CKPT" \
  --name_prefix "$FW_FEATURE_CONTROLLER_GRID_PREFIX" \
  --controller_root ./wandb/cora/Sync/controller \
  --device cuda:0 \
  --log_home ./wandb \
  --eta_values 0.005 0.01 0.02 \
  --fair_score_k_values 0.1 0.15 0.2 \
  --controller_lrs 5e-4 1e-3 \
  --fair_weights 5e4 1e5 \
  --utility_weights 0.1 0.3 \
  --k_tracking_weights 0.0 \
  --controller_epochs 1000 \
  --controller_replay_num_samples 1 \
  --controller_replay_refresh 100 \
  --num_generation 64 \
  --sample_batch_size 32768 \
  --fair_label_attr y \
  --fair_score_guidance_normalize True \
  --run_generated_eval \
  --generated_eval_max_graphs 64 \
  --generated_eval_lp_epochs 1000 \
  --generated_eval_lp_patience 5 \
  --generated_eval_lp_batch_size 16384 \
  --pareto_x_metric lp/score_sp_abs_gap_mean \
  --pareto_y_metric lp/auc_mean \
  --pareto_label_points front \
  --skip_existing
```

The grid writes outputs under `wandb/cora/Sync/controller/`, including:

- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_manifest.jsonl`
- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_summary.csv`
- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.jpg`
- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.front.csv`

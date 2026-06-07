# FairWire Feature Shift Experiments

This folder runs post-training statistical parity shift experiments for the
node-conditioned FairWire feature model. During sampling, the real node
features, sensitive labels, and task labels are kept fixed, and the eta/k shift
is applied to the generated edge logits.

Dependency details are kept in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 1. Required Inputs

To run the eta/k grid, prepare:

- A trained node-conditioned FairWire checkpoint, usually with `aA=0.0` and
  `aX=0.0`.
- A reference graph pickle at `graphs/<dataset>_feat.pkl`.

The reference graph is required by `evaluate_generated_graphs.py` and is checked
by `fair_grid_eval.py` before the grid starts.

```bash
mkdir -p graphs

python make_reference_graph.py --dataset cora --out_path graphs/cora_feat.pkl
python make_reference_graph.py --dataset citeseer --out_path graphs/citeseer_feat.pkl
python make_reference_graph.py --dataset amazon_photo --out_path graphs/amazon_photo_feat.pkl
```

## 2. Checkpoint Preparation

Training can be done in this folder, but you can also reuse a checkpoint trained
in `FairWire_feature`. Pass it with `--model_path`; no retraining is needed as
long as the checkpoint dataset matches the grid dataset.

```bash
# Reuse a checkpoint trained in FairWire_feature.
export FEATURE_AA0_CKPT=../FairWire_feature/cora_0.0_0.0_cpts/Sync_T3.pth
```

If you want to train locally in this folder instead:

```bash
python train.py -d cora -aA 0.0 -aX 0.0 --gpu 0
python train.py -d citeseer -aA 0.0 -aX 0.0 --gpu 0
python train.py -d amazon_photo -aA 0.0 -aX 0.0 --gpu 0

export FEATURE_AA0_CKPT=cora_0.0_0.0_cpts/Sync_T3.pth
```

Local training writes checkpoints such as:

```text
cora_0.0_0.0_cpts/Sync_T3.pth
```

If you trained with a different diffusion step count in `FairWire_feature`, use
that checkpoint path directly, for example:

```bash
export FEATURE_AA0_CKPT=../FairWire_feature/cora_0.0_0.0_cpts/Sync_T8.pth
```

## 3. Eta/K Grid Search With Normalized Guidance

The example below runs an eta/k grid for the `aA=0, aX=0` Cora checkpoint,
samples generated graphs, evaluates LP AUC and score-SP, and writes the Pareto
curve plus summary CSV files. Normalized guidance is enabled with
`--fair_score_guidance_normalize`.

```bash
python fair_grid_eval.py \
  --repo_dir . \
  --dataset cora \
  --model_path "$FEATURE_AA0_CKPT" \
  --num_samples 64 \
  --eta_values 0.001 0.005 0.01 0.05 0.1 \
  --k_values 0.3 0.5 0.8 \
  --seeds 0 1 2 \
  --include_baseline \
  --baseline_k 1.0 \
  --gen_device cuda:0 \
  --lp_device cuda:0 \
  --fair_score_guidance_normalize \
  --lp_epochs 1000 \
  --out_dir fair_grid_feature_shift_norm_cora_aA0 \
  --skip_existing
```

`--fair_score_guidance_normalize` is an alias for `--sp_guidance_normalize`;
it sets normalized shift guidance to true. The baseline run uses `eta=0.0`, so
no shift is applied.

The grid writes:

- `fair_grid_feature_shift_norm_cora_aA0/summary_long.csv`
- `fair_grid_feature_shift_norm_cora_aA0/summary_long_cora.csv`
- `fair_grid_feature_shift_norm_cora_aA0/aggregated_results.csv`
- `fair_grid_feature_shift_norm_cora_aA0/pareto_curve_cora.jpg`
- `fair_grid_feature_shift_norm_cora_aA0/pareto_curve.jpg`
- `fair_grid_feature_shift_norm_cora_aA0/pareto_front.csv`
- Per-run files under
  `fair_grid_feature_shift_norm_cora_aA0/evaluated_graphs/<eta_k_seed>/`,
  including `summary.csv` and `per_graph.csv`.

The most useful columns are:

- `selected_auc`, chosen from `lp/auc_mean` or `aggregate_lp/auc`.
- `selected_sp`, chosen from `lp/score_sp_abs_gap_mean`,
  `aggregate_lp/score_sp_abs_gap`, or compatible SP-gap aliases.
- `selected_auc_mean` and `selected_sp_mean` in `aggregated_results.csv`, which
  are used for the Pareto curve.

To run another dataset, create that dataset's reference graph, point
`FEATURE_AA0_CKPT` to the matching checkpoint, and update `--dataset` plus
`--out_dir`.

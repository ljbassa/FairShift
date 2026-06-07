# FairWire Shift Experiments

This folder runs post-training statistical parity shift experiments on FairWire
checkpoints. Dependency details are kept in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 1. Required Inputs

To run the eta/k grid, prepare these files first:

- A trained FairWire checkpoint, usually with `aA=0.0` and `aX=0.0`.
- A reference graph pickle at `graphs/<dataset>_feat.pkl`.

The grid script checks for the reference graph under this folder, so create it
from inside `FairWire_shift`.

```bash
mkdir -p graphs

python make_reference_graph.py --dataset cora --out_path graphs/cora_feat.pkl
python make_reference_graph.py --dataset citeseer --out_path graphs/citeseer_feat.pkl
python make_reference_graph.py --dataset amazon_photo --out_path graphs/amazon_photo_feat.pkl
```

## 2. Checkpoint Preparation

You can train the `aA=0, aX=0` checkpoints directly in this folder:

```bash
python train.py -d cora -aA 0.0 -aX 0.0 --gpu 0
python train.py -d citeseer -aA 0.0 -aX 0.0 --gpu 0
python train.py -d amazon_photo -aA 0.0 -aX 0.0 --gpu 0
```

Local training saves checkpoints such as:

```text
cora_0.0_0.0_cpts/Sync_T3.pth
```

This folder's `train.py` reads the diffusion step count from
`configs/<dataset>/train_Sync.yaml`. If you already trained the same model in
`FairWire_fairness_loss`, you do not need to retrain here. Use that checkpoint
directly with `--model_path`.

Examples:

```bash
# Local FairWire_shift checkpoint
export FW_AA0_CKPT=cora_0.0_0.0_cpts/Sync_T3.pth

# Or reuse a checkpoint trained in FairWire_fairness_loss
export FW_AA0_CKPT=../FairWire_fairness_loss/cora_0.0_0.0_cpts/Sync_T8.pth
```

When using a checkpoint trained in `FairWire_fairness_loss`, make sure the
checkpoint dataset matches `--dataset`. For example, use a Cora checkpoint with
`--dataset cora`.

## 3. Eta/K Grid Search With Normalized Guidance

Use `fair_grid_eval.py` to sweep eta and k for the `aA=0` checkpoint. The
example below uses normalized guidance, samples graphs, evaluates link
prediction, and plots the LP AUC versus score-SP Pareto curve.

```bash
python fair_grid_eval.py \
  --repo_dir . \
  --dataset cora \
  --model_path "$FW_AA0_CKPT" \
  --num_samples 64 \
  --eta_values 0.005 0.01 0.015 0.03 0.05 \
  --k_values 0.1 0.3 0.5 1.0 \
  --seeds 0 1 2 \
  --include_baseline \
  --baseline_k 1.0 \
  --gen_device cuda:0 \
  --lp_device cuda:0 \
  --fair_score_guidance_normalize \
  --lp_epochs 1000 \
  --out_dir fair_grid_generated_lp_norm_cora_T8 \
  --skip_existing
```

`--fair_score_guidance_normalize` is an alias for `--sp_guidance_normalize`; it
sets normalized shift guidance to true. The baseline run uses `eta=0.0`, so no
shift is applied.

The script writes:

- Generated PyG graph lists under
  `fair_grid_generated_lp_norm_cora_T8/generated_graphs/`.
- Per-run evaluation files under
  `fair_grid_generated_lp_norm_cora_T8/evaluated_graphs/<eta_k_seed>/`,
  including `summary.csv` and `per_graph.csv`.
- A top-level run table:
  `fair_grid_generated_lp_norm_cora_T8/summary_long.csv`.
- A dataset-specific run table:
  `fair_grid_generated_lp_norm_cora_T8/summary_long_cora.csv`.
- Aggregated eta/k results:
  `fair_grid_generated_lp_norm_cora_T8/aggregated_results.csv`.
- Pareto outputs:
  `fair_grid_generated_lp_norm_cora_T8/pareto_curve_cora.jpg`,
  `fair_grid_generated_lp_norm_cora_T8/pareto_curve.jpg`, and
  `fair_grid_generated_lp_norm_cora_T8/pareto_front.csv`.

The main columns to inspect are:

- `selected_auc`, chosen from `lp/auc_mean` or `aggregate_lp/auc`.
- `selected_sp`, chosen from `lp/score_sp_abs_gap_mean`,
  `aggregate_lp/score_sp_abs_gap`, or compatible SP-gap aliases.
- `selected_auc_mean` and `selected_sp_mean` in `aggregated_results.csv`, which
  are used for the Pareto plot.

To run another dataset, rebuild that dataset's reference graph, set
`FW_AA0_CKPT` to the matching checkpoint, and replace `--dataset` and
`--out_dir` accordingly.

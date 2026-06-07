# FairWire Fairness Loss Experiments

This folder contains the FairWire training workflow and the controller-based
fairness guidance experiments. Install dependencies from the separate
requirements file before running the commands below.

```bash
pip install -r requirements.txt
```

## 1. Prepare Reference Graphs

The generated-graph evaluation scripts compare samples against a reference
graph stored as a pickle file. Build the reference graph for each dataset before
running Pareto or controller evaluations.

```bash
mkdir -p graphs

python make_reference_graph.py --dataset cora --out_path graphs/cora_feat.pkl
python make_reference_graph.py --dataset citeseer --out_path graphs/citeseer_feat.pkl
python make_reference_graph.py --dataset amazon_photo --out_path graphs/amazon_photo_feat.pkl
```

## 2. Train FairWire Checkpoints

`train.py` supports separate FairWire multipliers for node features and
adjacency:

- `-aX` or `--alphaX`: feature fairness multiplier.
- `-aA` or `--alphaA`: adjacency fairness multiplier.
- `--T` or `--diffusion_T`: diffusion steps. This overrides the value in
  `configs/<dataset>/train_Sync.yaml`.

The commands below train the `aA=0, aX=0` checkpoints used later by the
controller examples.

```bash
python train.py -d cora --stage fairwire -aA 0.0 -aX 0.0 --T 8 --gpu 0
python train.py -d citeseer --stage fairwire -aA 0.0 -aX 0.0 --T 8 --gpu 0
python train.py -d amazon_photo --stage fairwire -aA 0.0 -aX 0.0 --T 8 --gpu 0
```

The checkpoint directory is named from the dataset and multipliers. For example,
the first command writes:

```text
cora_0.0_0.0_cpts/Sync_T8.pth
```

To run FairWire-style sweeps, change `-aA`, `-aX`, and `--T` independently. For
example:

```bash
python train.py -d cora --stage fairwire -aA 10.0 -aX 0.0 --T 8 --gpu 0
python train.py -d cora --stage fairwire -aA 10.0 -aX 1.0 --T 16 --gpu 0
```

## 3. Basic aA Grid and Pareto Evaluation

A basic experiment is to train several checkpoints with different `aA` values
while keeping `aX` and `T` fixed.

```bash
for AA in 0.0 0.1 1.0 10.0 50.0 100.0; do
  python train.py -d cora --stage fairwire -aA "$AA" -aX 0.0 --T 8 --gpu 0
done
```

After the checkpoints exist, run the aA Pareto helper. It samples graphs from
each checkpoint, evaluates LP AUC and score-SP, writes CSV summaries, and plots
the Pareto curve.

```bash
python scripts/run_aA_pareto.py \
  --repo_dir . \
  --dataset cora \
  --T 8 \
  --alphaX 0.0 \
  --aA_values 0.0 0.1 1.0 10.0 50.0 100.0 \
  --num_samples 64 \
  --seeds 0 1 2 \
  --sample_gpu 0 \
  --eval_device cuda:0 \
  --out_dir fairwire_aA_pareto_cora_T8 \
  --label_points front \
  --skip_existing \
  -- --max_graphs 64
```

Typical outputs are written under `fairwire_aA_pareto_cora_T8/`, including:

- `summary_long.csv`
- `summary_agg.csv`
- `metrics_auc_sp.csv`
- `cora_pareto_curve.png`
- `cora_pareto_front.csv`

## 4. Train a Controller From the aA=0 Checkpoint

The controller workflow starts from a trained `aA=0, aX=0` FairWire checkpoint.
The example below uses normalized fairness guidance and runs generated-graph
evaluation so that LP AUC and score-SP are written automatically.

```bash
export STAGE1_AA0_CKPT=cora_0.0_0.0_cpts/Sync_T8.pth
export FW_CONTROLLER_RUN=replace_with_fw_controller_run_name

python train_controller.py \
  --controller_pretrained_ckpt "$STAGE1_AA0_CKPT" \
  --name "$FW_CONTROLLER_RUN" \
  --log_home ./wandb \
  --device cuda:0 \
  --seed 0 \
  --controller_epochs 1000 \
  --controller_lr 1e-3 \
  --controller_replay_num_samples 1 \
  --controller_replay_refresh 100 \
  --num_generation 64 \
  --sample_batch_size 32768 \
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

Look for the generated evaluation summary in the controller output directory.
The most important columns are:

- `lp/auc_mean`
- `lp/score_sp_abs_gap_mean`
- `aggregate_lp/auc`
- `aggregate_lp/score_sp_abs_gap`

The generated CSV files for one controller run are written under
`wandb/<dataset>/Sync/controller/<FW_CONTROLLER_RUN_NAME>/generated_samples/`.

## 5. Controller Grid Search

Use `scripts/run_controller_grid.py` to sweep controller hyperparameters. With
`--run_generated_eval`, the script trains every controller, evaluates generated
graphs, builds a summary CSV, and plots the Pareto curve for LP AUC versus
score-SP.

```bash
export FW_CONTROLLER_GRID_PREFIX=replace_with_fw_controller_grid_prefix

python scripts/run_controller_grid.py \
  --repo_dir . \
  --stage1_ckpt "$STAGE1_AA0_CKPT" \
  --dataset cora \
  --name_prefix "$FW_CONTROLLER_GRID_PREFIX" \
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

By default, the grid script writes outputs under
`wandb/<dataset>/Sync/controller/`, including:

- `<FW_CONTROLLER_GRID_PREFIX>_manifest.jsonl`
- `<FW_CONTROLLER_GRID_PREFIX>_summary.csv`
- `<FW_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.jpg`
- `<FW_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_score_sp.front.csv`

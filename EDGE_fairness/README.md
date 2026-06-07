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
fair_grid_generated_lp_norm_cora/summary_long.csv
fair_grid_generated_lp_norm_cora/summary_long_cora.csv
fair_grid_generated_lp_norm_cora/pareto_curve_cora.jpg
```

The CSV files used for the LP AUC vs score-SP table are produced by the command above. The per-run CSV files are under
`fair_grid_generated_lp_norm_cora/evaluated_graphs/`, and the aggregate CSVs are the top-level `summary_long*.csv`
files.

To keep a short generic filename as well:

```bash
cp fair_grid_generated_lp_norm_cora/summary_long_cora.csv \
  fair_grid_generated_lp_norm_cora/summary.csv
```

You can redraw only the Pareto curve from an existing summary CSV without regenerating graphs:

```bash
python fair_grid_eval_generated_graphs.py \
  --repo_dir . \
  --dataset cora \
  --summary_csv fair_grid_generated_lp_norm_cora/summary_long_cora.csv \
  --auc_candidates lp/auc_mean aggregate_lp/auc \
  --sp_candidates lp/score_sp_abs_gap_mean aggregate_lp/score_sp_abs_gap \
  --out_dir fair_grid_generated_lp_norm_cora
```

## Results

Training outputs are stored under:

```text
wandb/<dataset>/multinomial_diffusion/multistep/<run_name>/
```

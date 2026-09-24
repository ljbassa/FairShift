# FairWire Feature Fairness Loss Experiments

This folder trains and evaluates the Stage-2 fairness controller for the
node-conditioned FairWire feature model. Dependency details are kept in
`requirements.txt`.

```bash
pip install -r requirements.txt
```

Run the commands below from the `FairWire_feature_fairness_loss` directory.

The feature variant keeps the input node features, sensitive attributes, and
labels fixed while generating edges. Its edge denoiser also receives the original
node features. Stage 2 freezes that denoiser and learns the fairness controller
from recorded reverse-diffusion trajectories.

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
wandb/cora/Sync/controller/sp/<FW_FEATURE_CONTROLLER_RUN_NAME>/
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

The grid writes outputs under `wandb/cora/Sync/controller/sp/`, including:

- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_manifest.jsonl`
- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_summary.csv`
- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_sp.jpg`
- `<FW_FEATURE_CONTROLLER_GRID_PREFIX>_pareto_lp_auc_vs_sp.front.csv`

Without `--controller_root`, the grid infers the dataset from checkpoint folders
named `<dataset>_<aA>_<aX>_cpts` and uses
`<log_home>/<dataset>/Sync/controller/<metric>/`. For other checkpoint layouts,
pass `--dataset citeseer` (or the matching dataset); the fallback is `cora`.
`--pareto_title` overrides the dataset-based plot title. The feature grid also
supports `--pareto_front_csv` and `--pareto_extra_summary_csv` for custom front
exports and merging earlier grid summaries.

## SP / EO controller selection

Add `--fair_score_metric eo` to `train_controller.py` or
`scripts/run_controller_grid.py` to train an EO controller. The default is `sp`;
the existing SP objective and feature handling are unchanged. EO compares the
running scores of same-group and different-group pairs, weighted by an independent
unguided denoiser EMA. These positive-condition weights are detached from the
controller. `--fair_score_eo_min_mass 1e-6` masks groups with insufficient positive
mass.

Here EO means equal opportunity on positive edges. For each pair group `g`,
the training proxy is `sum(w_e * q_e) / sum(w_e)`, with `q_e` the guided running
probability and `w_e` the detached unguided positive-edge weight. The proxy gap
compares same-group and different-group pairs, using `--fair_label_attr y` by
default (`s` selects sensitive attributes). If either group is empty or has
positive mass at or below the threshold, its graph contributes zero fairness
loss and guidance. This proxy differs from the reported `lp/eo_abs_gap_mean`:
LP evaluation compares predicted link scores on actual held-out positive edges.

Training outputs, checkpoints, generated graphs, grid manifests, summaries, and
plots use `controller/sp/<name>/` or `controller/eo/<name>/`. Explicit `--out_dir`
without `--log_home` uses `<out_dir>/<metric>/`. Metric-specific `latest_run.json` files are written
inside the corresponding metric directory. For an EO grid, omit the example's
`--pareto_x_metric` override so it automatically uses `lp/eo_abs_gap_mean`.
Standalone summary and plot scripts also accept `--fair_score_metric eo`.

`sample.py` restores the metric and positive-mass threshold from the controller
checkpoint. Legacy checkpoints default to SP. An explicit conflicting metric is
rejected. For static EO guidance, pass `--fair_score_sp --fair_score_metric eo`.
Guided sample exports use an `sp/` or `eo/` subdirectory of the requested save directory
(including `--save_pkl_dir` and the parent of `--save_pt_path`).

For example, using the Cora feature checkpoint prepared above:

```bash
export FW_FEATURE_EO_RUN=cora_feature_eo

python train_controller.py \
  --controller_pretrained_ckpt "$FEATURE_STAGE1_AA0_CKPT" \
  --name "$FW_FEATURE_EO_RUN" --log_home ./wandb --device cuda:0 \
  --fair_score_metric eo --fair_score_eo_min_mass 1e-6 --fair_label_attr y \
  --fair_score_eta 0.005 --fair_score_k 0.15 \
  --fair_score_learn_k False --fair_score_learn_eta True \
  --fair_score_fair_loss_weight 1e5 --fair_score_k_tracking_loss_weight 0.0 \
  --fair_score_utility_loss_weight 0.1 --run_generated_eval

python scripts/run_controller_grid.py \
  --stage1_ckpt "$FEATURE_STAGE1_AA0_CKPT" --dataset cora \
  --name_prefix "${FW_FEATURE_EO_RUN}_grid" --device cuda:0 \
  --fair_score_metric eo --eta_values 0.005 0.01 \
  --fair_score_k_values 0.1 0.15 --run_generated_eval

python sample.py \
  --model_path "wandb/cora/Sync/controller/eo/${FW_FEATURE_EO_RUN}/check/controller_best.pt" \
  --device cuda:0 --num_samples 64 --save_samples --save_dir generated_samples
```

The sampling command restores EO from the checkpoint and writes to
`generated_samples/eo/`. For fixed guidance without a trained controller, use:

```bash
python sample.py --model_path "$FEATURE_STAGE1_AA0_CKPT" \
  --fair_score_sp --fair_score_metric eo --fair_score_eta 0.005 --fair_score_k 0.15 \
  --device cuda:0 --save_samples --save_dir generated_samples
```

The EO implementation is in `Model/fairness_surrogate.py` and
`Model/fair_diffusion.py`; `fairness_options.py` handles metric selection and
output paths. The controller entry points and grid/summary/plot scripts carry
these options through training, sampling, and evaluation.

CPU checks for the fairness calculations, checkpoint compatibility, and metric
output separation can be run with `python -m pytest -q tests`.

## Code-only synchronization

FairShift includes source, configuration, dependency files, tests, and this
README. Checkpoints, datasets, reference graph pickles, generated graphs,
`wandb/`, result CSVs, plots, logs, and Python caches are excluded. Prepare the
inputs locally and generate evaluation outputs with the commands above.

## EO (Equal Opportunity) 평가와 사용법

현재 feature controller는 고정된 노드 특징과 라벨을 사용하여 SP 또는 EO 보정을 학습합니다. `--fair_score_metric eo`로 EO 목적을 선택하고, `--fair_score_fair_loss_weight`로 선택한 fairness 손실의 가중치를 조절합니다. 학습 proxy와 아래의 생성 그래프 EO 평가 지표는 구분해야 합니다. 구체적인 학습·grid·샘플링 명령은 위의 [SP / EO controller selection](#sp--eo-controller-selection)을 참고합니다.

`evaluate_generated_graphs.py`의 `samplepy_group_fairness()`는 생성 그래프의
held-out 양성 간선(`Y_uv=1`)에서 다음 값을 계산합니다.

```text
EO = | mean(p_uv | group_u = group_v, Y_uv = 1)
     - mean(p_uv | group_u != group_v, Y_uv = 1) |
```

여기서 `p_uv`는 GAE의 링크 예측 확률이고, `Y_uv`는 노드 클래스가 아닌
간선의 존재 여부입니다. 그룹은 저장된 PyG 그래프의 `sens`를 우선 사용하며,
없으면 `--label_attr`(기본 `y`)를 사용합니다. 이 값은 확률 점수 기반의
Equal Opportunity 차이입니다. 음성 간선 조건의 차이는 포함하지 않으며,
`--threshold`로 이진화한 TPR 차이도 아닙니다. 양성 간선의 동일/상이 그룹
중 하나가 비어 있으면 EO는 `NaN`이므로 0으로 해석하면 안 됩니다.

다음 명령은 이 폴더에서 실행합니다. `CPU_CORE`는 사용 가능한 코어 번호로,
`FW_CKPT`는 실제 학습된 체크포인트로 설정합니다. 기존 데이터와 체크포인트,
PyTorch/DGL/PyG 실행 환경이 필요합니다. 생성 파일은 Git에 포함하지 않습니다.

```bash
export CPU_CORE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export FW_CKPT=../FairWire_feature/cora_0.0_0.0_cpts/Sync_T3.pth

taskset -c "$CPU_CORE" python make_reference_graph.py \
  --dataset cora --out_path graphs/cora_feat.pkl

taskset -c "$CPU_CORE" python sample.py \
  --model_path "$FW_CKPT" --num_samples 1 --seed 0 --gpu 0 \
  --save_pt_path saved_generated/cora_eo_check.pyg.pt --skip_internal_eval

taskset -c "$CPU_CORE" python evaluate_generated_graphs.py \
  --graph_path saved_generated/cora_eo_check.pyg.pt --dataset cora --seed 0 \
  --out_per_graph_csv results/cora_eo.per_graph.csv \
  --out_summary_csv results/cora_eo.summary.csv
```

평가기에서 EO를 별도로 켤 필요는 없습니다. 양성 간선을 80/10/10으로
train/validation/test에 분리하고 validation AUC로 GAE를 선택한 뒤,
test 쌍의 AUC/SP/EO를 함께 기록합니다. 실행에는 GAE 학습이 포함됩니다.

- 그래프별 CSV: `lp/auc`, `lp/eo_gap`, `lp/eo_abs_gap`.
- 요약 CSV: `lp/eo_gap_mean`, `lp/eo_abs_gap_mean`, `lp/eo_abs_gap_std`.
- 전체 test 쌍을 합친 EO: `aggregate_lp/eo_abs_gap`. 그래프별 EO의 평균과
  합친 쌍의 EO는 집계 방식이 다릅니다.

`lp/eo_gap`과 `lp/eo_abs_gap`은 모두 절댓값입니다. AUC와 EO를 함께
비교하고, 동일한 데이터·seed·평가 프로토콜을 사용해야 합니다.

위 샘플링 예시는 Stage-1 체크포인트를 사용합니다. 학습된 controller를
평가하려면 `FW_CKPT`를 해당 run의 `full_model_best.pt`로 바꾸거나,
Stage-1 경로와 `--controller_path`를 함께 지정합니다. 기존
`train_controller.py --run_generated_eval` 경로도 EO 열을 함께 출력합니다.

Controller를 사용하면 `--save_pt_path`의 부모 아래에 `sp/` 또는 `eo/`가
추가됩니다. 예를 들어 EO controller로 위 파일을 저장했다면 평가기의
`--graph_path`는 `saved_generated/eo/cora_eo_check.pyg.pt`로 바꿉니다.

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

## EO (Equal Opportunity) 평가와 사용법

현재 feature controller는 고정된 노드 특징과 라벨을 사용하여 SP 보정을 학습합니다. `--fair_score_fair_loss_weight`는 SP controller 손실의 가중치이며, EO는 생성 그래프의 **평가 지표**입니다. 이 폴더에는 EO controller 목적함수를 선택하는 옵션이 없습니다.

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

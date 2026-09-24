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

## EO (Equal Opportunity) 평가와 사용법

현재 Stage-2 controller는 SP 보정을 학습합니다. `--fair_score_fair_loss_weight`는 해당 SP 손실의 가중치이며 EO 손실을 선택하지 않습니다. EO는 학습된 controller 또는 Stage-1 모델이 생성한 그래프의 **평가 지표**입니다. 이 폴더에는 `--fair_score_metric eo` 같은 EO 학습/샘플링 옵션이 없습니다.

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
export FW_CKPT=cora_0.0_0.0_cpts/Sync_T8.pth

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

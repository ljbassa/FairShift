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

## EO (Equal Opportunity) 평가와 사용법

이 폴더는 실제 노드 특징과 라벨을 고정하고 간선에 SP 보정을 적용합니다. `--sp_shift`, `--sp_eta`, `--sp_k`, `--sp_guidance_normalize`는 SP 설정이며, EO는 생성 그래프의 **평가 지표**입니다. 현재 sampler에는 EO 목적 선택 옵션이 없습니다.

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
  --sp_shift --sp_eta 0.005 --sp_k 0.15 --sp_guidance_normalize \
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

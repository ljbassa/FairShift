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

## EO (Equal Opportunity) 평가와 사용법

이 폴더의 `--sp_shift`, `--sp_eta`, `--sp_k`, `--sp_guidance_normalize`는 SP 점수 차이에 대한 샘플링 보정을 설정합니다. EO는 보정 후 생성 그래프의 **평가 지표**로 확인합니다. 현재 FairWire sampler에는 EO를 직접 보정 대상으로 선택하는 옵션이 없습니다.

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
export FW_CKPT=cora_0.0_0.0_cpts/Sync_T3.pth

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

## 실행 시간 측정 도구

`runtime_comparison/`에는 측정용 Python 코드만 포함되어 있습니다.
`benchmark_pipeline.py`는 준비된 체크포인트로 생성부터 GAE 평가 완료까지
측정하며, 체크포인트 학습 및 기준 그래프 준비 시간은 제외합니다.
위에서 설정한 CPU/GPU 환경과 `graphs/cora_feat.pkl`을 사용합니다.
`--output-dir`에는 아직 존재하지 않는 디렉터리를 지정합니다.

```bash
taskset -c "$CPU_CORE" python runtime_comparison/benchmark_pipeline.py \
  --checkpoint "$FW_CKPT" --dataset cora --eta 0.005 \
  --num-samples 1 --seed 0 --output-dir results/runtime_cora
```

이 도구는 SP shift의 기본 `k=1.0`, 비정규화 설정을 사용합니다. 위 EO
샘플링 예시의 `k=0.15`, 정규화 설정과 측정 조건이 다릅니다.
`benchmark_train.py`는 한 학습 epoch와 validation pass의 시간을 측정하고,
`prepare_timing_checkpoint.py` 및 `run_approximate.py`는 수렴하지 않은
짧은 warmup 체크포인트로 실행 시간만 가늠하는 도구입니다. 이 경로의
그래프 품질/AUC/SP는 학습이 완료된 모델의 성능으로 해석하면 안 됩니다.
`inspect_approximate_graphs.py`는 해당 산출물을 CPU에서 검사합니다.

학습 측정과 warmup은 같은 저장소의 `../FairWire_fairness_loss`에 있는
Stage-1 모델과 설정을 사용합니다. `run_approximate.py`는 자신을 실행한
Python 환경을 하위 프로세스에도 사용합니다. 측정 결과, 체크포인트,
로그는 이 저장소 동기화에 포함하지 않습니다.

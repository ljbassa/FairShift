# FairWire Feature Experiments

This folder contains the node-conditioned FairWire training code. Compared with
the base FairWire training flow, this variant conditions the feature and edge
denoising networks on node label information when it is available in the
dataset. The model configuration exposes this through the `hidden_Y` entries in
`configs/<dataset>/train_Sync.yaml`.

Dependency details are kept in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Training

Run `train.py` from this folder. The main arguments are:

- `-d` or `--dataset`: dataset name.
- `-aA` or `--alphaA`: adjacency fairness multiplier.
- `-aX` or `--alphaX`: feature fairness multiplier.
- `--T`: optional override for `diffusion.T` in
  `configs/<dataset>/train_Sync.yaml`.
- `--gpu`: CUDA device id.

Basic training commands:

```bash
python train.py -d cora --alphaA 0.0 --alphaX 0.0 --T 3 --gpu 0
python train.py -d citeseer --alphaA 0.1 --alphaX 0.0 --T 3 --gpu 0
python train.py -d amazon_photo --alphaA 0.1 --alphaX 0.0 --T 3 --gpu 0
```

The checkpoint directory is named from the dataset and fairness multipliers.
For example:

```bash
python train.py -d cora -aA 10.0 -aX 0.0 --T 3 --gpu 0
```

writes:

```text
cora_10.0_0.0_cpts/Sync_T3.pth
```

If `--T` is omitted, training uses the value in the dataset YAML file.

## Training Multiple aA Values

Use `run_train_batch.py` when you want to train several `aA` values
sequentially and save logs plus a summary table.

```bash
python run_train_batch.py \
  --repo_dir . \
  --dataset cora \
  --alphaX 0.0 \
  --alphaA_values 0.0 0.1 1.0 10.0 50.0 100.0 \
  --T 3 \
  --gpu 0 \
  --skip_existing
```

The batch runner writes logs and summaries under `batch_runs/train/`, including:

- `summary.csv`
- `summary.json`
- one log file per `aA` value

Each successful run still writes the normal FairWire checkpoint directory, such
as `cora_1.0_0.0_cpts/Sync_T3.pth`.

## Generating LP AUC / Score-SP CSVs

After the checkpoints exist, run `fair_grid_eval.py` to sample from each
checkpoint, evaluate the generated graphs, and aggregate the per-seed results.
The command below is the Cora `aA` sweep used to produce the CSV summaries for
the FW-fc baseline.

```bash
python fair_grid_eval.py \
  --repo_dir . \
  --python_exec "$(command -v python)" \
  --model_globs "cora_*_0.0_cpts/Sync_T3.pth" \
  --dataset cora \
  --num_samples 64 \
  --sample_gpu 0 \
  --seeds 0 1 2 \
  --out_dir fairwire_feature_aA_pareto_cora_T3 \
  --plot_x_metric lp/score_sp_abs_gap_mean \
  --plot_y_metric lp/auc_mean \
  --label_points front \
  -- \
  --device cuda:0 \
  --label_attr y \
  --sensitive_attr y \
  --lp_epochs 1000
```

The main CSV outputs are:

- `fairwire_feature_aA_pareto_cora_T3/summary_long.csv`
- `fairwire_feature_aA_pareto_cora_T3/summary_agg.csv`
- `fairwire_feature_aA_pareto_cora_T3/pareto_front.csv`
- Per-run evaluation files under
  `fairwire_feature_aA_pareto_cora_T3/eval_outputs/`, including
  `*.summary.csv` and `*.per_graph.csv`.

Use the same command for Citeseer or Amazon Photo by changing `--dataset`,
`--model_globs`, and `--out_dir` to the matching dataset and checkpoint
pattern.

## Supported Datasets

The training CLI accepts:

- `cora`
- `citeseer`
- `amazon_photo`
- `amazon_computer`
- `german`
- `pokec_n`

Dataset-specific architecture, optimizer, diffusion, and early-stopping settings
are stored in `configs/<dataset>/train_Sync.yaml`.

## EO (Equal Opportunity) 평가와 사용법

이 폴더는 실제 노드 특징과 그룹/태스크 라벨을 유지하는 feature 모델입니다. EO는 생성 그래프의 **평가 지표**로 제공하며, `train.py`의 `--alphaA`/`--alphaX`가 EO 목적함수를 선택하는 옵션은 아닙니다.

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

이 폴더의 현재 standalone GAE 평가 경로는 사용 가능한 `cuda:0`을
선택하므로 GPU는 위의 `CUDA_VISIBLE_DEVICES`로 제한합니다. CPU에서
평가하려면 이 환경변수를 빈 문자열로 설정합니다.

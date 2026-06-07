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

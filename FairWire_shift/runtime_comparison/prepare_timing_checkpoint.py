#!/usr/bin/env python3
"""Create an UNCONVERGED FairWire checkpoint for approximate runtime probes only.

This short warmup is excluded from FairShift timing. Resulting graphs and AUC/SP
are not valid model-quality estimates. There is no validation or convergence run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
BACKBONE_ROOT = Path(__file__).resolve().parents[2] / "FairWire_fairness_loss"
sys.path.insert(0, str(BACKBONE_ROOT))

import torch
import yaml

from data import load_dataset, preprocess
from Model import ModelSync
from setup_utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=["cora", "citeseer", "amazon_photo"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=32)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; select a physical GPU with CUDA_VISIBLE_DEVICES")

    torch.set_num_threads(4)
    set_seed(0)
    device = torch.device("cuda:0")
    config_path = BACKBONE_ROOT / "configs" / args.dataset / "train_Sync.yaml"
    config = yaml.safe_load(config_path.read_text())
    train = config["train"]
    graph = load_dataset(args.dataset)
    X, s, y, E, xm, sm, ym, em, xcs, xcy, ycs, pvals = preprocess(graph)
    X, s, E, xm, sm, em = [v.to(device) for v in (X, s, E, xm, sm, em)]
    if y is not None:
        y, ym, ycs = [v.to(device) for v in (y, ym, ycs)]
    model = ModelSync(
        X_marginal=xm, s_marginal=sm, y_marginal=ym, E_marginal=em,
        num_nodes=graph.num_nodes(), p_values=pvals, y_cond_s_marginal=ycs,
        gnn_X_config=config["gnn_X"], gnn_E_config=config["gnn_E"],
        **config["diffusion"],
    ).to(device)
    optimizer_x = torch.optim.AdamW(model.graph_encoder.pred_X.parameters(),
                                    **config["optimizer_X"])
    optimizer_e = torch.optim.AdamW(model.graph_encoder.pred_E.parameters(),
                                    **config["optimizer_E"])

    # Same uniform, shuffled unordered-pair batches as train.py, without the
    # DataLoader's full Python list of indices or worker processes for this probe.
    pairs = torch.triu_indices(graph.num_nodes(), graph.num_nodes(), offset=1).T
    batch_size = train["batch_size"]
    order = None
    position = len(pairs)
    model.train()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for update in range(args.updates):
        if position >= len(pairs):
            order = torch.randperm(len(pairs))
            position = 0
        batch = pairs[order[position:position + batch_size]].to(device)
        position += batch_size
        dst, src = batch.T
        loss_x, fair_x, loss_e, fair_e = model.log_p_t(
            X, E, src, dst, E[dst, src], s, y,
        )
        loss = loss_x + loss_e  # alphaA=alphaX=0, exactly train.py's base loss.
        optimizer_x.zero_grad()
        optimizer_e.zero_grad()
        loss.backward()
        for predictor in (model.graph_encoder.pred_X, model.graph_encoder.pred_E):
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), train["max_grad_norm"])
        optimizer_x.step()
        optimizer_e.step()
        print(json.dumps({"dataset": args.dataset, "update": update + 1,
                          "updates": args.updates, "loss_X": loss_x.item(),
                          "loss_E": loss_e.item()}), flush=True)
    torch.cuda.synchronize()
    metadata = {
        "timing_only": True,
        "converged": False,
        "quality_metrics_valid": False,
        "description": "Short unconverged warmup; use only for approximate execution timing",
        "optimizer_updates": args.updates,
        "warmup_seconds_excluded_from_fairshift": time.perf_counter() - started,
        "alpha_A": 0.0, "alpha_X": 0.0, "seed": 0, "cpu_threads": 4,
        "nodes": graph.num_nodes(), "batch_size": batch_size,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "source_config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    checkpoint = {
        "dataset": args.dataset, "train_yaml_data": config, "best_val_nll": None,
        "pred_X_state_dict": {key: value.detach().cpu()
                              for key, value in model.graph_encoder.pred_X.state_dict().items()},
        "pred_E_state_dict": {key: value.detach().cpu()
                              for key, value in model.graph_encoder.pred_E.state_dict().items()},
        "timing_probe": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(checkpoint, handle)
    print(json.dumps({"checkpoint": str(output), **metadata}), flush=True)


if __name__ == "__main__":
    main()

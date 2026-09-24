#!/usr/bin/env python3
"""CPU-only diagnostics for approximate FairShift runtime probes; safe to rerun."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

# These must precede torch imports: this inspector never needs CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch


def finite_float(value):
    result = float(value)
    return result if math.isfinite(result) else None


def inspect_graph(graph):
    n = int(graph.num_nodes)
    edge_index = graph.edge_index.cpu().long()
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    valid = lo != hi
    num_edges = int(torch.unique(lo[valid] * n + hi[valid]).numel())
    x = graph.x.cpu().float()
    feature_var = x.var(dim=0, unbiased=False)
    sens_values, sens_counts = torch.unique(graph.sens.cpu(), return_counts=True)
    num_pairs = n * (n - 1) // 2
    same_pairs = sum(int(count) * (int(count) - 1) // 2 for count in sens_counts)
    warnings = []
    if num_edges < 10:
        warnings.append("Too few positive edges for nonempty 80/10/10 GAE split.")
    if num_edges * 1.2 > num_pairs:
        warnings.append("Dense graph may have insufficient true nonedges for GAE negatives.")
    if not bool(torch.isfinite(x).all()):
        warnings.append("Nonfinite generated features.")
    if not bool((feature_var > 0).any()):
        warnings.append("All generated feature columns are constant across nodes.")
    if same_pairs == 0 or same_pairs == num_pairs:
        warnings.append("Missing same-sensitive or different-sensitive candidate pairs.")
    return {
        "nodes": n,
        "undirected_edges": num_edges,
        "candidate_pairs": num_pairs,
        "edge_density": num_edges / num_pairs if num_pairs else None,
        "feature_dimensions": int(x.shape[1]),
        "feature_nonzero_fraction": finite_float((x != 0).float().mean()),
        "feature_variance_all_entries": finite_float(x.var(unbiased=False)),
        "mean_feature_variance_across_nodes": finite_float(feature_var.mean()),
        "constant_feature_columns": int((feature_var == 0).sum()),
        "zero_feature_rows": int((x == 0).all(dim=1).sum()),
        "sensitive_counts": {str(int(v)): int(c) for v, c in zip(sens_values, sens_counts)},
        "same_sensitive_candidate_pairs": same_pairs,
        "different_sensitive_candidate_pairs": num_pairs - same_pairs,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "approximate")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    results = {"device": "cpu", "threads": 2, "datasets": {}}
    for dataset in ["cora", "citeseer", "amazon_photo"]:
        directory = args.directory / dataset
        graph_path = directory / "generated.pyg.pt"
        item = {"graph_path": str(graph_path), "status": "graph_pending"}
        if graph_path.is_file():
            graphs = torch.load(graph_path, map_location="cpu")
            if not isinstance(graphs, (list, tuple)):
                graphs = [graphs]
            item.update(status="inspected", graphs=[inspect_graph(graph) for graph in graphs])
        eval_path = directory / "evaluation.per_graph.csv"
        if eval_path.is_file():
            with eval_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            keys = ["lp/auc", "lp/sp_abs_gap", "lp/best_val_auc", "lp/best_epoch",
                    "lp/best_hidden_dim", "lp/best_dropout", "lp/best_lr",
                    "lp/train_num_pos", "lp/val_num_pos", "lp/test_num_pos", "lp/test_num_neg"]
            item["evaluation"] = [{key: finite_float(row[key]) for key in keys if row.get(key)} for row in rows]
            for index, row in enumerate(rows):
                warnings = item.get("graphs", [{}])[index].setdefault("warnings", [])
                if row.get("lp/error"):
                    warnings.append("Evaluation error: " + row["lp/error"])
                for key in ["lp/auc", "lp/sp_abs_gap"]:
                    if not row.get(key) or finite_float(row[key]) is None:
                        warnings.append("Missing or nonfinite " + key)
                if row.get("lp/auc") and finite_float(row["lp/auc"]) == 0.5:
                    warnings.append("AUC equals chance level; inspect features and model maturity.")
        results["datasets"][dataset] = item
    results["all_graphs_present"] = all(item["status"] == "inspected" for item in results["datasets"].values())
    output = args.directory / "graph_diagnostics.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()

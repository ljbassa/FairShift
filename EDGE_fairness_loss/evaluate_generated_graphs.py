#!/usr/bin/env python3
"""
Evaluate saved generated graphs with:
1) overlap/auc on fixed reference pairs
2) link prediction that mirrors FairWire sample.py's GAE evaluator structure.

The LP path follows eval_utils.prepare_for_GAE + Model.discriminator.GAETrainer:
- use a symmetrically normalized adjacency with self-loops
- split positive upper-triangle edges 80/10/10 into train/val/test
- sample validation/test negatives from true non-edges
- train a 1-layer GAE with the same hyperparameter grid and validation-AUC early stopping
- report AUC/SP/EO on generated-graph held-out test pairs

This file is self-contained so EDGE_fairness, EDGE_fairness_loss, FairWire, and
FairWire_feature use the same measurement structure without importing sample.py.
"""

import argparse
import csv
import itertools
import math
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def safe_group_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if values.size == 0 or mask.size == 0 or mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


def safe_abs(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(abs(value))


def safe_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_compat_metric_aliases(row: Dict[str, Any]) -> None:
    alias_pairs = [
        ("lp/auc", "value/linkpred_auc"),
        ("lp/sp_gap", "fair_gap"),
        ("lp/sp_abs_gap", "fair_abs_gap"),
        ("lp/sp_gap", "value/fair_gap"),
        ("lp/sp_abs_gap", "value/fair_abs_gap"),
    ]
    for src, dst in alias_pairs:
        if src in row and dst not in row:
            row[dst] = row[src]


def unique_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    row, col = edge_index
    mask = row != col
    row = row[mask]
    col = col[mask]
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    edges = torch.stack([lo, hi], dim=0)
    if edges.numel() == 0:
        return edges.reshape(2, 0)
    edges = torch.unique(edges.t(), dim=0).t().contiguous()
    order = torch.argsort(edges[0] * int(edge_index.max().item() + 1 if edge_index.numel() else 1) + edges[1])
    return edges[:, order]


def ensure_features(data) -> Any:
    if getattr(data, "x", None) is None or data.x is None:
        edge_index = data.edge_index.detach().cpu().long()
        deg = torch.bincount(edge_index[0], minlength=int(data.num_nodes)).float().view(-1, 1)
        data.x = deg
    else:
        data.x = data.x.float()
        if data.x.dim() == 1:
            data.x = data.x.view(-1, 1)
    return data


def get_local_attr_vector(data, attr: str) -> torch.Tensor:
    if not hasattr(data, attr):
        raise ValueError(f"data has no attribute {attr!r}")
    values = getattr(data, attr)
    if values is None:
        raise ValueError(f"data.{attr} is None")
    if values.dim() > 1:
        values = values.squeeze()
    return values.detach().cpu()


def get_lp_group_vector(data, preferred_attr: str = "y") -> torch.Tensor:
    # sample.py's link-prediction branch uses s as the pair-group label.
    # Saved FairWire graphs carry this as `sens`; EDGE graphs usually carry `y`.
    if hasattr(data, "sens") and getattr(data, "sens") is not None:
        return get_local_attr_vector(data, "sens").long()
    if hasattr(data, preferred_attr) and getattr(data, preferred_attr) is not None:
        return get_local_attr_vector(data, preferred_attr).long()
    if hasattr(data, "y") and getattr(data, "y") is not None:
        return get_local_attr_vector(data, "y").long()
    raise ValueError("No node group attribute found; expected sens or y.")


def global_id_mapping(data) -> Tuple[List[int], Dict[int, int]]:
    if hasattr(data, "orig_id") and data.orig_id is not None:
        gids = [int(v) for v in data.orig_id.detach().cpu().numpy().tolist()]
    else:
        gids = list(range(int(data.num_nodes)))
    return gids, {g: i for i, g in enumerate(gids)}


# -----------------------------
# Graph loading / reference pairs
# -----------------------------

def load_saved_graphs(graph_path: str) -> List[Any]:
    obj = torch.load(graph_path, map_location="cpu", weights_only=False)
    graphs = obj if isinstance(obj, list) else [obj]
    if not graphs:
        raise ValueError(f"No graphs found in {graph_path}")
    return graphs


def _candidate_reference_paths(graph_path: Path, dataset: str) -> List[Path]:
    candidates: List[Path] = []
    names = [f"{dataset}_feat.pkl", f"{dataset}.pkl"]

    def add(root: Path) -> None:
        for name in names:
            candidates.append(root / "graphs" / name)
            candidates.append(root / name)

    gp = graph_path.resolve()
    for root in [gp.parent, *gp.parent.parents]:
        add(root)
    cwd = Path.cwd().resolve()
    for root in [cwd, *cwd.parents]:
        add(root)

    out: List[Path] = []
    seen = set()
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            out.append(c)
    return out


def find_reference_graph_path(graph_path: str, dataset: str) -> Path:
    for c in _candidate_reference_paths(Path(graph_path), dataset):
        if c.exists():
            return c
    searched = "\n".join(str(c) for c in _candidate_reference_paths(Path(graph_path), dataset))
    raise FileNotFoundError(
        f"Could not find reference graph for dataset={dataset!r}.\n"
        f"Searched these candidate paths:\n{searched}"
    )


def load_reference_graph_from_dataset(graph_path: str, dataset: str) -> nx.Graph:
    ref_path = find_reference_graph_path(graph_path, dataset)
    with ref_path.open("rb") as f:
        g_ref = pickle.load(f)
    if not isinstance(g_ref, nx.Graph):
        raise TypeError(f"Reference object at {ref_path} is not a networkx.Graph")
    return g_ref


def build_reference_node_group_map(g_ref: nx.Graph, attr: str) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    fallback = "sens" if attr == "y" else "y"
    for n, attrs in g_ref.nodes(data=True):
        key = attr if attr in attrs else fallback if fallback in attrs else None
        if key is None:
            raise KeyError(f"Reference graph node {n} lacks attr {attr!r} or fallback {fallback!r}")
        val = attrs[key]
        if hasattr(val, "item"):
            val = val.item()
        out[int(n)] = val
    return out


def build_fixed_eval_pairs(
    g_ref: nx.Graph,
    max_pos_edges: int = 20000,
    neg_ratio: float = 1.0,
    seed: int = 0,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rng = random.Random(seed)
    nodes = [int(n) for n in g_ref.nodes()]

    pos_edges_all: List[Tuple[int, int]] = []
    edge_set = set()
    for u, v in g_ref.edges():
        if u == v:
            continue
        e = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        if e not in edge_set:
            edge_set.add(e)
            pos_edges_all.append(e)

    pos_edges = rng.sample(pos_edges_all, max_pos_edges) if max_pos_edges and len(pos_edges_all) > max_pos_edges else pos_edges_all
    num_neg = int(len(pos_edges) * neg_ratio)
    neg_edges: List[Tuple[int, int]] = []
    neg_set = set()
    while len(neg_edges) < num_neg:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in edge_set or e in neg_set:
            continue
        neg_set.add(e)
        neg_edges.append(e)

    pairs = pos_edges + neg_edges
    labels = np.concatenate([
        np.ones(len(pos_edges), dtype=np.int64),
        np.zeros(len(neg_edges), dtype=np.int64),
    ])
    return pairs, labels


def pair_same_group_mask(pairs: Sequence[Tuple[int, int]], node_groups: Dict[int, Any]) -> np.ndarray:
    return np.asarray([node_groups[int(u)] == node_groups[int(v)] for u, v in pairs], dtype=bool)


def edge_overlap_on_fixed_pairs(
    data,
    reference_pairs: Sequence[Tuple[int, int]],
    reference_labels: np.ndarray,
    reference_node_groups: Dict[int, Any],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    _gids, g2l = global_id_mapping(data)
    edge_index = unique_undirected_edge_index(data.edge_index)
    edge_set = {tuple(map(int, e)) for e in edge_index.t().tolist()}

    keep_idx: List[int] = []
    scores: List[float] = []
    kept_pairs: List[Tuple[int, int]] = []
    for i, (u, v) in enumerate(reference_pairs):
        if u in g2l and v in g2l and u != v:
            keep_idx.append(i)
            lu, lv = g2l[u], g2l[v]
            scores.append(float((min(lu, lv), max(lu, lv)) in edge_set))
            kept_pairs.append((u, v))

    if not keep_idx:
        raise RuntimeError("No reference pairs mapped to this generated graph. Check orig_id alignment.")

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    labels = reference_labels[keep_idx_arr]
    scores_arr = np.asarray(scores, dtype=np.float32)
    sens_mask = pair_same_group_mask(kept_pairs, reference_node_groups)
    metrics = {
        "overlap/auc": safe_auc(labels, scores_arr),
        "overlap/num_eval_pairs": float(len(keep_idx)),
        "overlap/edge_presence_rate": float(scores_arr.mean()) if scores_arr.size else float("nan"),
    }
    raw = {
        "keep_idx": keep_idx_arr,
        "labels": labels,
        "scores": scores_arr,
        "sens_mask": sens_mask.astype(bool),
    }
    return metrics, raw


# -----------------------------
# sample.py-compatible GAE LP
# -----------------------------

class SamplePyGCN(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_layers: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lins = nn.ModuleList()
        if num_layers >= 2:
            self.lins.append(nn.Linear(in_size, hidden_size))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_size, hidden_size))
            self.lins.append(nn.Linear(hidden_size, out_size))
        else:
            self.lins.append(nn.Linear(in_size, out_size))
        self.dropout = dropout

    def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        for lin in self.lins[:-1]:
            H = A @ lin(H)
            H = F.relu(H)
            H = F.dropout(H, p=self.dropout, training=self.training)
        return A @ self.lins[-1](H)


class SamplePyGAE(nn.Module):
    def __init__(self, in_size: int, num_layers: int, hidden_size: int, dropout: float):
        super().__init__()
        self.gcn = SamplePyGCN(in_size, hidden_size, num_layers, hidden_size, dropout)

    def forward(self, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return self.gcn(A, Z)


def samplepy_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def samplepy_normalize_adjacency(num_nodes: int, undirected_edges: torch.Tensor) -> torch.Tensor:
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if undirected_edges.numel() > 0:
        row, col = undirected_edges.long().cpu()
        A[row, col] = 1.0
        A[col, row] = 1.0
    A_hat = A + torch.eye(num_nodes, dtype=torch.float32)
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt.view(-1, 1) * A_hat * deg_inv_sqrt.view(1, -1)


def samplepy_get_edge_split(A_dense: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A_dense_upper = torch.triu(A_dense, diagonal=1)
    real_edges = A_dense_upper.nonzero()
    if real_edges.size(0) < 3:
        raise ValueError("Not enough positive edges for sample.py 80/10/10 GAE split.")

    real_edges = real_edges[torch.randperm(real_edges.size(0))]
    num_real = len(real_edges)
    num_train = int(num_real * 0.8)
    num_val = int(num_real * 0.1)
    num_test = num_real - num_train - num_val
    if num_val <= 0 or num_test <= 0:
        raise ValueError("Not enough positive edges to create non-empty val/test splits.")

    real_train, real_val, real_test = torch.split(real_edges, [num_train, num_val, num_test])

    neg_edges = torch.triu((A_dense == 0).float(), diagonal=1).nonzero()
    if neg_edges.size(0) < num_val + num_test:
        raise ValueError("Not enough negative edges for sample.py GAE val/test split.")
    neg_edges = neg_edges[torch.randperm(neg_edges.size(0))]
    neg_val = neg_edges[:num_val]
    neg_test = neg_edges[num_val:num_val + num_test]
    return real_train, real_val, real_test, neg_val, neg_test


def samplepy_prepare_for_gae(data) -> Dict[str, torch.Tensor]:
    data = ensure_features(data)
    num_nodes = int(data.num_nodes)
    full_edges = unique_undirected_edge_index(data.edge_index)
    A_full = samplepy_normalize_adjacency(num_nodes, full_edges)
    A_dense = A_full.clone()

    real_train, real_val, real_test, neg_val, neg_test = samplepy_get_edge_split(A_dense)

    train_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)

    row_train, col_train = real_train.T
    train_mask[row_train, col_train] = True

    edge_val = torch.cat([real_val, neg_val], dim=0)
    row_val, col_val = edge_val.T
    val_mask[row_val, col_val] = True

    edge_test = torch.cat([real_test, neg_test], dim=0)
    row_test, col_test = edge_test.T
    test_mask[row_test, col_test] = True

    A_train = samplepy_normalize_adjacency(num_nodes, real_train.t().contiguous())
    return {
        "A_full": A_full,
        "A_train": A_train,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_train_pos": torch.tensor(float(real_train.size(0))),
        "num_val_pos": torch.tensor(float(real_val.size(0))),
        "num_test_pos": torch.tensor(float(real_test.size(0))),
        "num_test_neg": torch.tensor(float(neg_test.size(0))),
    }


def samplepy_preprocess(A_train: torch.Tensor, A_full: torch.Tensor, X: torch.Tensor, s: torch.Tensor, Y: Optional[torch.Tensor], num_classes: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = samplepy_device()
    A_train = A_train.to(device)
    A_full = A_full.to(device)
    X = X.to(device).float()
    s_device = s.to(device)
    _s_one_hot = F.one_hot(s_device.long(), len(torch.unique(s_device)))
    if Y is not None:
        if num_classes is None:
            num_classes = int(Y.max().item()) + 1
        Y_device = Y.to(device)
        Y_one_hot = F.one_hot(Y_device.long(), num_classes)
        Z = torch.cat([X, _s_one_hot, Y_one_hot], dim=1)
    else:
        Z = X
    A_full_dense = A_full.clone()
    A_full_dense[A_full_dense != 0] = 1.0
    return A_train, Z, A_full_dense


def samplepy_group_fairness(labels: np.ndarray, preds: np.ndarray, pair_same_mask: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    idx_same = np.asarray(pair_same_mask, dtype=bool)
    idx_diff = ~idx_same
    idx_same_y1 = np.bitwise_and(idx_same, labels == 1)
    idx_diff_y1 = np.bitwise_and(idx_diff, labels == 1)

    parity = safe_abs(safe_diff(safe_group_mean(preds, idx_same), safe_group_mean(preds, idx_diff)))
    equality = safe_abs(safe_diff(safe_group_mean(preds, idx_same_y1), safe_group_mean(preds, idx_diff_y1)))
    return parity, equality


@torch.no_grad()
def samplepy_predict(A_train: torch.Tensor, Z: torch.Tensor, group_labels: torch.Tensor, A_full_dense: torch.Tensor, mask: torch.Tensor, model: SamplePyGAE) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
    model.eval()
    device = samplepy_device()
    mask = mask.to(device)
    group_labels = group_labels.to(device)
    Z_out = model(A_train, Z)
    pair_same = (group_labels.unsqueeze(1) == group_labels.unsqueeze(0))[mask].cpu().numpy()
    probs = torch.sigmoid(Z_out @ Z_out.T)[mask].cpu().numpy()
    labels = A_full_dense[mask].cpu().numpy().astype(np.int64)
    sp, eo = samplepy_group_fairness(labels, probs, pair_same)
    raw = {
        "labels": labels,
        "scores": probs,
        "sens_mask": pair_same.astype(bool),
    }
    return safe_auc(labels, probs), sp, eo, raw


def samplepy_config_list() -> List[Dict[str, Any]]:
    hyper_space = {
        "lr": [3e-2, 1e-2, 3e-3, 1e-3],
        "num_layers": [1],
        "hidden_size": [16, 32, 128, 512],
        "dropout": [0.0, 0.1, 0.2],
    }
    priority = ["dropout", "lr", "num_layers", "hidden_size"]
    configs = []
    for values in itertools.product(*(hyper_space[k] for k in priority)):
        configs.append(dict(zip(priority, values)))
    return configs


def samplepy_fit_trial(
    A_train: torch.Tensor,
    Z: torch.Tensor,
    group_labels: torch.Tensor,
    A_full_dense: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    num_layers: int,
    hidden_size: int,
    dropout: float,
    lr: float,
) -> Tuple[float, float, float, SamplePyGAE, Dict[str, float]]:
    device = samplepy_device()
    model = SamplePyGAE(
        in_size=Z.size(1),
        num_layers=int(num_layers),
        hidden_size=int(hidden_size),
        dropout=float(dropout),
    ).to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    best_auc = -1.0
    best_sp = float("nan")
    best_eo = float("nan")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    bad_epochs = 0

    train_dst, train_src = train_mask.to(device).nonzero().T
    train_size = len(train_dst)
    if train_size == 0:
        raise ValueError("Empty sample.py GAE train mask.")
    batch_size = 16384
    num_nodes = Z.size(0)

    for epoch in range(1, 1000 + 1):
        model.train()
        Z_out = model(A_train, Z)

        if train_size <= batch_size:
            batch_dst = train_dst
            batch_src = train_src
        else:
            batch_ids = torch.randint(low=0, high=train_size, size=(batch_size,), device=device)
            batch_dst = train_dst[batch_ids]
            batch_src = train_src[batch_ids]

        pos_pred = (Z_out[batch_src] * Z_out[batch_dst]).sum(dim=-1)
        real_batch_size = len(batch_dst)
        neg_src = torch.randint(0, num_nodes, (real_batch_size,), device=device)
        neg_dst = torch.randint(0, num_nodes, (real_batch_size,), device=device)
        neg_pred = (Z_out[neg_src] * Z_out[neg_dst]).sum(dim=-1)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        label = torch.cat([
            torch.ones(real_batch_size, device=device),
            torch.zeros(real_batch_size, device=device),
        ], dim=0)
        loss = loss_func(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        auc, sp, eo, _raw = samplepy_predict(A_train, Z, group_labels, A_full_dense, val_mask, model)
        if np.isfinite(auc) and auc > best_auc:
            bad_epochs = 0
            best_auc = float(auc)
            best_sp = float(sp)
            best_eo = float(eo)
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if bad_epochs == 5:
            break

    model.load_state_dict(best_state)
    meta = {
        "best_val_auc": float(best_auc) if best_auc >= 0 else float("nan"),
        "best_val_sp": float(best_sp),
        "best_val_eo": float(best_eo),
        "best_epoch": float(best_epoch),
        "best_num_layers": float(num_layers),
        "best_hidden_dim": float(hidden_size),
        "best_dropout": float(dropout),
        "best_lr": float(lr),
    }
    return best_auc, best_sp, best_eo, model, meta


def samplepy_train_and_eval(data, group_attr: str) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    data = ensure_features(data)
    split = samplepy_prepare_for_gae(data)
    X = data.x.detach().cpu().float()
    group_labels = get_lp_group_vector(data, preferred_attr=group_attr).long()
    Y = None
    num_classes = None

    A_train, Z, A_full_dense = samplepy_preprocess(
        split["A_train"],
        split["A_full"],
        X,
        group_labels,
        Y,
        num_classes,
    )

    best_auc = -1.0
    best_model: Optional[SamplePyGAE] = None
    best_meta: Dict[str, float] = {}
    for config in samplepy_config_list():
        trial_auc, _trial_sp, _trial_eo, trial_model, trial_meta = samplepy_fit_trial(
            A_train,
            Z,
            group_labels,
            A_full_dense,
            split["train_mask"],
            split["val_mask"],
            **config,
        )
        if np.isfinite(trial_auc) and trial_auc > best_auc:
            best_auc = float(trial_auc)
            best_model = trial_model
            best_meta = trial_meta
        if trial_auc == 1.0:
            break

    if best_model is None:
        raise RuntimeError("Failed to train sample.py-compatible GAE with finite validation AUC.")

    test_auc, test_sp, test_eo, raw = samplepy_predict(
        A_train,
        Z,
        group_labels,
        A_full_dense,
        split["test_mask"],
        best_model,
    )

    metrics = {
        "lp/auc": float(test_auc),
        "lp/score_sp_gap": float(test_sp),
        "lp/score_sp_abs_gap": float(test_sp),
        "lp/sp_gap": float(test_sp),
        "lp/sp_abs_gap": float(test_sp),
        "lp/eo_gap": float(test_eo),
        "lp/eo_abs_gap": float(test_eo),
        "lp/score_mean_sensitive": safe_group_mean(raw["scores"], raw["sens_mask"]),
        "lp/score_mean_nonsensitive": safe_group_mean(raw["scores"], ~raw["sens_mask"]),
        "lp/hard_rate_sensitive": float("nan"),
        "lp/hard_rate_nonsensitive": float("nan"),
        "lp/num_eval_pairs": float(split["test_mask"].sum().item()),
    }
    meta = {
        **best_meta,
        "train_num_pos": float(split["num_train_pos"].item()),
        "val_num_pos": float(split["num_val_pos"].item()),
        "test_num_pos": float(split["num_test_pos"].item()),
        "test_num_neg": float(split["num_test_neg"].item()),
        "test_num_pairs": float(split["test_mask"].sum().item()),
    }
    return metrics, raw, meta


def samplepy_aggregate_fairness(labels: np.ndarray, scores: np.ndarray, sens_mask: np.ndarray) -> Dict[str, float]:
    sp, eo = samplepy_group_fairness(labels, scores, sens_mask)
    return {
        "auc": safe_auc(labels, scores),
        "score_sp_gap": sp,
        "score_sp_abs_gap": sp,
        "sp_gap": sp,
        "sp_abs_gap": sp,
        "eo_gap": eo,
        "eo_abs_gap": eo,
        "score_mean_sensitive": safe_group_mean(scores, sens_mask),
        "score_mean_nonsensitive": safe_group_mean(scores, ~sens_mask),
        "hard_rate_sensitive": float("nan"),
        "hard_rate_nonsensitive": float("nan"),
        "num_eval_pairs": float(len(labels)),
    }


# -----------------------------
# Ensemble summaries
# -----------------------------

def ensemble_mean_scores(raw_results: List[Dict[str, np.ndarray]], full_num_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    score_sums = np.zeros(full_num_pairs, dtype=np.float64)
    score_counts = np.zeros(full_num_pairs, dtype=np.int64)
    for r in raw_results:
        idx = r["keep_idx"]
        score_sums[idx] += r["scores"]
        score_counts[idx] += 1
    valid = score_counts > 0
    mean_scores = np.zeros(full_num_pairs, dtype=np.float64)
    mean_scores[valid] = score_sums[valid] / score_counts[valid]
    return mean_scores, valid


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_graphs(
    graphs: List[Any],
    args: argparse.Namespace,
    *,
    reference_graph_path: Optional[str] = None,
    total_loaded: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(int(getattr(args, "seed", 0)))

    label_attr = getattr(args, "label_attr", None) or getattr(args, "sensitive_attr", None) or "y"
    sensitive_attr = getattr(args, "sensitive_attr", None) or label_attr
    max_pos_edges = int(getattr(args, "max_pos_edges", 20000))
    neg_ratio = float(getattr(args, "neg_ratio", 1.0))
    seed = int(getattr(args, "seed", 0))

    if total_loaded is None:
        total_loaded = len(graphs)

    graph_index = getattr(args, "graph_index", None)
    if graph_index is not None:
        if graph_index < 0 or graph_index >= total_loaded:
            raise IndexError(f"graph_index={graph_index} out of range for {total_loaded} graphs")
        graphs = [graphs[graph_index]]

    if reference_graph_path is None:
        reference_graph_path = str(Path.cwd() / "graphs" / f"{args.dataset}_feat.pkl")

    g_ref = load_reference_graph_from_dataset(reference_graph_path, args.dataset)
    reference_pairs, reference_labels = build_fixed_eval_pairs(
        g_ref,
        max_pos_edges=max_pos_edges,
        neg_ratio=neg_ratio,
        seed=seed,
    )
    reference_node_groups = build_reference_node_group_map(g_ref, sensitive_attr)

    per_graph_rows: List[Dict[str, Any]] = []
    overlap_raw_rows: List[Dict[str, np.ndarray]] = []
    lp_raw_rows: List[Dict[str, np.ndarray]] = []

    out_per_graph_csv = getattr(args, "out_per_graph_csv", None)

    for i, data in enumerate(graphs):
        original_idx = graph_index if graph_index is not None else i
        row: Dict[str, Any] = {
            "graph_idx": float(original_idx),
            "lp/model": "samplepy_gae_1layer",
            "lp_protocol": "samplepy_gae",
        }

        overlap_metrics, overlap_raw = edge_overlap_on_fixed_pairs(
            data=data,
            reference_pairs=reference_pairs,
            reference_labels=reference_labels,
            reference_node_groups=reference_node_groups,
        )
        row.update(overlap_metrics)
        overlap_raw_rows.append(overlap_raw)

        try:
            lp_metrics, lp_raw, meta = samplepy_train_and_eval(data, group_attr=sensitive_attr)
            row["lp/best_val_auc"] = float(meta.get("best_val_auc", float("nan")))
            row["lp/best_epoch"] = float(meta.get("best_epoch", float("nan")))
            row["lp/best_num_layers"] = float(meta.get("best_num_layers", 1.0))
            row["lp/best_hidden_dim"] = float(meta.get("best_hidden_dim", float("nan")))
            row["lp/best_dropout"] = float(meta.get("best_dropout", float("nan")))
            row["lp/best_lr"] = float(meta.get("best_lr", float("nan")))
            row["lp/train_num_pos"] = float(meta.get("train_num_pos", float("nan")))
            row["lp/val_num_pos"] = float(meta.get("val_num_pos", float("nan")))
            row["lp/test_num_pos"] = float(meta.get("test_num_pos", float("nan")))
            row["lp/test_num_neg"] = float(meta.get("test_num_neg", float("nan")))
            row["lp/test_num_pairs"] = float(meta.get("test_num_pairs", float("nan")))
            row.update(lp_metrics)
            lp_raw_rows.append(lp_raw)
        except Exception as exc:
            row["lp/error"] = str(exc)
            for key in [
                "lp/auc",
                "lp/score_sp_gap",
                "lp/score_sp_abs_gap",
                "lp/sp_gap",
                "lp/sp_abs_gap",
                "lp/eo_gap",
                "lp/eo_abs_gap",
                "lp/score_mean_sensitive",
                "lp/score_mean_nonsensitive",
                "lp/hard_rate_sensitive",
                "lp/hard_rate_nonsensitive",
                "lp/num_eval_pairs",
                "lp/best_val_auc",
                "lp/best_epoch",
                "lp/best_num_layers",
                "lp/best_hidden_dim",
                "lp/best_dropout",
                "lp/best_lr",
                "lp/train_num_pos",
                "lp/val_num_pos",
                "lp/test_num_pos",
                "lp/test_num_neg",
                "lp/test_num_pairs",
            ]:
                row[key] = float("nan")

        add_compat_metric_aliases(row)
        per_graph_rows.append(row)
        if out_per_graph_csv:
            write_csv(per_graph_rows, Path(out_per_graph_csv))

    summary: Dict[str, Any] = {
        "num_loaded_graphs": float(total_loaded),
        "num_evaluated_graphs": float(len(per_graph_rows)),
        "reference_num_pairs": float(len(reference_pairs)),
        "reference_pos_pairs": float(reference_labels.sum()),
        "reference_neg_pairs": float((reference_labels == 0).sum()),
        "lp/model": "samplepy_gae_1layer",
        "lp_protocol": "samplepy_gae",
        "lp_search": 1.0,
        "lp_split": "samplepy_prepare_for_GAE_80_10_10",
    }

    numeric_keys: List[str] = []
    seen = set()
    for row in per_graph_rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and key not in seen:
                seen.add(key)
                numeric_keys.append(key)

    for key in numeric_keys:
        vals = [float(row[key]) for row in per_graph_rows if key in row]
        summary[f"{key}_mean"] = safe_mean(vals)
        summary[f"{key}_std"] = safe_std(vals)

    overlap_mean_scores, overlap_valid = ensemble_mean_scores(overlap_raw_rows, len(reference_pairs))
    if overlap_valid.sum() > 0:
        summary["ensemble_overlap/auc"] = safe_auc(reference_labels[overlap_valid], overlap_mean_scores[overlap_valid])
        summary["ensemble_overlap/coverage_pairs"] = float(overlap_valid.sum())

    if lp_raw_rows:
        all_lp_labels = np.concatenate([r["labels"] for r in lp_raw_rows], axis=0)
        all_lp_scores = np.concatenate([r["scores"] for r in lp_raw_rows], axis=0)
        all_lp_sens = np.concatenate([r["sens_mask"] for r in lp_raw_rows], axis=0)
        agg_lp = samplepy_aggregate_fairness(all_lp_labels, all_lp_scores, all_lp_sens)
        for key, value in agg_lp.items():
            summary[f"aggregate_lp/{key}"] = value
        summary["aggregate_lp/num_graphs"] = float(len(lp_raw_rows))

    add_compat_metric_aliases(summary)
    if "aggregate_lp/sp_gap" in summary and "aggregate_fair_gap" not in summary:
        summary["aggregate_fair_gap"] = float(summary["aggregate_lp/sp_gap"])
    if "aggregate_lp/sp_abs_gap" in summary and "aggregate_fair_abs_gap" not in summary:
        summary["aggregate_fair_abs_gap"] = float(summary["aggregate_lp/sp_abs_gap"])
    if "aggregate_lp/auc" in summary and "aggregate_value/linkpred_auc" not in summary:
        summary["aggregate_value/linkpred_auc"] = float(summary["aggregate_lp/auc"])

    out_summary_csv = getattr(args, "out_summary_csv", None)
    if out_summary_csv:
        write_csv([summary], Path(out_summary_csv))

    return per_graph_rows, summary


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved generated graphs with sample.py-compatible GAE link prediction.")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to *.pyg.pt or *.pyg_full.pt")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name used to find graphs/{dataset}_feat.pkl")
    parser.add_argument("--graph_index", type=int, default=None, help="Optional: evaluate only one graph from the saved list")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--label_attr", type=str, default="y")
    parser.add_argument("--sensitive_attr", type=str, default=None)
    parser.add_argument("--sensitive_value", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--edge_sensitive_mode", type=str, default="either", choices=["either", "both"], help=argparse.SUPPRESS)

    parser.add_argument("--max_pos_edges", type=int, default=20000, help="Max positive reference edges for overlap evaluation pairs")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Neg/pos ratio for fixed reference pairs")

    # Legacy arguments accepted for compatibility with existing grid scripts.
    parser.add_argument("--lp_model", "--backbone", dest="lp_model", type=str, default="samplepy_gae")
    parser.add_argument("--lp_num_layers", type=int, default=1)
    parser.add_argument("--lp_hidden_dim", type=int, default=128)
    parser.add_argument("--lp_out_dim", type=int, default=64)
    parser.add_argument("--lp_dropout", type=float, default=0.1)
    parser.add_argument("--lp_lr", type=float, default=1e-2)
    parser.add_argument("--lp_weight_decay", type=float, default=0.0)
    parser.add_argument("--lp_epochs", type=int, default=1000)
    parser.add_argument("--lp_patience", type=int, default=5)
    parser.add_argument("--lp_batch_size", type=int, default=16384)
    parser.add_argument("--lp_test_ratio", type=float, default=0.1, help="Legacy no-op; sample.py split is fixed at 80/10/10.")
    parser.add_argument("--lp_val_ratio", type=float, default=0.1, help="Legacy no-op; sample.py split is fixed at 80/10/10.")
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lp_search", action="store_true", help="Legacy no-op; sample.py GAE grid is always used.")
    parser.add_argument("--lp_search_hidden_dims", type=int, nargs="+", default=[16, 32, 128, 512])
    parser.add_argument("--lp_search_lrs", type=float, nargs="+", default=[3e-2, 1e-2, 3e-3, 1e-3])
    parser.add_argument("--lp_search_dropouts", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--lp_search_num_layers", type=int, nargs="+", default=[1])

    parser.add_argument("--out_per_graph_csv", type=str, default=None)
    parser.add_argument("--out_summary_csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph_path)
    out_dir = graph_path.parent
    stem = graph_path.name[:-3] if graph_path.name.endswith(".pt") else graph_path.name

    if args.out_per_graph_csv is None:
        args.out_per_graph_csv = str(out_dir / f"{stem}.overlap_lp_gae_per_graph.csv")
    if args.out_summary_csv is None:
        args.out_summary_csv = str(out_dir / f"{stem}.overlap_lp_gae_summary.csv")

    graphs = load_saved_graphs(args.graph_path)
    _per_graph_rows, summary = evaluate_graphs(
        graphs=graphs,
        args=args,
        reference_graph_path=args.graph_path,
        total_loaded=len(graphs),
    )
    print(summary)


if __name__ == "__main__":
    main()

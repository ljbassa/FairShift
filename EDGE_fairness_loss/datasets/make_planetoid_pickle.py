# <repo_root>/datasets/make_planetoid_pickle.py
"""
Save a single NetworkX pickle that contains:
- edges (undirected)
- node attrs: x (float32 vec), y (int), orig_id (int)
- optional masks if present (train/val/test)
"""

import os
import argparse
import pickle as pkl
from typing import Optional, Dict, Tuple

import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import Amazon, Planetoid, WebKB
from torch_geometric.utils import to_undirected

_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Planetoid
    "cora": ("planetoid", "Cora"),
    "citeseer": ("planetoid", "CiteSeer"),
    "pubmed": ("planetoid", "PubMed"),
    # WebKB
    "cornell": ("webkb", "Cornell"),
    "texas": ("webkb", "Texas"),
    "wisconsin": ("webkb", "Wisconsin"),
    # Amazon
    "amazon_photo": ("amazon", "Photo"),
    "amazon_computer": ("amazon", "Computers"),
    "amazon_computers": ("amazon", "Computers"),
}

def _load_pyg_data(dataset_key: str, root: str):
    key = dataset_key.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset='{dataset_key}'. Supported={sorted(_REGISTRY.keys())}")
    ds_type, ds_name = _REGISTRY[key]

    # root 아래에 데이터셋 종류별 폴더 분리 저장
    if ds_type == "planetoid":
        ds_root = os.path.join(root, "Planetoid")
        dataset = Planetoid(root=ds_root, name=ds_name)
    elif ds_type == "webkb":
        ds_root = os.path.join(root, "WebKB")
        dataset = WebKB(root=ds_root, name=ds_name)
    elif ds_type == "amazon":
        ds_root = os.path.join(root, "Amazon")
        dataset = Amazon(root=ds_root, name=ds_name)
    else:
        raise ValueError(f"Unsupported dataset type='{ds_type}' for dataset='{dataset_key}'")

    return dataset[0]

def build_nx_with_xy(dataset_key: str, root: str, max_nodes: Optional[int] = None):
    data = _load_pyg_data(dataset_key, root=root)

    edge_index = to_undirected(data.edge_index)
    num_nodes = int(data.num_nodes)

    # optional induced subgraph (앞쪽 max_nodes)
    if max_nodes is not None and int(max_nodes) < num_nodes:
        keep = torch.arange(int(max_nodes), dtype=torch.long)
        keep_set = set(keep.tolist())

        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        edges = [(u, v) for u, v in zip(src, dst) if (u in keep_set and v in keep_set)]

        num_nodes = int(max_nodes)
        x = data.x[keep].cpu().numpy().astype(np.float32) if hasattr(data, "x") else None
        y = data.y[keep].cpu().numpy().astype(np.int64) if hasattr(data, "y") else None

    else:
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        edges = list(zip(src, dst))

        x = data.x.cpu().numpy().astype(np.float32) if hasattr(data, "x") else None
        y = data.y.cpu().numpy().astype(np.int64) if hasattr(data, "y") else None


    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)

    for i in range(num_nodes):
        g.nodes[i]["orig_id"] = int(i)
        if x is not None:
            g.nodes[i]["x"] = x[i]
        if y is not None:
            g.nodes[i]["y"] = int(y[i])

    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    help="cora/citeseer/pubmed/cornell/texas/wisconsin/amazon_photo/amazon_computer")
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--max_nodes", type=int, default=None)
    args = ap.parse_args()

    g = build_nx_with_xy(args.dataset, root=args.root, max_nodes=args.max_nodes)

    out = args.out or f"graphs/{args.dataset}_feat.pkl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(out, "wb") as f:
        pkl.dump(g, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"[OK] saved: {out} | nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

if __name__ == "__main__":
    main()

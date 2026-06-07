import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from data import load_dataset, preprocess, load_datasets_nc


def decode_binary_features(X_one_hot_3d: torch.Tensor) -> np.ndarray:
    X = X_one_hot_3d.argmax(dim=-1).transpose(0, 1)
    return X.cpu().numpy().astype(np.float32)


def main(dataset: str, out_path: str):
    if dataset in ["cora", "citeseer", "amazon_photo", "amazon_computer"]:
        g_real = load_dataset(dataset)
    else:
        g_real = load_datasets_nc(dataset)

    X_one_hot_3d_real, s_real, y_real, E_one_hot_real, \
        X_marginal, s_marginal, y_marginal, E_marginal, \
        X_cond_s_marginals, X_cond_y_marginals, y_cond_s_marginals, p_values = preprocess(g_real)

    X = decode_binary_features(X_one_hot_3d_real)
    s = s_real.cpu().numpy().astype(np.int64)

    if y_real is None:
        y = s.copy()
        sens = s.copy()
    else:
        y = y_real.cpu().numpy().astype(np.int64)
        sens = s.copy()

    G = nx.Graph()
    num_nodes = len(s)

    for i in range(num_nodes):
        G.add_node(
            i,
            orig_id=int(i),
            x=X[i],
            y=int(y[i]),
            sens=int(sens[i]),
        )

    src, dst = g_real.edges()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        G.add_edge(int(u), int(v))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved reference graph to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset, args.out_path)
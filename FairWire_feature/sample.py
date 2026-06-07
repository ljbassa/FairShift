import os
import pickle
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from data import load_dataset, preprocess, load_datasets_nc
from eval_utils import Evaluator
from setup_utils import set_seed


def decode_binary_features(X_one_hot_3d: torch.Tensor) -> np.ndarray:
    """
    X_one_hot_3d: (F, N, 2)
    return: (N, F) float32 numpy array
    """
    X = X_one_hot_3d.argmax(dim=-1).transpose(0, 1)
    return X.cpu().numpy().astype(np.float32)


def decode_classes(one_hot: torch.Tensor):
    if one_hot is None:
        return None
    return one_hot.argmax(dim=-1).cpu().numpy().astype(np.int64)


def build_pyg_data_from_sample(
    X_0_one_hot: torch.Tensor,
    s_0_one_hot: torch.Tensor,
    y_0_one_hot: torch.Tensor,
    E_0: torch.Tensor,
    node_orig_id: torch.Tensor,
) -> Data:
    """
    evaluator용 PyG Data 생성.
    두 번째 evaluator가 기대하는 핵심 필드:
      - x
      - edge_index
      - orig_id
      - y / sens
    """
    E_0 = E_0.detach().cpu()
    X = decode_binary_features(X_0_one_hot)   # (N, F)
    s = decode_classes(s_0_one_hot)           # (N,)
    y = decode_classes(y_0_one_hot)           # (N,) or None

    num_nodes = E_0.size(0)

    # undirected unique edges
    src, dst = torch.triu(E_0, diagonal=1).nonzero(as_tuple=True)

    # PyG evaluator 쪽에서는 bidirectional edge_index여도 unique_undirected_edge_index로 처리함
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).long()

    orig_id = node_orig_id.detach().cpu().long()

    # cora/citeseer/amazon_* 계열에서는 y_0_one_hot이 None일 수 있음
    # 이 경우 evaluator 기본값(--sensitive_attr y)과 맞추기 위해 y를 s로 둠
    if y is None:
        y_np = s.copy()
        sens_np = s.copy()
    else:
        y_np = y.copy()
        sens_np = s.copy()

    data = Data(
        x=torch.from_numpy(X).float(),
        edge_index=edge_index,
        orig_id=orig_id,
        y=torch.from_numpy(y_np).long(),
        sens=torch.from_numpy(sens_np).long(),
    )
    return data


def build_nx_graph_from_sample(
    dataset_name: str,
    E_0: torch.Tensor,
    X_0_one_hot: torch.Tensor,
    s_0_one_hot: torch.Tensor,
    y_0_one_hot: torch.Tensor = None,
    node_orig_id: torch.Tensor = None,
) -> nx.Graph:
    if node_orig_id is None:
        node_orig_id = torch.arange(E_0.size(0), dtype=torch.long)
    else:
        node_orig_id = node_orig_id.detach().cpu().long()
    
    """
    inspection용 NetworkX Graph 생성.
    node attrs:
      - orig_id
      - x
      - y
      - sens
    """
    E_0 = E_0.detach().cpu()
    X = decode_binary_features(X_0_one_hot)
    s = decode_classes(s_0_one_hot)
    y = decode_classes(y_0_one_hot)

    G = nx.Graph()
    num_nodes = E_0.size(0)

    if y is None:
        y_np = s.copy()
        sens_np = s.copy()
    else:
        y_np = y.copy()
        sens_np = s.copy()

    for node_id in range(num_nodes):
        G.add_node(
            node_id,
            orig_id=int(node_orig_id[node_id]),
            x=X[node_id],
            y=int(y_np[node_id]),
            sens=int(sens_np[node_id]),
        )

    src, dst = torch.triu(E_0, diagonal=1).nonzero(as_tuple=True)
    edges = [(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())]
    G.add_edges_from(edges)

    G.graph["dataset"] = dataset_name
    G.graph["num_features"] = int(X.shape[1])

    return G


def save_sample_as_pkl(
    dataset_name: str,
    save_dir: str,
    sample_idx: int,
    E_0: torch.Tensor,
    X_0_one_hot: torch.Tensor,
    s_0_one_hot: torch.Tensor,
    y_0_one_hot: torch.Tensor = None,
    node_orig_id: torch.Tensor = None,
):
    os.makedirs(save_dir, exist_ok=True)

    nx_graph = build_nx_graph_from_sample(
        dataset_name=dataset_name,
        E_0=E_0,
        X_0_one_hot=X_0_one_hot,
        s_0_one_hot=s_0_one_hot,
        y_0_one_hot=y_0_one_hot,
        node_orig_id=node_orig_id,
    )

    save_path = os.path.join(save_dir, f"sample_{sample_idx:03d}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Saved pkl] {save_path}")


def main(args):
    state_dict = torch.load(args.model_path, map_location="cpu")
    dataset = state_dict["dataset"]

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]

    print(f"Loaded GraphMaker-{model_name} model trained on {dataset}")
    print(f"Val Nll {state_dict['best_val_nll']}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if dataset in ["cora", "citeseer", "amazon_photo", "amazon_computer"]:
        g_real = load_dataset(dataset)
    else:
        g_real = load_datasets_nc(dataset)

    X_one_hot_3d_real, s_real, y_real, E_one_hot_real, \
        X_marginal, s_marginal, y_marginal, E_marginal, \
        X_cond_s_marginals, X_cond_y_marginals, y_cond_s_marginals, p_values = preprocess(g_real)

    evaluator = None
    if not args.skip_internal_eval:
        s_one_hot_real = F.one_hot(s_real)
        if y_real is not None:
            Y_one_hot_3d_real = F.one_hot(y_real)
        else:
            Y_one_hot_3d_real = None

        evaluator = Evaluator(
            dataset,
            os.path.dirname(args.model_path),
            g_real,
            X_one_hot_3d_real,
            s_one_hot_real,
            Y_one_hot_3d_real
        )

    if y_real is not None:
        y_marginal = y_marginal.to(device)
        y_cond_s_marginals = y_cond_s_marginals.to(device)

    X_marginal = X_marginal.to(device)
    s_marginal = s_marginal.to(device)
    E_marginal = E_marginal.to(device)
    X_cond_s_marginals = X_cond_s_marginals.to(device)
    num_nodes = s_real.size(0)

    from Model import ModelSync

    model = ModelSync(
        X_marginal=X_marginal,
        s_marginal=s_marginal,
        y_marginal=y_marginal,
        E_marginal=E_marginal,
        num_nodes=num_nodes,
        p_values=p_values,
        y_cond_s_marginal=y_cond_s_marginals,
        gnn_X_config=train_yaml_data["gnn_X"],
        gnn_E_config=train_yaml_data["gnn_E"],
        **train_yaml_data["diffusion"]
    ).to(device)

    model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
    model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])
    model.to(device)
    model.eval()

    set_seed(args.seed)

    saved_graphs = []

    if args.save_pkl_dir is not None:
        Path(args.save_pkl_dir).mkdir(parents=True, exist_ok=True)

    if args.save_pt_path is not None:
        Path(args.save_pt_path).parent.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(args.num_samples):
        X_0_one_hot, s_0_one_hot, y_0_one_hot, E_0, node_orig_id = model.sample(
            is_diff_X=True,
            fixed_X_one_hot_3d=X_one_hot_3d_real,
            fixed_s=s_real,
            fixed_y=y_real,
        )

        # 기존 evaluator용 DGL graph
        src_all, dst_all = E_0.nonzero().T
        g_sample = dgl.graph((src_all, dst_all), num_nodes=num_nodes).cpu()

        if not args.skip_internal_eval:
            evaluator.add_sample(
                g_sample,
                X_0_one_hot.cpu(),
                s_0_one_hot.cpu(),
                y_0_one_hot.cpu() if y_0_one_hot is not None else y_0_one_hot
            )

        # second evaluator용 PyG Data
        pyg_data = build_pyg_data_from_sample(
            X_0_one_hot=X_0_one_hot.cpu(),
            s_0_one_hot=s_0_one_hot.cpu(),
            y_0_one_hot=y_0_one_hot.cpu() if y_0_one_hot is not None else y_0_one_hot,
            E_0=E_0.cpu(),
            node_orig_id=node_orig_id,
        )
        saved_graphs.append(pyg_data)

        # optional individual pkl 저장
        if args.save_pkl_dir is not None:
            save_sample_as_pkl(
                dataset_name=dataset,
                save_dir=args.save_pkl_dir,
                sample_idx=sample_idx,
                E_0=E_0.cpu(),
                X_0_one_hot=X_0_one_hot.cpu(),
                s_0_one_hot=s_0_one_hot.cpu(),
                y_0_one_hot=y_0_one_hot.cpu() if y_0_one_hot is not None else y_0_one_hot,
                node_orig_id=node_orig_id,
            )

    # second evaluator 입력용 pt 저장
    if args.save_pt_path is not None:
        torch.save(saved_graphs, args.save_pt_path)
        print(f"[Saved pt] {args.save_pt_path}  ({len(saved_graphs)} graphs)")

    if not args.skip_internal_eval:
        evaluator.summary()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--gpu", type=int, default=0, required=False, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--seed", type=int, default=0, help="Random seed for graph sampling.")

    # 기존 pkl 저장
    parser.add_argument(
        "--save_pkl_dir",
        type=str,
        default=None,
        help="If set, save each generated sample as a NetworkX .pkl file."
    )

    # 두 번째 evaluator용 pt 저장
    parser.add_argument(
        "--save_pt_path",
        type=str,
        default=None,
        help="If set, save all generated samples as a list[PyG Data] .pyg.pt file."
    )

    parser.add_argument(
        "--skip_internal_eval",
        action="store_true",
        help="Skip the repo's built-in evaluator.summary() and only save graphs."
    )

    args = parser.parse_args()
    main(args)

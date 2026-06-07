import os
import json
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


CHECKPOINT_FALLBACK_NAMES = (
    "full_model_best.pt",
    "full_model_best.pth",
    "controller_best.pt",
    "controller_best.pth",
    "full_model_final.pt",
    "full_model_final.pth",
    "controller_final.pt",
    "controller_final.pth",
    "full_model_last.pt",
    "full_model_last.pth",
    "controller_last.pt",
    "controller_last.pth",
)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower().strip()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


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


def extract_pyg_subgraph_by_nodes(data: Data, kept_nodes) -> Data:
    kept_nodes = sorted(int(v) for v in kept_nodes)
    if len(kept_nodes) == int(data.num_nodes):
        return data.clone()

    node_map = {old: new for new, old in enumerate(kept_nodes)}
    kept_set = set(kept_nodes)
    edge_index = data.edge_index.detach().cpu().long()
    new_edges = []
    for u, v in edge_index.t().tolist():
        if int(u) in kept_set and int(v) in kept_set:
            new_edges.append([node_map[int(u)], node_map[int(v)]])
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)

    index = torch.tensor(kept_nodes, dtype=torch.long)
    out = Data(edge_index=new_edge_index, num_nodes=len(kept_nodes))
    for attr in ("x", "y", "sens", "orig_id"):
        if hasattr(data, attr) and getattr(data, attr) is not None:
            value = getattr(data, attr).detach().cpu()
            setattr(out, attr, value.index_select(0, index))
    return out


def maybe_keep_largest_cc_pyg(data: Data, largest_cc: bool) -> Data:
    if not largest_cc:
        return data.clone()
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    edge_index = data.edge_index.detach().cpu().long()
    graph.add_edges_from((int(u), int(v)) for u, v in edge_index.t().tolist() if int(u) != int(v))
    if graph.number_of_nodes() == 0 or nx.is_connected(graph):
        return data.clone()
    kept_nodes = max(nx.connected_components(graph), key=len)
    return extract_pyg_subgraph_by_nodes(data, kept_nodes)


def build_nx_graph_from_pyg_data(dataset_name: str, data: Data) -> nx.Graph:
    graph = nx.Graph()
    data = data.cpu()
    for node_id in range(int(data.num_nodes)):
        attrs = {}
        if hasattr(data, "orig_id") and data.orig_id is not None:
            attrs["orig_id"] = int(data.orig_id[node_id])
        if hasattr(data, "x") and data.x is not None:
            attrs["x"] = data.x[node_id].numpy()
        if hasattr(data, "y") and data.y is not None:
            attrs["y"] = int(data.y[node_id])
        if hasattr(data, "sens") and data.sens is not None:
            attrs["sens"] = int(data.sens[node_id])
        graph.add_node(node_id, **attrs)
    edge_index = data.edge_index.detach().cpu().long()
    for u, v in edge_index.t().tolist():
        if int(u) != int(v):
            graph.add_edge(int(u), int(v))
    graph.graph["dataset"] = dataset_name
    return graph


def sanitize_tag_value(value):
    if value is None:
        return "none"
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).replace(".", "p")


def build_sample_tag(args, controller_enabled: bool):
    stem = Path(getattr(args, "sample_source_path", args.model_path)).stem
    eta_tag = "controller" if controller_enabled else (
        getattr(args, 'fair_score_eta', None)
        if getattr(args, 'fair_score_eta', None) is not None
        else getattr(args, 'sp_eta', None)
    )
    k_tag = "controller" if controller_enabled else (
        getattr(args, 'fair_score_k', None)
        if getattr(args, 'fair_score_k', None) is not None
        else getattr(args, 'sp_k', None)
    )
    return (
        f"{sanitize_tag_value(stem)}"
        f"_seed{sanitize_tag_value(args.seed)}"
        f"_eta{sanitize_tag_value(eta_tag)}"
        f"_k{sanitize_tag_value(k_tag)}"
        f"_ctrl{int(bool(controller_enabled))}"
    )


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


def print_sp_shift_report(sample_idx: int, sp_stats):
    if not sp_stats:
        print(f"[SP guidance] sample {sample_idx:03d}: no per-step stats recorded")
        return

    print(f"[SP guidance] sample {sample_idx:03d}")
    for stat in sp_stats:
        parts = [
            f"t={stat.get('t')}",
            f"eta_t={stat.get('eta_t', 0.0):.6g}",
            f"k={stat.get('sp_k', 0.0):.6g}",
            f"delta_sp_before={stat.get('delta_sp_before', 0.0):.6g}",
            f"abs_before={stat.get('abs_delta_sp_before', 0.0):.6g}",
            f"n_same={stat.get('n_same', 0)}",
            f"n_diff={stat.get('n_diff', 0)}",
            f"shift_applied={stat.get('shift_applied', False)}",
        ]
        if "delta_sp_after_prior" in stat:
            parts.extend([
                f"delta_sp_after_prior={stat.get('delta_sp_after_prior', 0.0):.6g}",
                f"abs_after_prior={stat.get('abs_delta_sp_after_prior', 0.0):.6g}",
            ])
        print("  " + " | ".join(parts))


def extract_controller_checkpoint(checkpoint):
    if checkpoint is None:
        return None
    if "controller" in checkpoint:
        return checkpoint
    if "fair_controller" in checkpoint:
        return {"controller": checkpoint["fair_controller"]}
    if "fair_score_k_raw" in checkpoint and "fair_score_eta_raw" in checkpoint:
        return {"controller": checkpoint}
    return None


def controller_option(checkpoint, *keys, default=None):
    if checkpoint is None:
        return default
    payload = extract_controller_checkpoint(checkpoint)
    if payload is None and isinstance(checkpoint, dict):
        payload = checkpoint
    if not isinstance(payload, dict):
        return default

    controller = payload.get("controller")
    if isinstance(controller, dict):
        for key in keys:
            if key in controller and controller[key] is not None:
                return controller[key]

    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]

    args = payload.get("args")
    if isinstance(args, dict):
        for key in keys:
            if key in args and args[key] is not None:
                return args[key]
    return default


def float_controller_option(checkpoint, *keys, default=None):
    value = controller_option(checkpoint, *keys, default=default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def has_full_denoiser_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return False
    has_split = "pred_X_state_dict" in checkpoint and "pred_E_state_dict" in checkpoint
    return has_split or "model" in checkpoint


def is_controller_only_checkpoint(checkpoint):
    return extract_controller_checkpoint(checkpoint) is not None and not has_full_denoiser_checkpoint(checkpoint)


def _direct_checkpoint_candidates(path):
    path = Path(path)
    candidates = []
    candidates.append(path)
    if path.suffix == ".pt":
        candidates.append(path.with_suffix(".pth"))
    elif path.suffix == ".pth":
        candidates.append(path.with_suffix(".pt"))

    name_variants = []
    for candidate in list(candidates):
        name = candidate.name
        if name.startswith("full_model_"):
            name_variants.append(candidate.with_name(name.replace("full_model_", "controller_", 1)))
        if name.startswith("controller_"):
            name_variants.append(candidate.with_name(name.replace("controller_", "full_model_", 1)))
    candidates.extend(name_variants)

    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix == ".pt":
            expanded.append(candidate.with_suffix(".pth"))
        elif candidate.suffix == ".pth":
            expanded.append(candidate.with_suffix(".pt"))
    return expanded


def _checkpoint_score(path, requested_path):
    path = Path(path)
    requested_text = str(requested_path)
    path_text = str(path)
    score = 0
    if "best" in path.name:
        score += 300
    elif "final" in path.name:
        score += 200
    elif "last" in path.name:
        score += 100
    if path.name.startswith("full_model_"):
        score += 50
    elif path.name.startswith("controller_"):
        score += 25

    requested_parts = Path(requested_text).parts
    if "controller" in requested_parts:
        idx = requested_parts.index("controller")
        if idx + 1 < len(requested_parts) and requested_parts[idx + 1] in path_text:
            score += 1000
    try:
        score += min(int(path.stat().st_mtime), 2_000_000_000) / 2_000_000_000
    except OSError:
        pass
    return score


def resolve_checkpoint_path(path, allow_search=True, purpose="model"):
    path = Path(path)
    for candidate in _direct_checkpoint_candidates(path):
        if candidate.exists():
            resolved = str(candidate)
            if str(candidate) != str(path):
                print(f"[sample] Resolved {purpose} checkpoint: {path} -> {resolved}")
            return resolved

    if not allow_search:
        raise FileNotFoundError(f"{purpose} checkpoint not found: {path}")

    roots = []
    for root in (Path.cwd() / "wandb", Path.cwd() / "cora_stage2_controller"):
        if root.exists():
            roots.append(root)

    found = []
    for root in roots:
        for name in CHECKPOINT_FALLBACK_NAMES:
            found.extend(root.rglob(name))
    found = sorted(set(found), key=lambda p: _checkpoint_score(p, path), reverse=True)

    if found:
        chosen = found[0]
        print(f"[sample][WARN] {purpose} checkpoint not found: {path}")
        print(f"[sample][WARN] Using nearest saved checkpoint instead: {chosen}")
        if len(found) > 1:
            print("[sample][WARN] Other checkpoint candidates:")
            for candidate in found[1:6]:
                print(f"  - {candidate}")
        return str(chosen)

    searched = ", ".join(str(root) for root in roots) if roots else "(no known checkpoint roots)"
    raise FileNotFoundError(
        f"{purpose} checkpoint not found: {path}. "
        f"Searched fallback names under: {searched}"
    )


def resolve_base_model_path_from_controller(controller_checkpoint):
    if not isinstance(controller_checkpoint, dict):
        return None
    for key in ("base_model_path", "model_path"):
        value = controller_checkpoint.get(key)
        if value:
            return value
    args = controller_checkpoint.get("args")
    if isinstance(args, dict):
        for key in ("controller_pretrained_ckpt", "base_model_path", "model_path"):
            value = args.get(key)
            if value:
                return value
    return None


def main(args):
    args.model_path = resolve_checkpoint_path(
        args.model_path,
        allow_search=args.allow_checkpoint_search,
        purpose="model",
    )
    args.sample_source_path = args.model_path
    state_dict = torch_load(args.model_path, map_location="cpu")

    if is_controller_only_checkpoint(state_dict):
        controller_model_path = args.model_path
        args.controller_path = args.controller_path or controller_model_path
        base_model_path = resolve_base_model_path_from_controller(state_dict)
        if base_model_path is None:
            raise ValueError(
                "Controller-only checkpoint was given as --model_path, but it has no base_model_path. "
                "Pass the stage-1 checkpoint as --model_path and this controller as --controller_path."
            )
        args.model_path = resolve_checkpoint_path(
            base_model_path,
            allow_search=args.allow_checkpoint_search,
            purpose="base model",
        )
        print(f"[sample] Loaded controller-only checkpoint; base model={args.model_path}")
        state_dict = torch_load(args.model_path, map_location="cpu")

    dataset = state_dict["dataset"]

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]

    print(f"Loaded GraphMaker-{model_name} model trained on {dataset}")
    if "best_val_nll" in state_dict:
        print(f"Val Nll {state_dict['best_val_nll']}")
    elif "loss" in state_dict:
        print(f"Controller loss {state_dict['loss']}")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[sample] device={device}")

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

    controller_checkpoint = None
    if args.controller_path is not None:
        args.controller_path = resolve_checkpoint_path(
            args.controller_path,
            allow_search=args.allow_checkpoint_search,
            purpose="controller",
        )
        args.sample_source_path = args.controller_path
        controller_checkpoint = torch_load(args.controller_path, map_location="cpu")
    else:
        controller_checkpoint = extract_controller_checkpoint(state_dict)
    controller_enabled = extract_controller_checkpoint(controller_checkpoint) is not None

    model_fair_score_eta = args.fair_score_eta if args.fair_score_eta is not None else args.sp_eta
    model_fair_score_k = args.fair_score_k if args.fair_score_k is not None else args.sp_k
    model_fair_label_attr = args.fair_label_attr
    if controller_enabled:
        model_fair_score_eta = (
            args.fair_score_eta
            if args.fair_score_eta is not None
            else float_controller_option(
                controller_checkpoint,
                "fair_score_eta_base",
                "fair_score_eta",
                default=args.sp_eta,
            )
        )
        model_fair_score_k = (
            args.fair_score_k
            if args.fair_score_k is not None
            else float_controller_option(
                controller_checkpoint,
                "fair_score_k",
                default=args.sp_k,
            )
        )
        model_fair_label_attr = controller_option(
            controller_checkpoint,
            "fair_label_attr",
            default=args.fair_label_attr,
        )

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
        fair_label_attr=model_fair_label_attr,
        fair_score_eta=model_fair_score_eta,
        fair_score_k=model_fair_score_k,
        fair_score_eta_scale=args.fair_score_eta_scale,
        fair_score_controller_train=controller_enabled,
        fair_score_guidance_normalize=args.fair_score_guidance_normalize,
        **train_yaml_data["diffusion"]
    ).to(device)

    if "pred_X_state_dict" in state_dict and "pred_E_state_dict" in state_dict:
        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"], strict=False)
    else:
        raise KeyError("Checkpoint must contain pred_X_state_dict/pred_E_state_dict or model.")

    if controller_enabled:
        model.load_fair_controller_state_dict(extract_controller_checkpoint(controller_checkpoint), strict=True)
        print("[controller] loaded fair eta/k controller")

    model.to(device)
    model.eval()

    set_seed(args.seed)

    saved_graphs = []
    saved_eval_graphs = []
    saved_nx_eval_graphs = []
    saved_nx_full_graphs = []

    if args.save_pkl_dir is not None:
        Path(args.save_pkl_dir).mkdir(parents=True, exist_ok=True)

    if args.save_pt_path is not None:
        Path(args.save_pt_path).parent.mkdir(parents=True, exist_ok=True)

    save_root = None
    if args.save_samples:
        source_path = getattr(args, "sample_source_path", args.model_path)
        save_root = Path(args.save_dir) if args.save_dir is not None else Path(os.path.dirname(source_path)) / "generated_samples"
        save_root.mkdir(parents=True, exist_ok=True)

    guidance_enabled = bool(args.sp_shift or args.fair_score_sp or controller_enabled)
    guidance_eta = model_fair_score_eta
    guidance_k = model_fair_score_k
    controller_replay_playback = bool(controller_enabled and guidance_enabled)
    if controller_replay_playback:
        print(
            "[sample] controller replay playback enabled: "
            "record frozen unguided z_raw/y first, then apply eta/k logit shift on that replay."
        )

    for sample_idx in range(args.num_samples):
        if controller_replay_playback:
            replay_out = model.sample(
                is_diff_X=True,
                batch_size=args.sample_batch_size,
                num_workers=args.num_workers,
                fixed_X_one_hot_3d=X_one_hot_3d_real,
                fixed_s=s_real,
                fixed_y=y_real,
                return_controller_replay=True,
            )
            replay = replay_out[-1]
            del replay_out
            sample_out = model.sample(
                is_diff_X=True,
                batch_size=args.sample_batch_size,
                num_workers=args.num_workers,
                sp_shift=True,
                sp_eta=guidance_eta,
                sp_k=guidance_k,
                sp_eta_schedule=args.sp_eta_schedule,
                sp_shift_clip=args.sp_shift_clip,
                return_sp_stats=args.sp_report,
                controller_replay=replay,
            )
            del replay
        else:
            sample_out = model.sample(
                is_diff_X=True,
                batch_size=args.sample_batch_size,
                num_workers=args.num_workers,
                fixed_X_one_hot_3d=X_one_hot_3d_real,
                fixed_s=s_real,
                fixed_y=y_real,
                sp_shift=guidance_enabled,
                sp_eta=guidance_eta,
                sp_k=guidance_k,
                sp_eta_schedule=args.sp_eta_schedule,
                sp_shift_clip=args.sp_shift_clip,
                return_sp_stats=args.sp_report,
            )

        sp_stats = None
        if args.sp_report:
            if len(sample_out) == 6:
                X_0_one_hot, s_0_one_hot, y_0_one_hot, E_0, node_orig_id, sp_stats = sample_out
            elif len(sample_out) == 5:
                X_0_one_hot, s_0_one_hot, y_0_one_hot, E_0, node_orig_id = sample_out
                sp_stats = getattr(model, "last_sp_shift_stats", None)
            else:
                raise ValueError(f"Unexpected sample() return length: {len(sample_out)}")
            print_sp_shift_report(sample_idx, sp_stats)
        else:
            if len(sample_out) == 5:
                X_0_one_hot, s_0_one_hot, y_0_one_hot, E_0, node_orig_id = sample_out
            elif len(sample_out) == 6:
                X_0_one_hot, s_0_one_hot, y_0_one_hot, E_0, node_orig_id, _sp_stats = sample_out
            else:
                raise ValueError(f"Unexpected sample() return length: {len(sample_out)}")

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

        if args.save_samples:
            eval_pyg_data = maybe_keep_largest_cc_pyg(pyg_data, args.largest_cc)
            saved_eval_graphs.append(eval_pyg_data)
            nx_full = build_nx_graph_from_sample(
                dataset_name=dataset,
                E_0=E_0.cpu(),
                X_0_one_hot=X_0_one_hot.cpu(),
                s_0_one_hot=s_0_one_hot.cpu(),
                y_0_one_hot=y_0_one_hot.cpu() if y_0_one_hot is not None else y_0_one_hot,
                node_orig_id=node_orig_id,
            )
            saved_nx_full_graphs.append(nx_full)
            saved_nx_eval_graphs.append(build_nx_graph_from_pyg_data(dataset, eval_pyg_data))

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

    if args.save_samples and save_root is not None:
        tag = build_sample_tag(args, controller_enabled)
        pyg_path = save_root / f"{tag}.pyg.pt"
        pyg_full_path = save_root / f"{tag}.pyg_full.pt"
        nx_eval_path = save_root / f"{tag}.nx_eval.pkl"
        meta_path = save_root / f"{tag}.meta.json"

        torch.save(saved_eval_graphs, pyg_path)
        torch.save(saved_graphs, pyg_full_path)
        with nx_eval_path.open("wb") as f:
            pickle.dump(saved_nx_eval_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

        saved_files = {
            "pyg_pt": str(pyg_path),
            "pyg_full_pt": str(pyg_full_path),
            "nx_eval_pkl": str(nx_eval_path),
        }
        if args.save_full_graph:
            nx_full_path = save_root / f"{tag}.nx_full.pkl"
            with nx_full_path.open("wb") as f:
                pickle.dump(saved_nx_full_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved_files["nx_full_pkl"] = str(nx_full_path)

        meta = {
            "model_path": args.model_path,
            "sample_source_path": getattr(args, "sample_source_path", args.model_path),
            "dataset": dataset,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "device": str(device),
            "largest_cc": bool(args.largest_cc),
            "fair_score_sp": bool(args.fair_score_sp or args.sp_shift or controller_enabled),
            "fair_score_eta": guidance_eta,
            "fair_score_k": guidance_k,
            "controller_enabled": bool(controller_enabled),
            "controller_replay_playback": bool(controller_replay_playback),
            "controller_path": args.controller_path,
            "fair_label_attr": model_fair_label_attr,
            "fair_sensitive_attr": args.fair_sensitive_attr,
            "fair_sensitive_value": args.fair_sensitive_value,
            "fair_edge_sensitive_mode": args.fair_edge_sensitive_mode,
            "saved_files": saved_files,
            "pyg_pt_semantics": "final graph used for metrics (after optional largest-CC filtering)",
            "pyg_full_pt_semantics": "full final sampled PyG graphs before optional largest-CC filtering",
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[saved_samples] meta={meta_path} | eval_pyg={pyg_path} | full_pyg={pyg_full_path}")

    if not args.skip_internal_eval:
        evaluator.summary()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--num_samples", "--num_generation", dest="num_samples", type=int, default=10,
                        help="Number of samples to generate.")
    parser.add_argument("--sample_batch_size", type=int, default=32768,
                        help="FairWire edge-pair batch size inside one graph sample.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0, required=False)
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for graph sampling.")
    parser.add_argument("--run_name", type=str, default=None, help="Compatibility no-op.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Compatibility no-op.")
    parser.add_argument("--run_dir", type=str, default=None, help="Compatibility no-op.")
    parser.add_argument("--controller_path", "--fair_score_controller_path", dest="controller_path", type=str, default=None,
                        help="Optional stage-2 eta/k controller checkpoint.")
    parser.add_argument("--allow_checkpoint_search", type=str2bool, default=True,
                        help="If --model_path is missing, search local FairWire controller output dirs for a compatible checkpoint.")
    parser.add_argument("--fair_score_sp", action="store_true",
                        help="Apply EDGE-style score statistical-parity guidance during sampling.")
    parser.add_argument("--fair_score_eta", type=float, default=None,
                        help="Static eta for score-SP guidance, or initial eta when loading a controller.")
    parser.add_argument("--fair_score_k", type=float, default=None,
                        help="Static k for score-SP guidance, or initial k when loading a controller.")
    parser.add_argument("--fair_score_eta_scale", type=float, default=1.0,
                        help="EDGE-compatible eta scale; controller checkpoints use fair_score_eta as base eta.")
    parser.add_argument("--fair_score_guidance_normalize", type=eval, default=True,
                        help="Normalize score-SP guidance by mean absolute raw gradient.")
    parser.add_argument("--fair_score_apply_sample", type=eval, default=None, help="Compatibility no-op.")
    parser.add_argument("--sp_shift", action="store_true",
                        help="Alias for --fair_score_sp.")
    parser.add_argument("--sp_eta", type=float, default=0.0,
                        help="Alias for --fair_score_eta when --fair_score_eta is omitted.")
    parser.add_argument("--sp_k", type=float, default=0.15,
                        help="Alias for --fair_score_k when --fair_score_k is omitted.")
    parser.add_argument("--sp_eta_schedule", type=str, default="constant", choices=["constant", "early", "late"],
                        help="Schedule for static eta across reverse diffusion timesteps.")
    parser.add_argument("--sp_shift_clip", type=float, default=None,
                        help="Optional absolute clamp for each edge-existence logit shift.")
    parser.add_argument("--sp_report", action="store_true",
                        help="Print per-step fairness guidance diagnostics.")
    parser.add_argument("--fair_sensitive_attr", type=str, default="y", help="Compatibility no-op.")
    parser.add_argument("--fair_label_attr", type=str, default="y", help="Node label attr used by score-SP guidance.")
    parser.add_argument("--fair_sensitive_value", type=int, default=None, help="Compatibility no-op.")
    parser.add_argument("--fair_edge_sensitive_mode", type=str, default="either", help="Compatibility no-op.")
    parser.add_argument("--largest_cc", type=str2bool, default=True,
                        help="EDGE-compatible save_samples option: save *.pyg.pt after largest-CC filtering.")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save EDGE-style *.pyg.pt, *.pyg_full.pt, *.nx_eval.pkl, and *.meta.json files.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory used by --save_samples.")
    parser.add_argument("--save_full_graph", action="store_true",
                        help="With --save_samples, also save *.nx_full.pkl.")

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

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg

from diffusion.utils import add_parent_path

# Data
add_parent_path(level=1)
from datasets.data import get_data
from datasets.evaluator import compute_edge_score_sp_stats_from_components

# Model
from model import get_model


def resolve_fair_label_attr(eval_args, args):
    return (
        getattr(eval_args, 'fair_label_attr', None)
        or getattr(eval_args, 'fair_sensitive_attr', None)
        or getattr(args, 'fair_label_attr', None)
        or getattr(args, 'fair_sensitive_attr', None)
        or 'y'
    )


def to_networkx_graph(pyg_data, label_attr=None, sensitive_attr=None):
    """Convert a PyG Data object to an undirected NetworkX graph."""
    pyg_data = pyg_data.cpu()

    node_attrs = []
    # Keep parity with evaluate.py conversion. Note: x is intentionally not used for evaluator.
    target_attr = label_attr or sensitive_attr or 'y'
    for attr in ('y', 'orig_id', target_attr):
        if hasattr(pyg_data, attr) and getattr(pyg_data, attr) is not None and attr not in node_attrs:
            node_attrs.append(attr)

    if node_attrs:
        return pyg.utils.to_networkx(pyg_data, to_undirected=True, node_attrs=node_attrs)
    return pyg.utils.to_networkx(pyg_data, to_undirected=True)


def maybe_keep_largest_cc(g_gen, largest_cc=True):
    if largest_cc and g_gen.number_of_nodes() > 0:
        largest_cc_nodes = max(nx.connected_components(g_gen), key=len)
        return g_gen.subgraph(largest_cc_nodes).copy(), sorted(largest_cc_nodes)
    return g_gen.copy(), list(range(g_gen.number_of_nodes()))


def infer_safe_sample_batch_size(num_nodes: int, requested_total: int) -> int:
    if requested_total <= 0:
        return 1
    if num_nodes >= 2000:
        return 1
    if num_nodes >= 1000:
        return min(2, requested_total)
    if num_nodes >= 300:
        return min(4, requested_total)
    return min(8, requested_total)


def resolve_node_labels_from_pyg(pyg_data, label_attr=None):
    if label_attr is None or not hasattr(pyg_data, label_attr):
        return None
    values = getattr(pyg_data, label_attr)
    if values is None:
        return None
    if torch.is_tensor(values):
        values = values.detach().cpu().reshape(-1)
    else:
        values = torch.as_tensor(values).reshape(-1)
    return values


def to_cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu_tree(v) for v in obj)
    return obj


def sanitize_tag_value(value):
    if value is None:
        return 'none'
    text = str(value)
    for src, dst in (
        ('/', '_'),
        ('\\', '_'),
        (' ', ''),
        (':', '_'),
        ('.', 'p'),
        ('-', 'm'),
    ):
        text = text.replace(src, dst)
    return text


def build_sample_tag(eval_args, args):
    return (
        f"ckpt{eval_args.checkpoint}"
        f"_n{eval_args.num_samples}"
        f"_seed{eval_args.seed}"
        f"_eta{sanitize_tag_value(getattr(args, 'fair_score_eta', None))}"
        f"_k{sanitize_tag_value(getattr(args, 'fair_score_k', None))}"
        f"_attr{sanitize_tag_value(getattr(args, 'fair_sensitive_attr', None))}"
        f"_val{sanitize_tag_value(getattr(args, 'fair_sensitive_value', None))}"
        f"_mode{sanitize_tag_value(getattr(args, 'fair_edge_sensitive_mode', None))}"
        f"_lcc{int(bool(eval_args.largest_cc))}"
    )


def extract_pyg_subgraph_by_nodes(data, kept_nodes):
    """
    Build a PyG Data subgraph that matches the exact graph used for evaluation.
    This makes *.pyg.pt consistent with evaluate.py's final graph after optional LCC filtering.
    """
    data = data.cpu()
    kept_nodes = torch.as_tensor(kept_nodes, dtype=torch.long)
    kept_nodes, _ = torch.sort(kept_nodes.unique())

    if kept_nodes.numel() == data.num_nodes:
        return data

    old_to_new = -torch.ones(data.num_nodes, dtype=torch.long)
    old_to_new[kept_nodes] = torch.arange(kept_nodes.numel(), dtype=torch.long)

    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    edge_mask = (old_to_new[row] >= 0) & (old_to_new[col] >= 0)
    new_edge_index = torch.stack([
        old_to_new[row[edge_mask]],
        old_to_new[col[edge_mask]],
    ], dim=0)

    out = data.__class__()
    out.edge_index = new_edge_index
    out.num_nodes = int(kept_nodes.numel())

    for key, value in data:
        if key == 'edge_index':
            continue
        if key == 'num_nodes':
            continue
        if torch.is_tensor(value):
            if value.dim() > 0 and value.size(0) == data.num_nodes:
                out[key] = value[kept_nodes]
            elif value.dim() > 0 and value.size(0) == data.edge_index.size(1):
                out[key] = value[edge_mask]
            else:
                out[key] = value
        else:
            out[key] = value

    return out


###########
## Setup ##
###########
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='2023-05-29_18-29-35')
    parser.add_argument('--dataset', type=str, default='polblogs')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument(
        '--sample_batch_size',
        type=int,
        default=None,
        help='graphs to sample per micro-batch; default: conservative auto choice to avoid OOM',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=int, default=5500)
    parser.add_argument(
        '--run_dir',
        type=str,
        default=None,
        help='optional explicit trained-run directory; defaults to ./wandb/{dataset}/multinomial_diffusion/multistep/{run_name}',
    )
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--largest_cc',
        type=eval,
        default=True,
        help='keep only the largest connected component before evaluation (matches original evaluate.py)',
    )
    parser.add_argument(
        '--return_edge_deltas',
        action='store_true',
        help='also materialize diffusion edge-delta traces; off by default because evaluation does not use them',
    )
    parser.add_argument('--fair_score_sp', action='store_true')
    parser.add_argument('--fair_score_eta', type=float, default=None)
    parser.add_argument('--fair_score_k', type=float, default=None)
    parser.add_argument('--fair_score_apply_sample', type=eval, default=None)
    parser.add_argument('--fair_score_guidance_normalize', type=eval, default=None)
    parser.add_argument('--fair_label_attr', type=str, default=None)
    parser.add_argument('--fair_sensitive_attr', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--fair_sensitive_value', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        '--fair_edge_sensitive_mode',
        type=str,
        default=None,
        choices=['either', 'both'],
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument(
        '--save_full_graph',
        action='store_true',
        help='also save full generated NetworkX graphs before any largest-CC trimming; full PyG graphs are always saved as *.pyg_full.pt when --save_samples is used',
    )
    return parser


def run_evaluate(eval_args):
    torch.manual_seed(eval_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_args.seed)

    if getattr(eval_args, 'run_dir', None) is not None:
        log_dir = str(Path(eval_args.run_dir).expanduser().resolve())
    else:
        log_dir = f'./wandb/{eval_args.dataset}/multinomial_diffusion/multistep/{eval_args.run_name}'
    path_args = f'{log_dir}/args.pickle'
    path_check = f'{log_dir}/check/checkpoint_{eval_args.checkpoint-1}.pt'

    with open(path_args, 'rb') as f:
        args = pickle.load(f)

    args.device = eval_args.device
    if eval_args.fair_sensitive_attr is not None:
        args.fair_sensitive_attr = eval_args.fair_sensitive_attr
    if eval_args.fair_sensitive_value is not None:
        args.fair_sensitive_value = eval_args.fair_sensitive_value
    if eval_args.fair_edge_sensitive_mode is not None:
        args.fair_edge_sensitive_mode = eval_args.fair_edge_sensitive_mode
    args.fair_label_attr = resolve_fair_label_attr(eval_args, args)
    if eval_args.fair_score_sp:
        args.fair_score_sp = True
    if eval_args.fair_score_eta is not None:
        args.fair_score_eta = eval_args.fair_score_eta
    if eval_args.fair_score_k is not None:
        args.fair_score_k = eval_args.fair_score_k
    if eval_args.fair_score_apply_sample is not None:
        args.fair_score_apply_sample = eval_args.fair_score_apply_sample
    if eval_args.fair_score_guidance_normalize is not None:
        args.fair_score_guidance_normalize = eval_args.fair_score_guidance_normalize

    (
        train_loader,
        eval_loader,
        test_loader,
        num_node_feat,
        num_node_classes,
        num_edge_classes,
        max_degree,
        augmented_feature_dict,
        initial_graph_sampler,
        eval_evaluator,
        test_evaluator,
        monitoring_statistics,
    ) = get_data(args)

    model = get_model(args, initial_graph_sampler=initial_graph_sampler)
    checkpoint = torch.load(path_check, map_location=args.device)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model = model.to(args.device)
    model.eval()

    reference_num_nodes = test_evaluator.reference.number_of_nodes()
    sample_batch_size = eval_args.sample_batch_size
    if sample_batch_size is None:
        sample_batch_size = infer_safe_sample_batch_size(reference_num_nodes, eval_args.num_samples)
    sample_batch_size = max(1, min(sample_batch_size, eval_args.num_samples))

    save_root = None
    store_generated_graphs = bool(eval_args.save_samples or getattr(eval_args, 'collect_generated_graphs', False))
    saved_pyg_eval_datas = []
    saved_pyg_raw_datas = []
    saved_full_nxgraphs = []
    saved_deltas = []
    if eval_args.save_samples:
        save_root = Path(eval_args.save_dir) if eval_args.save_dir is not None else Path(log_dir) / 'generated_samples'
        save_root.mkdir(parents=True, exist_ok=True)

    # Sample in micro-batches to reduce peak GPU memory.
    # Evaluation metrics remain the original full evaluator output.
    generated_nxgraphs = []
    generated_soft_fair_stats = []
    num_done = 0
    while num_done < eval_args.num_samples:
        cur_batch = min(sample_batch_size, eval_args.num_samples - num_done)
        return_soft_scores = bool(getattr(args, 'fair_score_sp', False))

        sample_out = model.sample(
            cur_batch,
            return_edge_deltas=eval_args.return_edge_deltas,
            return_soft=return_soft_scores,
            keep_zeros=False,
            delta_as_sparse_adj=False,
        )

        if eval_args.return_edge_deltas:
            sampled_pygraph, cur_deltas = sample_out
        else:
            sampled_pygraph = sample_out
            cur_deltas = None

        sampled_pygraph = sampled_pygraph.cpu()
        pyg_datas = sampled_pygraph.to_data_list()
        batch_full_edge_score_prob = getattr(sampled_pygraph, 'full_edge_score_prob', None)
        if batch_full_edge_score_prob is not None:
            batch_full_edge_score_prob = batch_full_edge_score_prob.cpu()
            edge_offsets = [0]
            for count in sampled_pygraph.edges_per_graph.cpu().tolist():
                edge_offsets.append(edge_offsets[-1] + int(count))
        else:
            edge_offsets = None

        for graph_idx, pyg_data in enumerate(pyg_datas):
            pyg_data = pyg_data.cpu()
            g_full = to_networkx_graph(
                pyg_data,
                label_attr=getattr(args, 'fair_label_attr', 'y'),
            )
            g_eval, kept_nodes = maybe_keep_largest_cc(g_full, largest_cc=eval_args.largest_cc)
            generated_nxgraphs.append(g_eval)

            if batch_full_edge_score_prob is not None and hasattr(pyg_data, 'full_edge_index'):
                edge_start = edge_offsets[graph_idx]
                edge_end = edge_offsets[graph_idx + 1]
                node_labels = resolve_node_labels_from_pyg(
                    pyg_data,
                    label_attr=getattr(args, 'fair_label_attr', 'y'),
                )
                soft_fair_stats = compute_edge_score_sp_stats_from_components(
                    full_edge_index=pyg_data.full_edge_index,
                    edge_scores=batch_full_edge_score_prob[edge_start:edge_end],
                    node_labels=node_labels,
                    kept_nodes=kept_nodes,
                )
                if soft_fair_stats is not None:
                    generated_soft_fair_stats.append(soft_fair_stats)

            if store_generated_graphs:
                saved_pyg_raw_datas.append(pyg_data)
                saved_pyg_eval_datas.append(extract_pyg_subgraph_by_nodes(pyg_data, kept_nodes))
            if eval_args.save_samples and eval_args.save_full_graph:
                saved_full_nxgraphs.append(g_full)

        if store_generated_graphs and eval_args.return_edge_deltas and cur_deltas is not None:
            cur_deltas = to_cpu_tree(cur_deltas)
            if isinstance(cur_deltas, list):
                saved_deltas.extend(cur_deltas)
            else:
                saved_deltas.append(cur_deltas)

        num_done += cur_batch

        del sample_out
        del sampled_pygraph
        del pyg_datas
        if cur_deltas is not None:
            del cur_deltas
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = test_evaluator.evaluate(generated_nxgraphs)
    if generated_soft_fair_stats:
        for key in generated_soft_fair_stats[0].keys():
            metrics[f'value/{key}'] = float(np.nanmean([stat[key] for stat in generated_soft_fair_stats]))
        if 'ref/fair_edge_sp_gap' in metrics:
            metrics['ref/fair_edge_score_sp_gap'] = float(metrics['ref/fair_edge_sp_gap'])
            metrics['ref/fair_edge_score_sp_abs_gap'] = float(metrics['ref/fair_edge_sp_abs_gap'])

    if eval_args.save_samples:
        tag = build_sample_tag(eval_args, args)
        pyg_path = save_root / f'{tag}.pyg.pt'          # exact graphs used for metrics
        pyg_full_path = save_root / f'{tag}.pyg_full.pt'  # full final samples before LCC filtering
        nx_eval_path = save_root / f'{tag}.nx_eval.pkl'
        metrics_path = save_root / f'{tag}.metrics.json'
        meta_path = save_root / f'{tag}.meta.json'

        torch.save(saved_pyg_eval_datas, pyg_path)
        torch.save(saved_pyg_raw_datas, pyg_full_path)

        with nx_eval_path.open('wb') as f:
            pickle.dump(generated_nxgraphs, f, protocol=pickle.HIGHEST_PROTOCOL)

        saved_files = {
            'pyg_pt': str(pyg_path),
            'pyg_full_pt': str(pyg_full_path),
            'nx_eval_pkl': str(nx_eval_path),
        }

        if eval_args.save_full_graph:
            nx_full_path = save_root / f'{tag}.nx_full.pkl'
            with nx_full_path.open('wb') as f:
                pickle.dump(saved_full_nxgraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved_files['nx_full_pkl'] = str(nx_full_path)

        if eval_args.return_edge_deltas:
            deltas_path = save_root / f'{tag}.deltas.pt'
            torch.save(saved_deltas, deltas_path)
            saved_files['deltas_pt'] = str(deltas_path)

        with metrics_path.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        saved_files['metrics_json'] = str(metrics_path)

        meta = {
            'run_name': eval_args.run_name,
            'dataset': eval_args.dataset,
            'checkpoint': eval_args.checkpoint,
            'seed': eval_args.seed,
            'num_samples': eval_args.num_samples,
            'sample_batch_size': sample_batch_size,
            'device': eval_args.device,
            'largest_cc': bool(eval_args.largest_cc),
            'return_edge_deltas': bool(eval_args.return_edge_deltas),
            'fair_score_sp': bool(getattr(args, 'fair_score_sp', False)),
            'fair_score_eta': getattr(args, 'fair_score_eta', None),
            'fair_score_k': getattr(args, 'fair_score_k', None),
            'fair_score_apply_sample': getattr(args, 'fair_score_apply_sample', None),
            'fair_score_guidance_normalize': getattr(args, 'fair_score_guidance_normalize', False),
            'fair_label_attr': getattr(args, 'fair_label_attr', None),
            'fair_sensitive_attr': getattr(args, 'fair_sensitive_attr', None),
            'fair_sensitive_value': getattr(args, 'fair_sensitive_value', None),
            'fair_edge_sensitive_mode': getattr(args, 'fair_edge_sensitive_mode', None),
            'saved_files': saved_files,
            'pyg_pt_semantics': 'final graph used for metrics (after optional largest-CC filtering)',
            'pyg_full_pt_semantics': 'full final sampled PyG graphs before optional largest-CC filtering (all generated nodes kept)',
            'save_full_graph_semantics': 'controls saving extra *.nx_full.pkl copies; full PyG graphs are already saved in *.pyg_full.pt',
        }
        with meta_path.open('w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        msg = f"[saved_samples] meta={meta_path} | eval_pyg={saved_files['pyg_pt']} | full_pyg={saved_files['pyg_full_pt']}"
        if 'nx_full_pkl' in saved_files:
            msg += f" | full_nx={saved_files['nx_full_pkl']}"
        print(msg, file=sys.stderr)

    return {
        'metrics': metrics,
        'pyg_eval_datas': saved_pyg_eval_datas,
        'pyg_full_datas': saved_pyg_raw_datas,
        'generated_nxgraphs': generated_nxgraphs,
        'sample_batch_size': sample_batch_size,
        'log_dir': log_dir,
    }


def main(argv=None):
    parser = build_parser()
    eval_args = parser.parse_args(argv)
    result = run_evaluate(eval_args)
    print(result['metrics'])


if __name__ == '__main__':
    main()

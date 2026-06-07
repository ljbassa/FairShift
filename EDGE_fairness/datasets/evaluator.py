import dgl
import numpy as np
import networkx as nx
import random
from eval_utils.graph_statistics import compute_graph_statistics, linkpred_auc_from_pairs
from eval_utils.evaluation.evaluator import Evaluator
from eval_utils.evaluation.graph_structure_evaluation import MMDEval, NSPDKEvaluation


def _resolve_node_labels(g, label_attr='y'):
    node_labels = {}
    for node, attrs in g.nodes(data=True):
        if label_attr not in attrs:
            return None
        value = attrs[label_attr]
        if hasattr(value, "item"):
            value = value.item()
        node_labels[node] = value
    return node_labels


def compute_edge_sp_stats(g, label_attr='y'):
    node_labels = _resolve_node_labels(
        g,
        label_attr=label_attr,
    )
    if node_labels is None:
        return None

    num_nodes = g.number_of_nodes()
    label_counts = {}
    for label in node_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    num_edges_same = 0
    for u, v in g.edges():
        if node_labels[u] == node_labels[v]:
            num_edges_same += 1

    same_label_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())
    total_pairs = num_nodes * (num_nodes - 1) // 2
    diff_label_edges = g.number_of_edges() - num_edges_same
    diff_label_pairs = total_pairs - same_label_pairs

    rate_sensitive = float(num_edges_same / same_label_pairs) if same_label_pairs > 0 else np.nan
    rate_nonsensitive = float(diff_label_edges / diff_label_pairs) if diff_label_pairs > 0 else np.nan
    gap = rate_sensitive - rate_nonsensitive if np.isfinite(rate_sensitive) and np.isfinite(rate_nonsensitive) else np.nan

    return {
        'fair_edge_sensitive_rate': rate_sensitive,
        'fair_edge_nonsensitive_rate': rate_nonsensitive,
        'fair_edge_sp_gap': gap,
        'fair_edge_sp_abs_gap': abs(gap) if np.isfinite(gap) else np.nan,
    }


def compute_edge_score_sp_stats_from_components(
    full_edge_index,
    edge_scores,
    node_labels,
    kept_nodes=None,
):
    if full_edge_index is None or edge_scores is None or node_labels is None:
        return None

    if hasattr(full_edge_index, "detach"):
        full_edge_index = full_edge_index.detach().cpu().numpy()
    else:
        full_edge_index = np.asarray(full_edge_index)

    if hasattr(edge_scores, "detach"):
        edge_scores = edge_scores.detach().cpu().numpy()
    else:
        edge_scores = np.asarray(edge_scores)

    if hasattr(node_labels, "detach"):
        node_labels = node_labels.detach().cpu().numpy()
    else:
        node_labels = np.asarray(node_labels)

    full_edge_index = np.asarray(full_edge_index, dtype=np.int64)
    edge_scores = np.asarray(edge_scores, dtype=np.float64).reshape(-1)
    node_labels = np.asarray(node_labels).reshape(-1)

    if full_edge_index.ndim != 2 or full_edge_index.shape[0] != 2:
        raise ValueError('full_edge_index must have shape [2, E]')
    if full_edge_index.shape[1] != edge_scores.shape[0]:
        raise ValueError('full_edge_index and edge_scores must describe the same number of edges')

    if kept_nodes is not None:
        kept_nodes = np.asarray(list(kept_nodes), dtype=np.int64)
        kept_mask = np.zeros(node_labels.shape[0], dtype=bool)
        kept_mask[kept_nodes] = True
        edge_keep = kept_mask[full_edge_index[0]] & kept_mask[full_edge_index[1]]
        full_edge_index = full_edge_index[:, edge_keep]
        edge_scores = edge_scores[edge_keep]

    if edge_scores.size == 0:
        return {
            'fair_edge_score_sensitive_rate': np.nan,
            'fair_edge_score_nonsensitive_rate': np.nan,
            'fair_edge_score_sp_gap': np.nan,
            'fair_edge_score_sp_abs_gap': np.nan,
        }

    src = full_edge_index[0]
    dst = full_edge_index[1]
    sensitive_mask = node_labels[src] == node_labels[dst]
    nonsensitive_mask = ~sensitive_mask

    rate_sensitive = float(edge_scores[sensitive_mask].mean()) if sensitive_mask.any() else np.nan
    rate_nonsensitive = float(edge_scores[nonsensitive_mask].mean()) if nonsensitive_mask.any() else np.nan
    gap = rate_sensitive - rate_nonsensitive if np.isfinite(rate_sensitive) and np.isfinite(rate_nonsensitive) else np.nan

    return {
        'fair_edge_score_sensitive_rate': rate_sensitive,
        'fair_edge_score_nonsensitive_rate': rate_nonsensitive,
        'fair_edge_score_sp_gap': gap,
        'fair_edge_score_sp_abs_gap': abs(gap) if np.isfinite(gap) else np.nan,
    }

class NetworkEvaluator:
    def __init__(
        self,
        reference_nx_graph,
        lp_max_pos_edges=50000,
        lp_neg_ratio=1.0,
        lp_seed=0,
        fair_label_attr='y',
        fair_sensitive_value=None,
        fair_edge_sensitive_mode='either',
    ):
        self.reference = reference_nx_graph
        self.reference_stats = compute_graph_statistics(nx.to_scipy_sparse_array(self.reference))
        self.fair_label_attr = fair_label_attr
        self.reference_fair_stats = compute_edge_sp_stats(
            self.reference,
            label_attr=self.fair_label_attr,
        )

        # --- Link prediction AUC pairs (fixed once for speed & comparability) ---
        self.lp_edge_pairs, self.lp_labels = self._build_linkpred_eval_pairs(
            self.reference,
            max_pos_edges=lp_max_pos_edges,
            neg_ratio=lp_neg_ratio,
            seed=lp_seed,
        )

        # In our "generated adjacency as score" setup, reference vs reference yields AUC=1.0
        self.reference_lp_auc = 1.0

    @staticmethod
    def _build_linkpred_eval_pairs(g_ref, max_pos_edges=50000, neg_ratio=1.0, seed=0):
        """
        Build (edge_pairs, labels) from reference graph:
        - positives: existing edges
        - negatives: sampled non-edges
        This is used for "generated adjacency as score" AUC evaluation.
        """
        rng = random.Random(seed)
        nodes = list(g_ref.nodes())

        # normalize undirected edge representation
        ref_edge_set = set()
        pos_edges_all = []
        for u, v in g_ref.edges():
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in ref_edge_set:
                continue
            ref_edge_set.add(e)
            pos_edges_all.append(e)

        # cap positives for scalability
        if max_pos_edges is not None and len(pos_edges_all) > max_pos_edges:
            pos_edges = rng.sample(pos_edges_all, max_pos_edges)
        else:
            pos_edges = pos_edges_all

        num_neg = int(len(pos_edges) * float(neg_ratio))

        # sample negative non-edges
        neg_edges = []
        neg_set = set()
        # rejection sampling (OK for sparse graphs like cora/polblogs)
        while len(neg_edges) < num_neg:
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in ref_edge_set or e in neg_set:
                continue
            neg_set.add(e)
            neg_edges.append(e)

        edge_pairs = pos_edges + neg_edges
        labels = np.concatenate([
            np.ones(len(pos_edges), dtype=np.int32),
            np.zeros(len(neg_edges), dtype=np.int32),
        ])
        return edge_pairs, labels


    def evaluate(self, target_nx_graphs):
        metric_per_graphs = [compute_graph_statistics(nx.to_scipy_sparse_array(target_nx_graph)) for target_nx_graph in target_nx_graphs]
        # merge metrics and compute mean
        metrics = {f'nmae/{k}': abs((np.mean([m[k] for m in metric_per_graphs]) - self.reference_stats[k])/self.reference_stats[k]) for k in metric_per_graphs[0].keys()}
        metrics.update({f'value/{k}': np.mean([m[k] for m in metric_per_graphs]) for k in metric_per_graphs[0].keys()})
        
        # --- add link prediction AUC ---
        aucs = [
            linkpred_auc_from_pairs(g, self.lp_edge_pairs, self.lp_labels)
            for g in target_nx_graphs
        ]
        auc_mean = float(np.nanmean(aucs))

        metrics['value/linkpred_auc'] = auc_mean
        metrics['nmae/linkpred_auc'] = abs((auc_mean - self.reference_lp_auc) / self.reference_lp_auc)

        fair_stats = [
            compute_edge_sp_stats(
                g,
                label_attr=self.fair_label_attr,
            )
            for g in target_nx_graphs
        ]
        fair_stats = [stat for stat in fair_stats if stat is not None]
        if fair_stats:
            for key in fair_stats[0].keys():
                metrics[f'value/{key}'] = float(np.nanmean([stat[key] for stat in fair_stats]))
            if self.reference_fair_stats is not None:
                metrics['ref/fair_edge_sp_gap'] = float(self.reference_fair_stats['fair_edge_sp_gap'])
                metrics['ref/fair_edge_sp_abs_gap'] = float(self.reference_fair_stats['fair_edge_sp_abs_gap'])

        return metrics

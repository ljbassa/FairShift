import torch
import os
import pickle
import gc
from diffusion.loss import elbo_bpd
from diffusion.utils import add_parent_path
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np


add_parent_path(level=2)
from diffusion.experiment import DiffusionExperiment
from diffusion.experiment import add_exp_args as add_exp_args_parent

# for calculate edge overlap
import numpy as np
from eval_utils.graph_statistics import linkpred_auc_from_pairs


def _undirected_edge_set(g_nx):
    """networkx Graph -> undirected edge set {(u,v)} with u<v, no self-loops"""
    s = set()
    for u, v in g_nx.edges():
        if u == v:
            continue
        if u < v:
            s.add((u, v))
        else:
            s.add((v, u))
    return s


def edge_match_stats(g_ref, g_gen):
    """
    Return edge overlap stats between reference and generated graphs.
    Percentages are relative to |E_ref|, as you requested.
    """
    ref_edges = _undirected_edge_set(g_ref)
    gen_edges = _undirected_edge_set(g_gen)

    tp = len(ref_edges & gen_edges)     # ref edge를 맞춘 수
    fn = len(ref_edges - gen_edges)     # ref에 있는데 gen이 못 만든 수
    fp = len(gen_edges - ref_edges)     # ref에 없는데 gen이 만든 수

    n_ref = len(ref_edges)
    n_gen = len(gen_edges)

    def pct(x):
        return (float(x) / float(n_ref) * 100.0) if n_ref > 0 else 0.0

    precision = tp / max(1, n_gen)
    recall = tp / max(1, n_ref)
    jaccard = tp / max(1, (n_ref + n_gen - tp))

    return {
        "ref_num_edges": int(n_ref),
        "gen_num_edges": [int(n_gen), f"{pct(n_gen):.3f}%"],
        "tp_matched_ref_edges": [int(tp), f"{pct(tp):.3f}%"],
        "fn_missed_ref_edges": [int(fn), f"{pct(fn):.3f}%"],
        "fp_extra_gen_edges": [int(fp), f"{pct(fp):.3f}%"],
        "total_wrong_edges": [int(fn + fp), f"{pct(fn + fp):.3f}%"],
        "precision": float(precision),
        "recall": float(recall),
        "jaccard": float(jaccard),
    }


def _print_edge_stats(title, best_epoch, auc_mean, best_graph_auc, stats: dict):
    print("\n" + "=" * 80)
    print(title)
    print(f"Best epoch: {best_epoch}")
    print(f"eval/value/linkpred_auc (mean): {auc_mean:.6f}")
    print(f"best graph auc (within that eval batch): {best_graph_auc:.6f}")
    print("-" * 80)
    print(f"ref_num_edges: {stats['ref_num_edges']}")
    print(f"gen_num_edges: {stats['gen_num_edges'][0]} ({stats['gen_num_edges'][1]})")
    print(f"tp_matched_ref_edges: {stats['tp_matched_ref_edges'][0]} ({stats['tp_matched_ref_edges'][1]})")
    print(f"fn_missed_ref_edges: {stats['fn_missed_ref_edges'][0]} ({stats['fn_missed_ref_edges'][1]})")
    print(f"fp_extra_gen_edges: {stats['fp_extra_gen_edges'][0]} ({stats['fp_extra_gen_edges'][1]})")
    print(f"total_wrong_edges: {stats['total_wrong_edges'][0]} ({stats['total_wrong_edges'][1]})")
    print("-" * 80)
    print(f"precision: {stats['precision']:.6f}")
    print(f"recall: {stats['recall']:.6f}")
    print(f"jaccard: {stats['jaccard']:.6f}")
    print("=" * 80 + "\n")

def _get_y_int(v):
    # torch scalar / numpy scalar / python int 모두 대응
    if v is None:
        return None
    if hasattr(v, "item"):
        return int(v.item())
    return int(v)


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


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--num_generation', type=int, default=64)
    parser.add_argument(
        '--eval_num_generation',
        type=int,
        default=None,
        help='override num_generation during train-time eval only',
    )
    parser.add_argument(
        '--test_num_generation',
        type=int,
        default=None,
        help='override num_generation during test-on-best only',
    )
    parser.add_argument(
        '--sample_batch_size',
        type=int,
        default=None,
        help='graphs to sample per micro-batch during eval/test; lowers peak GPU memory',
    )
    parser.add_argument(
        '--cpu_offload_generated',
        type=eval,
        default=True,
        help='move sampled PyG batches to CPU before conversion/evaluation',
    )
    parser.add_argument(
        '--empty_cache_after_sampling',
        type=eval,
        default=True,
        help='call torch.cuda.empty_cache() after each eval/test sample micro-batch',
    )
    parser.add_argument('--return_edge_deltas', action='store_true',
            help='collect per-step edge deltas during sampling (active diffusion)')
    parser.add_argument('--return_soft', action='store_true',
            help='use soft delta (p(edge=1|x_t)-a_prev) instead of hard -1/0/+1')
    parser.add_argument('--keep_zeros', action='store_true',
            help='keep zero deltas too (default: only changed edges)')
    parser.add_argument('--delta_as_sparse_adj', action='store_true',
            help='also return sparse delta adjacency per step')


class GraphExperiment(DiffusionExperiment):

    def _generation_count(self, phase):
        if phase == "eval" and self.args.eval_num_generation is not None:
            return max(1, int(self.args.eval_num_generation))
        if phase == "test" and self.args.test_num_generation is not None:
            return max(1, int(self.args.test_num_generation))
        return max(1, int(self.args.num_generation))

    def _sample_batch_size(self, requested_total):
        configured = getattr(self.args, "sample_batch_size", None)
        if configured is None:
            return requested_total
        return max(1, min(int(configured), int(requested_total)))

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available() and str(self.args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def _generated_to_networkx(self, pyg_data):
        node_attrs = []
        if hasattr(pyg_data, "y") and pyg_data.y is not None:
            node_attrs.append("y")
        if hasattr(pyg_data, "orig_id"):
            node_attrs.append("orig_id")
        label_attr = getattr(self.args, "fair_label_attr", None) or "y"
        if hasattr(pyg_data, label_attr) and getattr(pyg_data, label_attr) is not None and label_attr not in node_attrs:
            node_attrs.append(label_attr)

        if node_attrs:
            return pyg.utils.to_networkx(pyg_data, to_undirected=True, node_attrs=node_attrs)
        return pyg.utils.to_networkx(pyg_data, to_undirected=True)

    def _sample_generated_graphs(self, phase):
        total_num_generation = self._generation_count(phase)
        sample_batch_size = self._sample_batch_size(total_num_generation)

        generated_graphs = []
        deltas = [] if self.args.return_edge_deltas else None
        num_done = 0

        while num_done < total_num_generation:
            cur_batch = min(sample_batch_size, total_num_generation - num_done)

            if self.args.return_edge_deltas:
                generated_pyg_datas, cur_deltas = self.model.sample(
                    cur_batch,
                    return_edge_deltas=True,
                    return_soft=self.args.return_soft,
                    keep_zeros=self.args.keep_zeros,
                    delta_as_sparse_adj=self.args.delta_as_sparse_adj,
                )
                cur_deltas = to_cpu_tree(cur_deltas)
                if isinstance(cur_deltas, list):
                    deltas.extend(cur_deltas)
                else:
                    deltas.append(cur_deltas)
            else:
                generated_pyg_datas = self.model.sample(cur_batch)
                cur_deltas = None

            if getattr(self.args, "cpu_offload_generated", True):
                generated_pyg_datas = generated_pyg_datas.cpu()

            pyg_data_list = generated_pyg_datas.to_data_list()
            for pyg_data in pyg_data_list:
                generated_graphs.append(self._generated_to_networkx(pyg_data))

            del pyg_data_list
            del generated_pyg_datas
            del cur_deltas
            num_done += cur_batch

            if getattr(self.args, "empty_cache_after_sampling", True):
                self._cleanup_cuda()

        return generated_graphs, deltas

    def run(self):
        # 기존 학습 루프 수행
        super(GraphExperiment, self).run()

        # 학습 종료 후 딱 1회 출력
        if hasattr(self, "_best_mean_auc_stats") and self._best_mean_auc_stats is not None:
            _print_edge_stats(
                title="[FINAL REPORT] Best mean linkpred_auc snapshot edge stats",
                best_epoch=self._best_mean_auc_epoch,
                auc_mean=self._best_mean_auc,
                best_graph_auc=self._best_mean_auc_best_graph_auc,
                stats=self._best_mean_auc_stats,
            )

            # (선택) wandb에 '표'로 1회만 올리고 싶다면
            try:
                import wandb
                table = wandb.Table(columns=["metric", "value"])
                s = self._best_mean_auc_stats
                rows = [
                    ("best_epoch", self._best_mean_auc_epoch),
                    ("auc_mean", self._best_mean_auc),
                    ("best_graph_auc", self._best_mean_auc_best_graph_auc),
                    ("ref_num_edges", s["ref_num_edges"]),
                    ("gen_num_edges", f"{s['gen_num_edges'][0]} ({s['gen_num_edges'][1]})"),
                    ("tp_matched_ref_edges", f"{s['tp_matched_ref_edges'][0]} ({s['tp_matched_ref_edges'][1]})"),
                    ("fn_missed_ref_edges", f"{s['fn_missed_ref_edges'][0]} ({s['fn_missed_ref_edges'][1]})"),
                    ("fp_extra_gen_edges", f"{s['fp_extra_gen_edges'][0]} ({s['fp_extra_gen_edges'][1]})"),
                    ("total_wrong_edges", f"{s['total_wrong_edges'][0]} ({s['total_wrong_edges'][1]})"),
                    ("precision", s["precision"]),
                    ("recall", s["recall"]),
                    ("jaccard", s["jaccard"]),
                ]
                for k, v in rows:
                    table.add_data(k, v)
                wandb.log({"final/best_edge_stats": table}, step=self._best_mean_auc_epoch)
            except Exception as e:
                print("[WARN] wandb.Table logging failed:", e)
        else:
            print("\n[FINAL REPORT] No best_mean_auc stats found (did eval run at least once?)\n")
            
    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        data_count = 0
        for pyg_data in self.train_loader:
            self.optimizer.zero_grad()
            if self.args.parallel != 'dp':
                pyg_data = pyg_data.to(self.args.device)
            # pyg_data.num_entries = self.model._calc_num_entries(pyg_data)
            loss = elbo_bpd(self.model, pyg_data)
            loss.backward()
            
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()

            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs
            loss_count += pyg_data.num_graphs
            data_count += pyg_data.num_graphs#pyg_data.num_graphs
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, data_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
            # self.model.complex_data = None
        if self.scheduler_epoch: self.scheduler_epoch.step()
        train_dict = {'bpd': loss_sum / loss_count, 'lr': self.optimizer.param_groups[0]['lr']}
        
        return train_dict


    def eval_fn(self, epoch):
        self.model.eval()
        eval_dict = {}
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            data_count = 0

            for pyg_data in self.eval_loader:
                if self.args.parallel != 'dp':
                    pyg_data = pyg_data.to(self.args.device)
                loss = elbo_bpd(self.model, pyg_data) 
                loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs
                loss_count += pyg_data.num_graphs 
                data_count += pyg_data.num_graphs

            print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, data_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')            
            eval_dict['bpd'] = loss_sum/loss_count
            generated_graphs, deltas = self._sample_generated_graphs("eval")


            w = 8 if self._generation_count("eval") >= 64 else 2
            fig, axes = plt.subplots(w, w, figsize=(17,17))
            for i, g_gen in enumerate(generated_graphs[:w**2]):
                nx.draw(g_gen, ax=axes[i%w][i//w], node_size=30)

            os.makedirs(os.path.join(self.log_path, "eval"), exist_ok=True)
            plt.savefig(os.path.join(self.log_path, f"eval/sample{epoch}.png"))
            plt.close()

            # statistics evaluation
            metrics = self.eval_evaluator.evaluate(generated_graphs)
            eval_dict.update(metrics)

            auc_mean = float(metrics.get("value/linkpred_auc", float("nan")))

            if not hasattr(self, "_best_mean_auc"):
                self._best_mean_auc = -1.0
                self._best_mean_auc_epoch = -1
                self._best_mean_auc_stats = None
                self._best_mean_auc_best_graph_auc = -1.0

            if np.isfinite(auc_mean) and auc_mean > self._best_mean_auc:
                lp_pairs = self.eval_evaluator.lp_edge_pairs
                lp_labels = self.eval_evaluator.lp_labels
                aucs = [linkpred_auc_from_pairs(g, lp_pairs, lp_labels) for g in generated_graphs]

                best_i = int(np.nanargmax(aucs))
                best_graph_auc = float(aucs[best_i])
                best_g = generated_graphs[best_i]

                g_ref = self.eval_evaluator.reference
                stats = edge_match_stats(g_ref, best_g)

                self._best_mean_auc = auc_mean
                self._best_mean_auc_epoch = int(epoch + 1)
                self._best_mean_auc_stats = stats
                self._best_mean_auc_best_graph_auc = best_graph_auc

            auc = eval_dict.get("value/linkpred_auc", None)
            if auc is not None:
                if not hasattr(self, "_best_lp_auc"):
                    self._best_lp_auc = -1.0

                if auc > self._best_lp_auc:
                    self._best_lp_auc = float(auc)
                    out_dir = os.path.join(self.log_path, "best_graph_dump")
                    os.makedirs(out_dir, exist_ok=True)

                    out_pkl = os.path.join(out_dir, "best_generated_graphs_with_y.pkl")
                    with open(out_pkl, "wb") as f:
                        pickle.dump(generated_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

                    out_meta = os.path.join(out_dir, "best_meta.json")
                    with open(out_meta, "w") as f:
                        json.dump({"epoch": int(epoch), "value/linkpred_auc": float(auc)}, f, indent=2)

                    print(f"\n[SAVE] best graphs updated: auc={auc:.4f} -> {out_pkl}")

            del generated_graphs
            del deltas
            self._cleanup_cuda()

        return eval_dict

    def test_fn(self, epoch):
        self.model.eval()
        test_dict = {}
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            data_count = 0

            for pyg_data in self.test_loader:
                if self.args.parallel != 'dp':
                    pyg_data = pyg_data.to(self.args.device)
                # pyg_data.num_entries = self.model._calc_num_entries(pyg_data)
                loss = elbo_bpd(self.model, pyg_data) 
                loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs#len(x)
                loss_count += pyg_data.num_graphs #len(x)
                data_count += pyg_data.num_graphs #pyg_data.num_graphs
            print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, data_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')            
            test_dict['bpd'] = loss_sum/loss_count
            generated_graphs, deltas = self._sample_generated_graphs("test")

            w = 8 if self._generation_count("test") >= 64 else 2
            fig, axes = plt.subplots(w, w, figsize=(17,17))
            for i, g_gen in enumerate(generated_graphs[:w**2]):
                nx.draw(g_gen, ax=axes[i%w][i//w], node_size=30)

            os.makedirs(os.path.join(self.log_path, "test"), exist_ok=True)
            plt.savefig(os.path.join(self.log_path, f"test/sample{epoch}.png"))
            plt.close()

            # statistics evaluation
            metrics = self.test_evaluator.evaluate(generated_graphs)
            test_dict.update(metrics)

            del generated_graphs
            del deltas
            self._cleanup_cuda()

        return test_dict 

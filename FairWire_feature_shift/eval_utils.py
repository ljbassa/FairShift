import dgl
import dgl.function as fn
import dgl.sparse as dglsp
import networkx as nx
import numpy as np
import os
import secrets
import subprocess as sp
import torch

from functools import partial
from pprint import pprint
from scipy import stats
from string import ascii_uppercase, digits

from Model import BaseEvaluator, MLPTrainer, SGCTrainer, GCNTrainer,\
    APPNPTrainer, GAETrainer, CNEvaluator

def get_triangle_count(nx_g):
    triangle_count = sum(nx.triangles(nx.to_undirected(nx_g)).values()) / 3
    return triangle_count

def linkx_homophily(graph, y):
    r"""Homophily measure from `Large Scale Learning on Non-Homophilous Graphs:
    New Benchmarks and Strong Simple Methods
    <https://arxiv.org/abs/2110.14446>`__

    Mathematically it is defined as follows:

    .. math::
      \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, \frac{\sum_{v\in C_k}|\{u\in
      \mathcal{N}(v): y_v = y_u \}|}{\sum_{v\in C_k}|\mathcal{N}(v)|} -
      \frac{|\mathcal{C}_k|}{|\mathcal{V}|} \right),

    where :math:`C` is the number of node classes, :math:`C_k` is the set of
    nodes that belong to class k, :math:`\mathcal{N}(v)` are the predecessors
    of node :math:`v`, :math:`y_v` is the class of node :math:`v`, and
    :math:`\mathcal{V}` is the set of nodes.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    y : torch.Tensor
        The node labels, which is a tensor of shape (|V|).

    Returns
    -------
    float
        The homophily value.
    """
    with graph.local_scope():
        # Compute |{u\in N(v): y_v = y_u}| for each node v.
        src, dst = graph.edges()
        # Compute y_v = y_u for all edges.
        graph.edata["same_class"] = (y[src] == y[dst]).float()
        graph.update_all(
            fn.copy_e("same_class", "m"), fn.sum("m", "same_class_deg")
        )

        deg = graph.in_degrees().float()
        num_nodes = graph.num_nodes()
        num_classes = y.max(dim=0).values.item() + 1

        value = torch.tensor(0.0).to(graph.device)
        for k in range(num_classes):
            # Get the nodes that belong to class k.
            class_mask = y == k
            same_class_deg_k = graph.ndata["same_class_deg"][class_mask].sum()
            deg_k = deg[class_mask].sum()
            num_nodes_k = class_mask.sum()
            value += max(0, same_class_deg_k / deg_k - num_nodes_k / num_nodes)

        return value.item() / (num_classes - 1)

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def get_adj(dgl_g):
    # Get symmetrically normalized adjacency matrix.
    A = dgl_g.adj()
    N = dgl_g.num_nodes()
    I = dglsp.identity((N, N), device=dgl_g.device)
    A_hat = A + I
    D_hat = dglsp.diag(A_hat.sum(1)) ** -0.5
    A_norm = D_hat @ A_hat @ D_hat

    return A_norm

def get_edge_split(A_dense):
    # Exclude self-loops.
    A_dense_upper = torch.triu(A_dense, diagonal=1)
    real_edges = A_dense_upper.nonzero()

    real_indices = torch.randperm(real_edges.size(0))
    real_edges = real_edges[real_indices]

    num_real = len(real_edges)
    num_train = int(num_real * 0.8)
    num_val = int(num_real * 0.1)
    num_test = num_real - num_train - num_val

    real_train, real_val, real_test = torch.split(
        real_edges, [num_train, num_val, num_test])

    neg_edges = torch.triu((A_dense == 0).float(), diagonal=1).nonzero()
    neg_indices = torch.randperm(neg_edges.size(0))

    neg_val = neg_edges[neg_indices[:num_val]]
    neg_test = neg_edges[neg_indices[num_val:num_val+num_test]]

    return real_train, real_val, real_test, neg_val, neg_test

def prepare_for_GAE(A):
    A_dense = A.to_dense()

    real_train, real_val, real_test, neg_val, neg_test = get_edge_split(A_dense)

    num_nodes = A_dense.size(0)
    train_mask = torch.zeros(num_nodes, num_nodes)
    val_mask = torch.zeros(num_nodes, num_nodes)
    test_mask = torch.zeros(num_nodes, num_nodes)

    edge_train = real_train
    edge_val = torch.cat([real_val, neg_val], dim=0)
    edge_test = torch.cat([real_test, neg_test], dim=0)

    row_train, col_train = edge_train.T
    train_mask[row_train, col_train] = 1.

    row_val, col_val = edge_val.T
    val_mask[row_val, col_val] = 1.

    row_test, col_test = edge_test.T
    test_mask[row_test, col_test] = 1.

    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()

    real_row_train, real_col_train = real_train.T
    train_g = dgl.graph((real_row_train, real_col_train), num_nodes=num_nodes)
    train_g = dgl.to_bidirected(train_g)
    A_train = get_adj(train_g)

    return A_train, train_mask, val_mask, test_mask

def emd(p, q):
    return (
        torch.cumsum(p, dim=0) - torch.cumsum(q, dim=0)
    ).abs().sum().item()

def get_pairwise_emd(real_dists, sample_dists):
    emd_list = []
    for p in real_dists:
        for q in sample_dists:
            emd_list.append(emd(p, q))
    return float(np.mean(emd_list))

def get_deg_emd(real_degs, sample_degs):
    """Compute the earth mover distance (EMD) between
    two degree distributions.

    Parameters
    ----------
    real_degs : list of torch.Tensor of shape (|V1|)
        Node degrees of the real graphs.
    sample_degs : list of torch.Tensor of shape (|V2|)
        Node degrees of the sampled graphs.

    Returns
    -------
    emd
        The EMD value.
    """
    max_deg = max(
        max([deg.max().item() for deg in real_degs]),
        max([deg.max().item() for deg in sample_degs])
    )

    def get_degree_dist(deg):
        num_nodes = deg.size(0)
        freq = torch.zeros(num_nodes, max_deg + 1)
        freq[torch.arange(num_nodes), deg] = 1.
        freq = freq.sum(dim=0)
        return freq / (freq.sum() + 1e-6)

    real_dists = []
    for deg in real_degs:
        real_dists.append(get_degree_dist(deg))

    sample_dists = []
    for deg in sample_degs:
        sample_dists.append(get_degree_dist(deg))

    return get_pairwise_emd(real_dists, sample_dists)

def get_cluster_emd(real_vals, sample_vals, bins=100):
    """Compute the earth mover distance (EMD) between
    two clustering coefficient distributions.

    Parameters
    ----------
    real_vals : list of list of length (|V1|)
        Node clustering coefficients of the real graphs.
    sample_vals : list of list of length (|V2|)
        Node clustering coefficients of the sampled graphs.
    bins : int
        Number of equal-width bins in the given range.

    Returns
    -------
    emd
        The EMD value.
    """
    def get_cluster_dist(vals):
        hist, _ = np.histogram(
            vals, bins=bins, range=(0.0, 1.0), density=False)
        hist = torch.from_numpy(hist)
        return hist / (hist.sum() + 1e-6)

    real_dists = []
    for vals in real_vals:
        real_dists.append(get_cluster_dist(vals))

    sample_dists = []
    for vals in sample_vals:
        sample_dists.append(get_cluster_dist(vals))

    return get_pairwise_emd(real_dists, sample_dists)

class Evaluator:
    def __init__(self,
                 data_name,
                 model_dir,
                 dgl_g_real,
                 X_one_hot_3d_real,
                 s_one_hot_real,
                 Y_one_hot_real,
                 seed=0
                 ):
        """
        Parameters
        ----------
        data_name : str
            Name of the dataset.
        dgl_g_real : dgl.DGLGraph
            Real graph.
        X_one_hot_3d_real : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d_real[f, :, :] is the one-hot encoding of the f-th node
            attribute in the real graph.
        s_one_hot_real : torch.Tensor of shape (|V|, S)
            One-hot encoding of the sensitive attributes in the real graph.
        Y_one_hot_real : torch.Tensor of shape (|V|, C)
            One-hot encoding of the node label in the real graph.
        """
        self.data_name = data_name
        self.Y_one_hot_real = Y_one_hot_real
        # If the number of edges in a newly added graph exceeds this limit,
        # a subgraph will be used for certain metric computations.
        self.edge_limit = min(dgl_g_real.num_edges(), 20000)

        # Split datasets without a built-in split.
        add_mask = False
        torch.manual_seed(0)
        if data_name in ["amazon_photo", "amazon_computer"]:
            add_mask = True
        

        dgl_g_real, X_real, s_real, Y_real, data_dict_real = self.preprocess_g(
            dgl_g_real,
            X_one_hot_3d_real,
            s_one_hot_real,
            Y_one_hot_real,
            add_mask)
        self.data_dict_real = data_dict_real
        self.data_dict_sample_list = []
        A_real = get_adj(dgl_g_real)
        torch.manual_seed(seed)
        if Y_real is not None:
            num_classes = len(Y_real.unique())
        

            os.makedirs(f"{data_name}_cpts", exist_ok=True)
            self.mlp_evaluator = BaseEvaluator(MLPTrainer,
                                               f"{model_dir}/mlp.pth",
                                               num_classes,
                                               train_mask=dgl_g_real.ndata["train_mask"],
                                               val_mask=dgl_g_real.ndata["val_mask"],
                                               test_mask=dgl_g_real.ndata["test_mask"],
                                               X=X_real,
                                               s=s_real,
                                               Y=Y_real)

            self.sgc_one_layer_evaluator = BaseEvaluator(
                partial(SGCTrainer, num_gnn_layers=1),
                f"{model_dir}/sgc_one_layer.pth",
                num_classes,
                train_mask=dgl_g_real.ndata["train_mask"],
                val_mask=dgl_g_real.ndata["val_mask"],
                test_mask=dgl_g_real.ndata["test_mask"],
                A=A_real,
                X=X_real,
                s=s_real,
                Y=Y_real)

            self.sgc_two_layer_evaluator = BaseEvaluator(
                partial(SGCTrainer, num_gnn_layers=2),
                f"{model_dir}/sgc_two_layer.pth",
                num_classes,
                train_mask=dgl_g_real.ndata["train_mask"],
                val_mask=dgl_g_real.ndata["val_mask"],
                test_mask=dgl_g_real.ndata["test_mask"],
                A=A_real,
                X=X_real,
                s=s_real,
                Y=Y_real)

            self.gcn_evaluator = BaseEvaluator(
                partial(GCNTrainer, num_gnn_layers=2),
                f"{model_dir}/gcn.pth",
                num_classes,
                train_mask=dgl_g_real.ndata["train_mask"],
                val_mask=dgl_g_real.ndata["val_mask"],
                test_mask=dgl_g_real.ndata["test_mask"],
                A=A_real,
                X=X_real,
                s=s_real,
                Y=Y_real)

            self.appnp_one_layer_evaluator = BaseEvaluator(
                partial(APPNPTrainer, num_gnn_layers=1),
                f"{model_dir}/appnp_one_layer.pth",
                num_classes,
                train_mask=dgl_g_real.ndata["train_mask"],
                val_mask=dgl_g_real.ndata["val_mask"],
                test_mask=dgl_g_real.ndata["test_mask"],
                A=A_real,
                X=X_real,
                s=s_real,
                Y=Y_real)

            self.appnp_two_layer_evaluator = BaseEvaluator(
                partial(APPNPTrainer, num_gnn_layers=2),
                f"{model_dir}/appnp_two_layer.pth",
                num_classes,
                train_mask=dgl_g_real.ndata["train_mask"],
                val_mask=dgl_g_real.ndata["val_mask"],
                test_mask=dgl_g_real.ndata["test_mask"],
                A=A_real,
                X=X_real,
                s=s_real,
                Y=Y_real)
        else:
            num_classes=None

            # Generate train/val/test mask for link prediction.
            # Fix the raw graph split for reproducibility.
            
            A_real_train, train_mask, val_mask, test_mask = prepare_for_GAE(A_real)

            self.gae_one_layer_evaluator = BaseEvaluator(
                partial(GAETrainer, num_gnn_layers=1),
                f"{model_dir}/gae_one_layer.pth",
                num_classes,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                A_train=A_real_train,
                A_full=A_real,
                X=X_real,
                s=s_real,
                Y=Y_real)
            
            #self.gae_two_layer_evaluator = BaseEvaluator(
            #    partial(GAETrainer, num_gnn_layers=2),
            #    f"{model_dir}/gae_two_layer.pth",
            #    num_classes,
            #    train_mask=train_mask,
            #    val_mask=val_mask,
            #    test_mask=test_mask,
            #    A_train=A_real_train,
            #    A_full=A_real,
            #    X=X_real,
            #    s=s_real,
            #    Y=Y_real)

            self.cn_evaluator = CNEvaluator(
                f"{model_dir}/cn.pth",
                A_train=A_real_train,
                s=s_real,
                Y=Y_real,
                A_full=A_real,
                val_mask=val_mask,
                test_mask=test_mask
            )

    def add_mask_cora(self, dgl_g, Y_one_hot):
        num_nodes = dgl_g.num_nodes()
        train_mask = torch.zeros(num_nodes)
        val_mask = torch.zeros(num_nodes)
        test_mask = torch.zeros(num_nodes)

        # Based on the raw graph
        num_val_nodes = {
            0: 61,
            1: 36,
            2: 78,
            3: 158,
            4: 81,
            5: 57,
            6: 29
        }

        num_test_nodes = {
            0: 130,
            1: 91,
            2: 144,
            3: 319,
            4: 149,
            5: 103,
            6: 64
        }

        num_classes = Y_one_hot.size(-1)
        for y in range(num_classes):
            nodes_y = (Y_one_hot[:, y] == 1.).nonzero().squeeze(-1)
            nid_y = torch.randperm(len(nodes_y))
            nodes_y = nodes_y[nid_y]

            train_mask[nodes_y[:20]] = 1.

            start = 20
            end = start + num_val_nodes[y]
            val_mask[nodes_y[start: end]] = 1.

            start = end
            end = start + num_test_nodes[y]
            test_mask[nodes_y[start: end]] = 1.

        dgl_g.ndata["train_mask"] = train_mask.bool()
        dgl_g.ndata["val_mask"] = val_mask.bool()
        dgl_g.ndata["test_mask"] = test_mask.bool()

        return dgl_g
    
    def add_mask_nc(self, dgl_g, Y_one_hot):
        num_nodes = dgl_g.num_nodes()
        
        train_mask = torch.zeros(num_nodes)
        val_mask = torch.zeros(num_nodes)
        test_mask = torch.zeros(num_nodes)

        num_classes = Y_one_hot.size(-1)
        for y in range(num_classes):
            nodes_y = (Y_one_hot[:, y] == 1.).nonzero().squeeze(-1)
            
            pop = len(nodes_y)
            train_size = int(pop*0.6)
            test_size = int(pop*0.2)
            
            nid_y = torch.randperm(len(nodes_y))
            nodes_y = nodes_y[nid_y]

            train_mask[nodes_y[:train_size]] = 1.
            val_mask[nodes_y[train_size:train_size+test_size]] = 1.
            test_mask[nodes_y[train_size+test_size:]] = 1.

        dgl_g.ndata["train_mask"] = train_mask.bool()
        dgl_g.ndata["val_mask"] = val_mask.bool()
        dgl_g.ndata["test_mask"] = test_mask.bool()
        
        print("Current train size for synthetic graph: ", sum(train_mask))
        print("Current val size for synthetic graph: ", sum(val_mask))
        print("Current test size for synthetic graph: ", sum(test_mask))

        return dgl_g

    def add_mask_citeseer(self, dgl_g, Y_one_hot):
        num_nodes = dgl_g.num_nodes()
        train_mask = torch.zeros(num_nodes)
        val_mask = torch.zeros(num_nodes)
        test_mask = torch.zeros(num_nodes)

        # Based on the raw graph
        num_val_nodes = {
            0: 29,
            1: 86,
            2: 116,
            3: 106,
            4: 94,
            5: 69
        }

        num_test_nodes = {
            0: 77,
            1: 182,
            2: 181,
            3: 231,
            4: 169,
            5: 160
        }

        num_classes = Y_one_hot.size(-1)
        for y in range(num_classes):
            nodes_y = (Y_one_hot[:, y] == 1.).nonzero().squeeze(-1)
            nid_y = torch.randperm(len(nodes_y))
            nodes_y = nodes_y[nid_y]

            train_mask[nodes_y[:20]] = 1.

            start = 20
            end = start + num_val_nodes[y]
            val_mask[nodes_y[start: end]] = 1.

            start = end
            end = start + num_test_nodes[y]
            test_mask[nodes_y[start: end]] = 1.

        dgl_g.ndata["train_mask"] = train_mask.bool()
        dgl_g.ndata["val_mask"] = val_mask.bool()
        dgl_g.ndata["test_mask"] = test_mask.bool()

        return dgl_g

    def add_mask_benchmark(self, dgl_g, Y_one_hot):
        num_nodes = dgl_g.num_nodes()
        train_mask = torch.zeros(num_nodes)
        val_mask = torch.zeros(num_nodes)
        test_mask = torch.zeros(num_nodes)

        num_classes = Y_one_hot.size(-1)
        for y in range(num_classes):
            nodes_y = (Y_one_hot[:, y] == 1.).nonzero().squeeze(-1)
            nid_y = torch.randperm(len(nodes_y))
            nodes_y = nodes_y[nid_y]

            # Based on the raw paper.
            train_mask[nodes_y[:20]] = 1.
            val_mask[nodes_y[20: 50]] = 1.
            test_mask[nodes_y[50:]] = 1.

        dgl_g.ndata["train_mask"] = train_mask.bool()
        dgl_g.ndata["val_mask"] = val_mask.bool()
        dgl_g.ndata["test_mask"] = test_mask.bool()

        return dgl_g

    def add_mask(self, dgl_g, Y_one_hot):
        if self.data_name == "cora":
            return self.add_mask_cora(dgl_g, Y_one_hot)
        elif self.data_name == "citeseer":
            return self.add_mask_citeseer(dgl_g, Y_one_hot)
        elif self.data_name in ["amazon_photo", "amazon_computer"]:
            return self.add_mask_benchmark(dgl_g, Y_one_hot)
        elif self.data_name in ["pokec_z", "pokec_n", "bail", "credit", "german"]:
            return self.add_mask_nc(dgl_g, Y_one_hot)
        else:
            raise ValueError(f'Unexpected data name: {self.data_name}')

    def sample_subg(self, dgl_g):
        # Sample edge-induced subgraph for costly computation.
        A = dgl_g.adj().to_dense()
        A_upper = torch.triu(A, diagonal=1)
        # (|E|, 2)
        edges = A_upper.nonzero()
        indices = torch.randperm(edges.size(0))[:self.edge_limit // 2]
        src, dst = edges[indices].T
        sub_g = dgl.graph((src, dst), num_nodes=dgl_g.num_nodes())
        sub_g = dgl.to_bidirected(sub_g)

        return sub_g

    def k_order_g(self, dgl_g, k):
        # Get DGLGraph of A^k.
        A = dgl_g.adj().to_dense()
        A_new = A
        for _ in range(k-1):
            A_new = A_new @ A
        src, dst = A_new.nonzero().T
        new_g = dgl.graph((src, dst), num_nodes=dgl_g.num_nodes())
        return new_g

    def preprocess_g(self,
                     dgl_g,
                     X_one_hot_3d,
                     s_one_hot,
                     Y_one_hot,
                     add_mask):
        """
        Parameters
        ----------
        dgl_g : dgl.DGLGraph
            Graph.
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node
            attribute in the graph.
        s_one_hot : torch.Tensor of shape (|V|, S)
            One-hot encoding of the sensitive attributes in the graph.
        Y_one_hot : torch.Tensor of shape (|V|, C)
            One-hot encoding of the node label in the graph.
        add_mask : bool
            Whether to add a mask to the graph for node classification
            data split.

        Returns
        -------
        dgl_g : dgl.DGLGraph
            Graph, potentially with node mask added.
        X : torch.Tensor of shape (|V|, F)
            Node attributes.
        s : torch.Tensor of shape (|V|)
            Categorical sensitive attributes.
        Y : torch.Tensor of shape (|V|)
            Categorical node label.
        data_dict : dict
            Dictionary of graph statistics.
        """
        if add_mask and (Y_one_hot is not None):
            dgl_g = self.add_mask(dgl_g, Y_one_hot)

        F = X_one_hot_3d.size(0)
        # (|V|, F)
        X = torch.zeros(X_one_hot_3d.size(1), F)
        for f in range(F):
            X[:, f] = X_one_hot_3d[f].argmax(dim=1)

        if dgl_g.num_edges() > self.edge_limit:
            dgl_subg = self.sample_subg(dgl_g)
        else:
            dgl_subg = dgl_g

        nx_g = nx.DiGraph(dgl_subg.cpu().to_networkx())

        triangle_count = get_triangle_count(nx_g)
        if Y_one_hot is not None:
            Y = Y_one_hot.argmax(dim=-1)
            dgl_g_pow_2 = self.k_order_g(dgl_g, 2)
            linkx_A = linkx_homophily(dgl_g, Y)
            linkx_A_pow_2 = linkx_homophily(dgl_g_pow_2, Y)
        else:
            Y=None
            linkx_A = None
            linkx_A_pow_2 = None
        s = s_one_hot.argmax(dim=-1)
        
        dgl_g_pow_2 = self.k_order_g(dgl_g, 2)

        degs = dgl_g.in_degrees()
        cluster_coefs = list(nx.clustering(nx_g).values())

        data_dict = {
            "degs": degs,
            "cluster_coefs": cluster_coefs,
        }

        return dgl_g, X, s, Y, data_dict

    def add_sample(self,
                   dgl_g,
                   X_one_hot_3d,
                   s_one_hot,
                   Y_one_hot):
        """Add a generated sample for evaluation.

        Parameters
        ----------
        dgl_g : dgl.DGLGraph
            Generated graph.
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node
            attribute in the generated graph.
        s_one_hot : torch.Tensor of shape (|V|, S)
            One-hot encoding of the sensitive attributes in the generated graph.
        Y_one_hot : torch.Tensor of shape (|V|, C)
            One-hot encoding of the node label in the generated graph.
        """
        dgl_g_sample, X_sample, s_sample, Y_sample, data_dict_sample = self.preprocess_g(
            dgl_g,
            X_one_hot_3d,
            s_one_hot,
            Y_one_hot,
            add_mask=True)

        self.data_dict_sample_list.append(data_dict_sample)
        A_sample = get_adj(dgl_g_sample)
        if Y_sample is not None:
            self.mlp_evaluator.add_sample(
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=dgl_g_sample.ndata["train_mask"],
                val_mask=dgl_g_sample.ndata["val_mask"],
                test_mask=dgl_g_sample.ndata["test_mask"])
       

            self.sgc_one_layer_evaluator.add_sample(
                A=A_sample,
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=dgl_g_sample.ndata["train_mask"],
                val_mask=dgl_g_sample.ndata["val_mask"],
                test_mask=dgl_g_sample.ndata["test_mask"])

            self.sgc_two_layer_evaluator.add_sample(
                A=A_sample,
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=dgl_g_sample.ndata["train_mask"],
                val_mask=dgl_g_sample.ndata["val_mask"],
                test_mask=dgl_g_sample.ndata["test_mask"])

            self.gcn_evaluator.add_sample(
                A=A_sample,
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=dgl_g_sample.ndata["train_mask"],
                val_mask=dgl_g_sample.ndata["val_mask"],
                test_mask=dgl_g_sample.ndata["test_mask"])

            self.appnp_one_layer_evaluator.add_sample(
                A=A_sample,
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=dgl_g_sample.ndata["train_mask"],
                val_mask=dgl_g_sample.ndata["val_mask"],
                test_mask=dgl_g_sample.ndata["test_mask"])

            self.appnp_two_layer_evaluator.add_sample(
                A=A_sample,
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=dgl_g_sample.ndata["train_mask"],
                val_mask=dgl_g_sample.ndata["val_mask"],
                test_mask=dgl_g_sample.ndata["test_mask"])
        else:
            # Generate train/val/test mask.
            A_sample_train, train_mask, val_mask, test_mask = prepare_for_GAE(A_sample)
            
            self.gae_one_layer_evaluator.add_sample(
                A_train=A_sample_train,
                A_full=A_sample,
                X=X_sample,
                s=s_sample,
                Y=Y_sample,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask)
           
            self.cn_evaluator.add_sample(
                A_train=A_sample_train,
                s=s_sample,
                Y=Y_sample,
                A_full=A_sample,
                val_mask=val_mask,
                test_mask=test_mask
            )

    def summary(self):
        report = dict()

        report["deg_emd"] = get_deg_emd(
            [self.data_dict_real["degs"]],
            [data_dict_sample["degs"] for data_dict_sample in self.data_dict_sample_list])

        # clustering coefficient EMD
        report["cluster_emd"] = get_cluster_emd(
            [self.data_dict_real["cluster_coefs"]],
            [data_dict_sample["cluster_coefs"]
             for data_dict_sample in self.data_dict_sample_list]
        )

        print('\n')
        pprint(report)
        if self.Y_one_hot_real is not None:
            print('\nAPPNP 1-layer discriminator')
            self.appnp_one_layer_evaluator.summary()

            print('\nAPPNP 2-layer discriminator')
            self.appnp_two_layer_evaluator.summary()
        else:
            print('\nGAE 1-layer discriminator')
            self.gae_one_layer_evaluator.summary()

                

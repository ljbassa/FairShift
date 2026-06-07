import math
import torch
import os 
import networkx as nx
import numpy as np

import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from datasets.data_utils import EmpiricalEmptyGraphGenerator, NeuralEmptyGraphGenerator, preprocess, collate_fn, FEATURE_EXTRACTOR
from datasets.evaluator import NetworkEvaluator


class NetworkDataset(Dataset):
    def __init__(self, pyg_graph, num_iter, transform=None):
        super().__init__()
        self.pyg_data = pyg_graph
        self.transform = transform
        self.num_iter = num_iter

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.pyg_graph)
        return self.pyg_data

    def __len__(self):
        return self.num_iter


class GraphDataset(Dataset):
    def __init__(self, pyg_datas):
        super().__init__()
        self.pyg_datas = pyg_datas

    def __getitem__(self, index):
        return self.pyg_datas[index]#, self.denses[index]

    def __len__(self):
        return len(self.pyg_datas)


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str)
    # Train params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=eval, default=True)

    parser.add_argument('--empty_graph_sampler', type=str, default='empirical', help='empirical | neural') 
    parser.add_argument('--degree', action='store_true')
    parser.add_argument('--augmented_features', type=str, nargs="*", default=[])

def get_data_id(args):
    return '{}'.format(args.dataset)

def _load_graph_with_optional_feat(dataset_name: str):
    feat_path = f"graphs/{dataset_name}_feat.pkl"
    base_path = f"graphs/{dataset_name}.pkl"

    if os.path.exists(feat_path):
        return pkl.load(open(feat_path, "rb"))
    return pkl.load(open(base_path, "rb"))

def get_data(args):
    repeat = 1
    num_node_classes = None
    num_edge_classes = 2
    
    nx_graph = _load_graph_with_optional_feat(args.dataset)
    pyg_graph = preprocess(nx_graph, degree=args.degree)
    
    # node feature dim 기록 (있으면)
    num_node_feat = None
    if hasattr(pyg_graph, "x") and pyg_graph.x is not None:
        num_node_feat = pyg_graph.x.size(-1)

    max_degree = max([d for _, d in nx_graph.degree()]) 
    train_set = NetworkDataset(pyg_graph, num_iter=args.num_iter * args.batch_size, transform=None)
    test_set = eval_set = NetworkDataset(pyg_graph, num_iter=100, transform=None)
    
    initial_graph_sampler = EmpiricalEmptyGraphGenerator([train_set[0]], degree=args.degree, augment_features=args.augmented_features)
    eval_evaluator = test_evaluator = NetworkEvaluator(
        nx_graph,
        fair_label_attr=getattr(args, 'fair_label_attr', getattr(args, 'fair_sensitive_attr', 'y')),
    )

    #AUC 추가
    monitoring_statistics = ['nmae/clustering_coefficient', 'nmae/linkpred_auc']

    augmented_feature_dict = {k:FEATURE_EXTRACTOR[k]['data_spec'] for k in args.augmented_features}

    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size*repeat, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=partial(collate_fn))
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)

    return train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics
 

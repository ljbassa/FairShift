import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
import dgl
import networkx as nx
from typing import Dict, Tuple
import pickle
from os.path import join, dirname, realpath
import csv
import pickle as pkl
import requests
import os
import pdb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import zipfile
import io
import gdown

"""
NotLoaded: LCC, Filmtrust, Lastfm, UNC, oklahoma
"""


import requests


def feature_norm(self, features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


class Dataset(object):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset") -> None:
        self.adj_ = None
        self.features_ = None
        self.labels_ = None
        self.idx_train_ = None
        self.idx_val_ = None
        self.idx_test_ = None
        self.sens_ = None
        self.sens_idx_ = None
        self.is_normalize = is_normalize

        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.path_name = ""

    def download(self, url: str, filename: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(os.path.join(self.root, self.path_name, filename), "wb").write(r.content)

    def download_zip(self, url: str):
        r = requests.get(url)
        assert r.status_code == 200
        foofile = zipfile.ZipFile(io.BytesIO(r.content))
        foofile.extractall(os.path.join(self.root, self.path_name))

    def adj(self, datatype: str = "torch.sparse"):
        # assert str(type(self.adj_)) == "<class 'torch.Tensor'>"
        if self.adj_ is None:
            return self.adj_
        if datatype == "torch.sparse":
            return self.adj_
        elif datatype == "scipy.sparse":
            return sp.coo_matrix(self.adj_.to_dense())
        elif datatype == "np.array":
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError(
                "datatype should be torch.sparse, tf.sparse, np.array, or scipy.sparse"
            )

    def features(self, datatype: str = "torch.tensor"):
        if self.is_normalize and self.features_ is not None:
            self.features_ = feature_norm(self, self.features_)

        if self.features is None:
            return self.features_
        if datatype == "torch.tensor":
            return self.features_
        elif datatype == "np.array":
            return self.features_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def labels(self, datatype: str = "torch.tensor"):
        if self.labels_ is None:
            return self.labels_
        if datatype == "torch.tensor":
            return self.labels_
        elif datatype == "np.array":
            return self.labels_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_val(self, datatype: str = "torch.tensor"):
        if self.idx_val_ is None:
            return self.idx_val_
        if datatype == "torch.tensor":
            return self.idx_val_
        elif datatype == "np.array":
            return self.idx_val_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_train(self, datatype: str = "torch.tensor"):
        if self.idx_train_ is None:
            return self.idx_train_
        if datatype == "torch.tensor":
            return self.idx_train_
        elif datatype == "np.array":
            return self.idx_train_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_test(self, datatype: str = "torch.tensor"):
        if self.idx_test_ is None:
            return self.idx_test_
        if datatype == "torch.tensor":
            return self.idx_test_
        elif datatype == "np.array":
            return self.idx_test_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens(self, datatype: str = "torch.tensor"):
        if self.sens_ is None:
            return self.sens_
        if datatype == "torch.tensor":
            return self.sens_
        elif datatype == "np.array":
            return self.sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens_idx(self):
        if self.sens_idx_ is None:
            self.sens_idx_ = -1
        return self.sens_idx_


def mx_to_torch_sparse_tensor(sparse_mx, is_sparse=False, return_tensor_sparse=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not is_sparse:
        sparse_mx = sp.coo_matrix(sparse_mx)
    else:
        sparse_mx = sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Pokec_z(Dataset):
    def __init__(
        self,
        dataset_name="pokec_z",
        predict_attr_specify=None,
        return_tensor_sparse=True,
        is_normalize: bool = False,
        root: str = "./dataset",
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        if dataset_name != "nba":
            if dataset_name == "pokec_z":
                dataset = "region_job"
            elif dataset_name == "pokec_n":
                dataset = "region_job_2"
            else:
                dataset = None
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path = "./dataset/pokec/"
            test_idx = False
        else:
            dataset = "nba"
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "./dataset/NBA"
            test_idx = True

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            idx_sens_train,
        ) = self.load_pokec(
            dataset,
            sens_attr,
            predict_attr if predict_attr_specify == None else predict_attr_specify,
            path=path,
            label_number=label_number,
            sens_number=sens_number,
            seed=seed,
            test_idx=test_idx,
        )

        # adj=adj.todense(
        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        labels[labels > 1] = 1
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "pokec_z"
        self.url = "https://drive.google.com/u/0/uc?id=1FOYOIdFp6lI9LH5FJAzLhjFCMAxT6wb4&export=download"
        self.destination = os.path.join(self.root, self.path_name, "pokec_z.zip")
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job.csv")
        ):
            gdown.download(self.url, self.destination)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job_relationship.txt")
        ):
            gdown.download(self.url, self.destination, quiet=False)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "region_job.csv")
        )
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        sens = idx_features_labels[sens_attr].values
        
        idx_used = np.array(labels>=0) & np.array(sens>=0)
       
        features = features[idx_used, :]
        labels = labels[idx_used]
        sens = sens[idx_used]
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx = idx[idx_used]
        
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "region_job_relationship.txt"),
            dtype=np.int64,
        )
        
        edges_idx = np.array([(src in idx)& (dst in idx) for src, dst in edges_unordered])
        edges_unordered = edges_unordered[edges_idx, :]
        idx_map = {j: i for i, j in enumerate(idx)}

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.LongTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


class Pokec_n(Dataset):
    def __init__(
        self,
        dataset_name="pokec_n",
        predict_attr_specify=None,
        return_tensor_sparse=True,
        is_normalize: bool = False,
        root: str = "./dataset",
    ):
        super().__init__(is_normalize=is_normalize, root=root)
        if dataset_name != "nba":
            if dataset_name == "pokec_z":
                dataset = "region_job"
            elif dataset_name == "pokec_n":
                dataset = "region_job_2"
            else:
                dataset = None
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            sens_number = 200
            seed = 20
            path = "./dataset/pokec/"
            test_idx = False
        else:
            dataset = "nba"
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100
            sens_number = 50
            seed = 20
            path = "./dataset/NBA"
            test_idx = True

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            idx_sens_train,
        ) = self.load_pokec(
            dataset,
            sens_attr,
            predict_attr if predict_attr_specify == None else predict_attr_specify,
            path=path,
            label_number=label_number,
            sens_number=sens_number,
            seed=seed,
            test_idx=test_idx,
        )

        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )
        labels[labels > 1] = 1
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000,
        sens_number=500,
        seed=19,
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "pokec_n"
        self.url = "https://drive.google.com/u/0/uc?id=1wWm6hyCUjwnr0pWlC6OxZIj0H0ZSnGWs&export=download"
        self.destination = os.path.join(self.root, self.path_name, "pokec_n.zip")
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job_2.csv")
        ):
            gdown.download(self.url, self.destination, quiet=False)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "region_job_2_relationship.txt")
        ):
            gdown.download(self.url, self.destination, quiet=False)
            with zipfile.ZipFile(self.destination, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, self.path_name))

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "region_job_2.csv")
        )
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        sens = idx_features_labels[sens_attr].values
        
        idx_used = np.array(labels>=0) & np.array(sens>=0)
       
        features = features[idx_used, :]
        labels = labels[idx_used]
        sens = sens[idx_used]
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx = idx[idx_used]
        
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "region_job_2_relationship.txt"),
            dtype=np.int64,
        )
        
        edges_idx = np.array([(src in idx)& (dst in idx) for src, dst in edges_unordered])
        edges_unordered = edges_unordered[edges_idx, :]
        idx_map = {j: i for i, j in enumerate(idx)}


        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.LongTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train
class Bail(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(Bail, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_bail("bail")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_bail(
        self,
        dataset,
        sens_attr="WHITE",
        predict_attr="RECID",
        path="./dataset/bail/",
        label_number=100,
    ):
        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "bail"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "bail.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail.csv"
            file_name = "bail.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "bail_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail_edges.txt"
            file_name = "bail_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # build relationship

        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.LongTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0


class Credit(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(Credit, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_credit("credit")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_credit(
        self,
        dataset,
        sens_attr="Age",
        predict_attr="NoDefaultNextMonth",
        path="./dataset/credit/",
        label_number=6000,
    ):
        from scipy.spatial import distance_matrix

        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "credit"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "credit.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit.csv"
            file_name = "credit.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "credit_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit_edges.txt"
            file_name = "credit_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("Single")

        # build relationship
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.LongTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 1
class German(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(German, self).__init__(is_normalize=is_normalize, root=root)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_german("german")

        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)

        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_german(
        self,
        dataset,
        sens_attr="Gender",
        predict_attr="GoodCustomer",
        path="./dataset/german/",
        label_number=100,
    ):
        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "german"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))

        if not os.path.exists(os.path.join(self.root, self.path_name, "german.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german.csv"
            file_name = "german.csv"
            self.download(url, file_name)
        if not os.path.exists(
            os.path.join(self.root, self.path_name, "german_edges.txt")
        ):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german_edges.txt"
            file_name = "german_edges.txt"
            self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("OtherLoansAtStore")
        header.remove("PurposeOfLoan")

        # Sensitive Attribute
        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Female"] = 1
        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Male"] = 0

        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))

        labels = torch.LongTensor(labels)

        import random

        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.LongTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0


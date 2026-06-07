import dgl
import torch
import torch.nn.functional as F
import scipy.stats
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, \
    CoraGraphDataset, CiteseerGraphDataset
from datasets import Pokec_z, Pokec_n, Bail, Credit, German

def load_dataset(data_name):
    if data_name == "cora":
        dataset = CoraGraphDataset()
    elif data_name == "citeseer":
        dataset = CiteseerGraphDataset()
    elif data_name == "amazon_photo":
        dataset = AmazonCoBuyPhotoDataset()
    elif data_name == "amazon_computer":
        dataset = AmazonCoBuyComputerDataset()

    g = dataset[0]
    g = dgl.remove_self_loop(g)

    X = g.ndata['feat']
    X[X != 0] = 1.

    # Remove columns with constant values.
    non_full_zero_feat_mask = X.sum(dim=0) != 0
    X = X[:, non_full_zero_feat_mask]

    non_full_one_feat_mask = X.sum(dim=0) != X.size(0)
    X = X[:, non_full_one_feat_mask]

    g.ndata['feat'] = X
    return g

def load_datasets_nc(data_name):
    if data_name == "pokec_n":
        pokec_n = Pokec_n()
        adj, feats, labels, sens, idx_train, idx_val, idx_test = (
            pokec_n.adj("scipy.sparse"),
            pokec_n.features(),
            pokec_n.labels(),
            pokec_n.sens(),
            pokec_n.idx_train(),
            pokec_n.idx_val(),
            pokec_n.idx_test())
    elif data_name == "german":
        german = German()
        adj, feats, labels, sens, idx_train, idx_val, idx_test = (
            german.adj("scipy.sparse"),
            german.features(),
            german.labels(),
            german.sens(),
            german.idx_train(),
            german.idx_val(),
            german.idx_test())
    else:
        raise ValueError('Check dataset name!')

    g = dgl.from_scipy(adj)
    feats[feats != 0] = 1.

    # Remove columns with constant values.
    non_full_zero_feat_mask = feats.sum(dim=0) != 0
    feats = feats[:, non_full_zero_feat_mask]

    non_full_one_feat_mask = feats.sum(dim=0) != feats.size(0)
    feats = feats[:, non_full_one_feat_mask]
    
    num_nodes = g.num_nodes()
    
    train_mask = torch.zeros(num_nodes)
    val_mask = torch.zeros(num_nodes)
    test_mask = torch.zeros(num_nodes)
    
    train_mask[idx_train] = 1.
    val_mask[idx_val] = 1.
    test_mask[idx_test] = 1.
    
    g.ndata['feat'] = feats
    g.ndata['sens'] = sens
    g.ndata['label'] = labels
    g.ndata["train_mask"] = train_mask.bool()
    g.ndata["val_mask"] = val_mask.bool()
    g.ndata["test_mask"] = test_mask.bool()
    return g

    
        
def preprocess(g):
    """Prepare data for GraphMaker.

    Parameters
    ----------
    g : DGLGraph
        Graph to be preprocessed.

    Returns
    -------
    X_one_hot : torch.Tensor of shape (F, N, 2)
        X_one_hot[f, :, :] is the one-hot encoding of the f-th node attribute.
        N = |V|.
    s : torch.Tensor of shape (N)
        Categorical node labels.
    y : torch.Tensor of shape (N)
        Categorical node labels.
    E_one_hot : torch.Tensor of shape (N, N, 2)
        - E_one_hot[:, :, 0] indicates the absence of an edge.
        - E_one_hot[:, :, 1] is the original adjacency matrix.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    s_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    X_cond_s_marginals : torch.Tensor of shape (F, C, 2)
        X_cond_Y_marginals[f, k] is the marginal distribution of the f-th node
        attribute conditioned on the node label being k.
    """
    X = g.ndata['feat']
    if 'sens' in g.ndata:
        s = g.ndata['sens']
        y = g.ndata['label']
    else:
        s = g.ndata['label']
        y= None
    N = g.num_nodes()
    src, dst = g.edges()
    p_values = []
    X_one_hot_list = []
    for f in range(X.size(1)):
        # (N, 2)
        p_values.append(scipy.stats.pearsonr(X[:, f], s)[1])
        X_f_one_hot = F.one_hot(X[:, f].long())
        X_one_hot_list.append(X_f_one_hot)
    # (F, N, 2)
    X_one_hot = torch.stack(X_one_hot_list, dim=0).float()

    E = torch.zeros(N, N)
    E[dst, src] = 1.
    # (N, N, 2)
    E_one_hot = F.one_hot(E.long()).float()

    # (F, 2)
    X_one_hot_count = X_one_hot.sum(dim=1)
    # (F, 2)
    X_marginal = X_one_hot_count / X_one_hot_count.sum(dim=1, keepdim=True)
    
    if y is not None:
        # (N, C)
        y_one_hot = F.one_hot(y).float()
        # (C)
        y_one_hot_count = y_one_hot.sum(dim=0)
        # (C)
        y_marginal = y_one_hot_count / y_one_hot_count.sum() 
    else:
        y_marginal = None
    
    # (N, C)
    s_one_hot = F.one_hot(s).float()
    # (C)
    s_one_hot_count = s_one_hot.sum(dim=0)
    # (C)
    s_marginal = s_one_hot_count / s_one_hot_count.sum()

    # (2)
    E_one_hot_count = E_one_hot.sum(dim=0).sum(dim=0)
    E_marginal = E_one_hot_count / E_one_hot_count.sum()

    # P(X_f | s)
    X_cond_s_marginals = []
    num_classes = s_marginal.size(-1)
    for k in range(num_classes):
        nodes_k = s == k
        X_one_hot_k = X_one_hot[:, nodes_k]
        # (F, 2)
        X_one_hot_k_count = X_one_hot_k.sum(dim=1)
        # (F, 2)
        X_marginal_k = X_one_hot_k_count / X_one_hot_k_count.sum(dim=1, keepdim=True)
        X_cond_s_marginals.append(X_marginal_k)
    # (F, C, 2)
    X_cond_s_marginals = torch.stack(X_cond_s_marginals, dim=1)
    
    if y is not None:
        # P(X_f | y)
        X_cond_y_marginals = []
        num_classes = y_marginal.size(-1)
        for k in range(num_classes):
            nodes_k = y == k
            X_one_hot_k = X_one_hot[:, nodes_k]
            # (F, 2)
            X_one_hot_k_count = X_one_hot_k.sum(dim=1)
            # (F, 2)
            X_marginal_k = X_one_hot_k_count / X_one_hot_k_count.sum(dim=1, keepdim=True)
            X_cond_y_marginals.append(X_marginal_k)
            
        y_cond_s_marginals = []
        num_classes = s_marginal.size(-1)
        for k in range(num_classes):
            nodes_k = s ==k
            y_one_hot_k = y_one_hot[nodes_k]
            y_one_hot_k_count = y_one_hot_k.sum(dim=0)
            y_marginal_k = y_one_hot_k_count / y_one_hot_k_count.sum()
            y_cond_s_marginals.append(y_marginal_k)
            
        # (F, C, 2)
        X_cond_y_marginals = torch.stack(X_cond_y_marginals, dim=1)
        y_cond_s_marginals = torch.stack(y_cond_s_marginals, dim=1)
        
    else:
        X_cond_y_marginals = None
        y_cond_s_marginals = None
        

    return X_one_hot, s, y, E_one_hot, X_marginal, s_marginal, y_marginal, E_marginal, X_cond_s_marginals, X_cond_y_marginals, y_cond_s_marginals, p_values

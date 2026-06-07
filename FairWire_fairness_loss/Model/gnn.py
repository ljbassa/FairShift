import torch
import torch.nn as nn

__all__ = ["GNN", "LinkPredictor"]

class GNNLayerWithY(nn.Module):
    """Graph Neural Network (GNN) / Message Passing Neural Network (MPNN) Layer.

    Parameters
    ----------
    hidden_X : int
        Hidden size for the node attributes.
    hidden_s : int
        Hidden size for the sensitive attributes, s.   
    hidden_Y : int
        Hidden size for the node label.
    hidden_t : int
        Hidden size for the normalized time step.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 hidden_X,
                 hidden_Y,
                 hidden_s,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_Y + hidden_s + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.update_Y = nn.Sequential(
            nn.Linear(hidden_Y, hidden_Y),
            nn.ReLU(),
            nn.LayerNorm(hidden_Y),
            nn.Dropout(dropout)
        )
        self.update_s = nn.Sequential(
            nn.Linear(hidden_s, hidden_s),
            nn.ReLU(),
            nn.LayerNorm(hidden_s),
            nn.Dropout(dropout)
        )

    def forward(self, A, h_X, h_Y, h_s, h_t):
        """
        Parameters
        ----------
        A : dglsp.SparseMatrix
            Adjacency matrix.
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Hidden representations for the node label.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Hidden representations for the sensitive attributes.
        h_t : torch.Tensor of shape (|V|, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Updated hidden representations for the node label.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Updated hidden representations for the sensitive attributes.
        """
        h_aggr_X = A @ torch.cat([h_X, h_Y, h_s], dim=1) #Design choice: should we include s in aggregation?
        h_aggr_Y = A @ h_Y
        h_aggr_s = A @ h_s

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1) # If s is not used in aggregation, it should be concatenated here.

        h_X = self.update_X(h_aggr_X)
        h_Y = self.update_Y(h_aggr_Y)
        h_s = self.update_s(h_aggr_s)

        return h_X, h_s, h_Y

class GNNLayer(nn.Module):
    """Graph Neural Network (GNN) / Message Passing Neural Network (MPNN) Layer.

    Parameters
    ----------
    hidden_X : int
        Hidden size for the node attributes.
    hidden_s : int
        Hidden size for the sensitive attributes, s.   
    hidden_t : int
        Hidden size for the normalized time step.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 hidden_X,
                 hidden_s,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_s + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.update_s = nn.Sequential(
            nn.Linear(hidden_s, hidden_s),
            nn.ReLU(),
            nn.LayerNorm(hidden_s),
            nn.Dropout(dropout)
        )

    def forward(self, A, h_X, h_s, h_t):
        """
        Parameters
        ----------
        A : dglsp.SparseMatrix
            Adjacency matrix.
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Hidden representations for the sensitive attributes.
        h_t : torch.Tensor of shape (|V|, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Updated hidden representations for the sensitive attributes.
        """
        h_aggr_X = A @ torch.cat([h_X, h_s], dim=1) #Design choice: should we include s in aggregation?
        h_aggr_s = A @ h_s

        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_aggr_X = torch.cat([h_aggr_X, h_t_expand], dim=1) # If s is not used in aggregation, it should be concatenated here.

        h_X = self.update_X(h_aggr_X)
        h_s = self.update_s(h_aggr_s)

        return h_X, h_s
    
class GNNTower(nn.Module):
    """Graph Neural Network (GNN) / Message Passing Neural Network (MPNN).

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_Y : int
        Number of classes for node label.
    num_classes_s : int
        Number of classes for the sensitive attributes.
    hidden_t : int
        Hidden size for the normalized time step.
    hidden_X : int
        Hidden size for the node attributes.
    hidden_Y : int
        Hidden size for the node label.
    hidden_s : int
        Hidden size for the sensitive attributes.
    out_size : int
        Output size of the final MLP layer.
    num_gnn_layers : int
        Number of GNN/MPNN layers.
    dropout : float
        Dropout rate.
    class_info : bool
        Whether node labels are used in the algorithm
    node_mode: bool
        Whether the encoder is used for node attribute prediction or structure
        prediction.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_s,
                 hidden_t,
                 hidden_X,
                 hidden_s,
                 out_size,
                 num_gnn_layers,
                 dropout,
                 node_mode,
                 num_classes_Y,
                 class_info,
                 hidden_Y=None
                 ):
        super().__init__()
        self.class_info = class_info
        self.node_mode = node_mode
        in_X = num_attrs_X * num_classes_X
        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

        self.mlp_in_t = nn.Sequential(
            nn.Linear(1, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU())
        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_X, hidden_X),
            nn.ReLU(),
            nn.Linear(hidden_X, hidden_X),
            nn.ReLU()
        )
        self.emb_s = nn.Embedding(num_classes_s, hidden_s)
        if class_info:
            self.emb_Y = nn.Embedding(num_classes_Y, hidden_Y)
            self.gnn_layers = nn.ModuleList([
                GNNLayerWithY(hidden_X,
                              hidden_Y,
                              hidden_s,
                              hidden_t,
                              dropout) 
                for _ in range(num_gnn_layers)])

            # +1 for the input attributes 
            hidden_cat = (num_gnn_layers + 1) * (hidden_X + hidden_Y + hidden_s) + hidden_t #change based on the use of s
        else: 
            self.gnn_layers = nn.ModuleList([
                GNNLayer(hidden_X,
                         hidden_s,
                         hidden_t,
                         dropout)
                for _ in range(num_gnn_layers)])
            hidden_cat = (num_gnn_layers + 1) * (hidden_X + hidden_s) + hidden_t #change based on the use of s
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, out_size))
            
    def forward(self,
                t_float,
                X_t_one_hot,
                A_t,
                s_real,
                Y_real):
        # Input projection.
        # (1, hidden_t)
        h_t = self.mlp_in_t(t_float).unsqueeze(0)
        h_X = self.mlp_in_X(X_t_one_hot)
        h_s = self.emb_s(s_real)
        
        h_X_list = [h_X]
        h_s_list = [h_s]
        if self.class_info:
            h_Y = self.emb_Y(Y_real)
            h_Y_list = [h_Y]
            for gnn in self.gnn_layers:
                h_X, h_s, h_Y = gnn(A_t, h_X, h_Y, h_s, h_t)
                h_X_list.append(h_X)
                h_s_list.append(h_s)
                h_Y_list.append(h_Y)
            h_t = h_t.expand(h_X.size(0), -1)
            h_cat = torch.cat(h_X_list + h_Y_list + h_s_list + [h_t], dim=1)
            if self.node_mode:
                # (|V|, F * C_X)
                logit = self.mlp_out(h_cat)
                # (|V|, F, C_X)
                logit = logit.reshape(Y_real.size(0), self.num_attrs_X, -1)
                return logit
            else:
                return self.mlp_out(h_cat)
            
        else:
            for gnn in self.gnn_layers:
                h_X, h_s = gnn(A_t, h_X, h_s, h_t)
                h_X_list.append(h_X)
                h_s_list.append(h_s)
            h_t = h_t.expand(h_X.size(0), -1)
            h_cat = torch.cat(h_X_list + h_s_list + [h_t], dim=1)
            if self.node_mode:
                # (|V|, F * C_X)
                logit = self.mlp_out(h_cat)
                # (|V|, F, C_X)
                logit = logit.reshape(s_real.size(0), self.num_attrs_X, -1)
                return logit
            else:
                return self.mlp_out(h_cat)

        
            

class LinkPredictor(nn.Module):
    """Model for structure prediction.

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_Y : int
        Number of classes for node label.
    num_classes_s : int
        Number of classes for sensitive attributes.
    num_classes_E : int
        Number of edge classes.
    hidden_t : int
        Hidden size for the normalized time step.
    hidden_X : int
        Hidden size for the node attributes.
    hidden_Y : int
        Hidden size for the node label.
    hidden_s : int
        Hidden size for the sensitive attributes.    
    hidden_E : int
        Hidden size for the edges.
    num_gnn_layers : int
        Number of GNN/MPNN layers.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_E,
                 num_classes_s,
                 hidden_t,
                 hidden_X,
                 hidden_s,
                 hidden_E,
                 num_gnn_layers,
                 dropout,
                 num_classes_Y,
                 hidden_Y=None):
        super().__init__()
        if num_classes_Y is not None:
            class_info = True
        else:
            class_info = False
        self.gnn_encoder = GNNTower(num_attrs_X,
                                    num_classes_X,
                                    num_classes_s,
                                    hidden_t,
                                    hidden_X,
                                    hidden_s,
                                    hidden_E,
                                    num_gnn_layers,
                                    dropout,
                                    num_classes_Y=num_classes_Y,
                                    hidden_Y=hidden_Y,
                                    class_info=class_info,
                                    node_mode=False)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_E, hidden_E),
            nn.ReLU(),
            nn.Linear(hidden_E, num_classes_E)
        )

    def forward(self,
                t_float,
                X_t_one_hot,
                A_t,
                s_real,
                Y_real,
                src,
                dst):
        # (|V|, hidden_E)
        h = self.gnn_encoder(t_float,
                             X_t_one_hot,
                             A_t,
                             s_real,
                             Y_real)
        # (|E|, hidden_E)
        h = h[src] * h[dst]
        # (|E|, num_classes_E)
        logit = self.mlp_out(h)

        return logit

class GNN(nn.Module):
    """P(X|Y, X^t, A^t) + P(A|Y, X^t, A^t)

    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    num_classes_s : int
        Number of classes for the sensitive attributes.
    num_classes_Y : int
        Number of classes for node label.
    num_classes_E : int
        Number of edge classes.
    gnn_X_config : dict
        Configuration of the GNN for reconstructing node attributes.
    gnn_E_config : dict
        Configuration of the GNN for reconstructing edges.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_E,
                 num_classes_s,
                 num_classes_Y,
                 gnn_X_config,
                 gnn_E_config
                 ):
        super().__init__()
        
        if num_classes_Y is None:
            class_info = False
        else:
            class_info = True 
            
        
        self.pred_X = GNNTower(num_attrs_X,
                               num_classes_X,
                               num_classes_s = num_classes_s,
                               num_classes_Y=num_classes_Y,
                               out_size=num_attrs_X * num_classes_X,
                               class_info=class_info,
                               node_mode=True,
                               **gnn_X_config)

        self.pred_E = LinkPredictor(num_attrs_X,
                                    num_classes_X,
                                    num_classes_s = num_classes_s,
                                    num_classes_Y=num_classes_Y,
                                    num_classes_E=num_classes_E,
                                    **gnn_E_config)

    def forward(self,
                t_float,
                X_t_one_hot,
                A_t,
                s_real,
                Y,
                batch_src,
                batch_dst):
        """
        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels.
        A_t : dglsp.SparseMatrix
            Row-normalized sampled adjacency matrix.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).

        Returns
        -------
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence.
        """
        logit_X = self.pred_X(t_float,
                              X_t_one_hot,
                              A_t,
                              s_real,
                              Y)

        logit_E = self.pred_E(t_float,
                              X_t_one_hot,
                              A_t,
                              s_real,
                              Y,
                              batch_src,
                              batch_dst)

        return logit_X, logit_E
class MLPLayerWithY(nn.Module):
    """
    Parameters
    ----------
    hidden_X : int
        Hidden size for the node attributes.
    hidden_s : int
        Hidden size of the sensitive attributes. 
    hidden_Y : int
        Hidden size for the node labels.
    hidden_t : int
        Hidden size for the normalized time step.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 hidden_X,
                 hidden_Y,
                 hidden_s,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_Y + hidden_s + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.update_Y = nn.Sequential(
            nn.Linear(hidden_Y, hidden_Y),
            nn.ReLU(),
            nn.LayerNorm(hidden_Y),
            nn.Dropout(dropout)
        )
        self.update_s = nn.Sequential(
            nn.Linear(hidden_s, hidden_s),
            nn.ReLU(),
            nn.LayerNorm(hidden_s),
            nn.Dropout(dropout)
        )

    def forward(self, h_X, h_Y, h_s, h_t):
        """
        Parameters
        ----------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Hidden representations for the node labels.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Hidden representations for the sensitive attributes.
        h_t : torch.Tensor of shape (1, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Updated hidden representations for the sensitive attributes.
        h_Y : torch.Tensor of shape (|V|, hidden_Y)
            Updated hidden representations for the node labels.
        """
        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_X = torch.cat([h_X, h_Y, h_s, h_t_expand], dim=1)

        h_X = self.update_X(h_X)
        h_Y = self.update_Y(h_Y)
        h_s = self.update_s(h_s)

        return h_X, h_s, h_Y

class MLPLayer(nn.Module):
    """
    Parameters
    ----------
    hidden_X : int
        Hidden size for the node attributes.
    hidden_s : int
        Hidden size for the sensitive attributes.
    hidden_t : int
        Hidden size for the normalized time step.
    dropout : float
        Dropout rate.
    """
    def __init__(self,
                 hidden_X,
                 hidden_s,
                 hidden_t,
                 dropout):
        super().__init__()

        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_s + hidden_t, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        self.update_s = nn.Sequential(
            nn.Linear(hidden_s, hidden_s),
            nn.ReLU(),
            nn.LayerNorm(hidden_s),
            nn.Dropout(dropout)
        )

    def forward(self, h_X, h_s, h_t):
        """
        Parameters
        ----------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Hidden representations for the node attributes.
        h_s : torch.Tensor of shape (|V|, hidden_s)
            Hidden representations for the sensitive attributes.
        h_t : torch.Tensor of shape (1, hidden_t)
            Hidden representations for the normalized time step.

        Returns
        -------
        h_X : torch.Tensor of shape (|V|, hidden_X)
            Updated hidden representations for the node attributes.
        h_s : torch.Tensor of shape (|V|, hidden_Y)
            Updated hidden representations for the node labels.
        """
        num_nodes = h_X.size(0)
        h_t_expand = h_t.expand(num_nodes, -1)
        h_X = torch.cat([h_X, h_s, h_t_expand], dim=1)

        h_X = self.update_X(h_X)
        h_s = self.update_s(h_s)
        return h_X, h_s

class MLPTower(nn.Module):
    def __init__(self,
                 num_attrs_X,
                 num_classes_X,
                 num_classes_s,
                 hidden_t,
                 hidden_X,
                 hidden_s,
                 num_mlp_layers,
                 dropout,
                 num_classes_Y=None,
                 hidden_Y=None,
                 class_info=False
                ):
        super().__init__()
        self.class_info = class_info
        in_X = num_attrs_X * num_classes_X
        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

        self.mlp_in_t = nn.Sequential(
            nn.Linear(1, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU())
        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_X, hidden_X),
            nn.ReLU(),
            nn.Linear(hidden_X, hidden_X),
            nn.ReLU()
        )
        self.emb_s = nn.Embedding(num_classes_s, hidden_s)
        if class_info:
            self.emb_Y = nn.Embedding(num_classes_Y, hidden_Y)

            self.mlp_layers = nn.ModuleList([
                MLPLayerWithY(hidden_X,
                              hidden_Y,
                              hidden_s,
                              hidden_t,
                              dropout)
                for _ in range(num_mlp_layers)])
            hidden_cat = (num_mlp_layers + 1) * (hidden_X + hidden_Y + hidden_s) + hidden_t
            
        else: 
            hidden_cat = (num_mlp_layers + 1) * (hidden_X + hidden_s) + hidden_t
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, in_X)
        )
    def forward(self,
                t_float,
                X_t_one_hot,
                s_real,
                Y_real=None):
        # Input projection.
        h_t = self.mlp_in_t(t_float).unsqueeze(0)
        h_X = self.mlp_in_X(X_t_one_hot)
        h_s = self.emb_s(s_real)
        h_X_list = [h_X]
        h_s_list = [h_s]
        h_t = h_t.expand(h_X.size(0), -1)
        if self.class_info:
            h_Y = self.emb_Y(Y_real)
            h_Y_list = [h_Y]
            for mlp in self.mlp_layers:
                h_X, h_s, h_Y = mlp(h_X, h_Y, h_s, h_t)
                h_X_list.append(h_X)
                h_Y_list.append(h_Y)
                h_s_list.append(h_s)
            h_cat = torch.cat(h_X_list + h_Y_list + h_s_list + [h_t], dim=1)

            logit = self.mlp_out(h_cat)
        # (|V|, F, C)
            logit = logit.reshape(Y_real.size(0), self.num_attrs_X, -1)
        else:
            for mlp in self.mlp_layers:
                h_X, h_s= mlp(h_X,  h_s, h_t)
                h_X_list.append(h_X)
                h_s_list.append(h_s)
            h_cat = torch.cat(h_X_list + h_s_list + [h_t], dim=1)

            logit = self.mlp_out(h_cat)
        # (|V|, F, C)
            logit = logit.reshape(h_X.size(0), self.num_attrs_X, -1)

        return logit

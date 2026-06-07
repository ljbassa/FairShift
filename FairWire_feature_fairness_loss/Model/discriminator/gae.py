import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .base import BaseTrainer
from ..common_blocks import GAE

class GAETrainer(BaseTrainer):
    def __init__(self, num_gnn_layers):
        hyper_space = {
            "lr": [3e-2, 1e-2, 3e-3, 1e-3],
            "num_layers": [num_gnn_layers],
            "hidden_size": [16, 32, 128, 512],
            "dropout": [0., 0.1, 0.2]
        }
        search_priority_increasing = ["dropout", "lr", "num_layers", "hidden_size"]

        super().__init__(hyper_space=hyper_space,
                         search_priority_increasing=search_priority_increasing,
                         patience=5)

        self.num_gnn_layers = num_gnn_layers

    def preprocess(self, A_train, A_full, X, s, Y):
        A_train = A_train.to(self.device)
        A_full = A_full.to(self.device)
        X = X.to(self.device).float()
        #X = F.normalize(X, p=1, dim=1)
        s = s.to(self.device)
        s = F.one_hot(s.long(), len(torch.unique(s)))
        if Y!=None:
            Y = Y.to(self.device)
            Y = F.one_hot(Y.long(), self.num_classes)
            Z = torch.cat([X, s, Y], dim=1)
        else:
            Z = X#torch.cat([X, s], dim=1)
        # row normalize
        
        

        A_full_dense = A_full.to_dense()
        A_full_dense[A_full_dense != 0] = 1.

        return A_train, Z, A_full_dense

    def _safe_group_mean(self, values, mask):
        denom = np.sum(mask)
        if denom == 0:
            return np.nan
        return float(np.sum(values[mask]) / denom)

    def fairness_metrics(self, labels, preds, pair_same_mask):
        idx_same = np.asarray(pair_same_mask, dtype=bool)
        idx_diff = ~idx_same

        idx_same_y1 = np.bitwise_and(idx_same, labels == 1)
        idx_diff_y1 = np.bitwise_and(idx_diff, labels == 1)

        parity = abs(self._safe_group_mean(preds, idx_same) - self._safe_group_mean(preds, idx_diff))
        equality = abs(self._safe_group_mean(preds, idx_same_y1) - self._safe_group_mean(preds, idx_diff_y1))
        return parity, equality

    @torch.no_grad()
    def predict_fit(self, A, Z, group_labels, A_dense, mask, model):
        model.eval()
        Z_out = model(A, Z)
        group_labels = group_labels.to(Z_out.device)
        pair_same = (group_labels.unsqueeze(1) == group_labels.unsqueeze(0))[mask].cpu().numpy()
        prob = torch.sigmoid(Z_out @ Z_out.T)[mask].cpu().numpy()
        
        label = A_dense[mask].cpu().numpy()
        
        sp, eo = self.fairness_metrics(label, prob, pair_same)
        return roc_auc_score(label, prob), sp, eo

    @torch.no_grad()
    def predict(self, A_train, s, A_full, X, Y, mask):
        A_train, Z, A_full_dense = self.preprocess(
            A_train, A_full, X, s, Y)

        model = self.model
        model.eval()
        Z_out = model(A_train, Z)
        group_labels = Y if Y is not None else s
        group_labels = group_labels.to(Z_out.device)
        pair_same = (group_labels.unsqueeze(1) == group_labels.unsqueeze(0))[mask].cpu().numpy()
        prob = torch.sigmoid(Z_out @ Z_out.T)[mask].cpu().numpy()
        label = A_full_dense[mask].cpu().numpy()

        sp, eo = self.fairness_metrics(label, prob, pair_same)
        return roc_auc_score(label, prob), sp, eo
    def fit_trial(self,
                  A_train,
                  Z,
                  group_labels,
                  A_full_dense,
                  train_mask,
                  val_mask,
                  num_layers,
                  hidden_size,
                  dropout,
                  lr):

        model = GAE(in_size=Z.size(1),
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    dropout=dropout).to(self.device)
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 1000
        num_patient_epochs = 0
        best_auc = 0
        best_model_state_dict = deepcopy(model.state_dict())

        num_nodes = Z.size(0)
        train_dst, train_src = train_mask.nonzero().T
        train_size = len(train_dst)

        batch_size = 16384
        for epoch in range(1, num_epochs + 1):
            model.train()

            Z_out = model(A_train, Z)

            if train_size <= batch_size:
                batch_dst = train_dst
                batch_src = train_src
            else:
                batch_ids = torch.randint(low=0, high=train_size, size=(batch_size,),
                                          device=self.device)
                batch_dst = train_dst[batch_ids]
                batch_src = train_src[batch_ids]

            pos_pred = (Z_out[batch_src] * Z_out[batch_dst]).sum(dim=-1)

            real_batch_size = len(batch_dst)
            neg_src = torch.randint(0, num_nodes, (real_batch_size,),
                                    device=self.device)
            neg_dst = torch.randint(0, num_nodes, (real_batch_size,),
                                    device=self.device)
            neg_pred = (Z_out[neg_src] * Z_out[neg_dst]).sum(dim=-1)

            pred = torch.cat([pos_pred, neg_pred], dim=0)
            label = torch.cat([torch.ones(real_batch_size),
                               torch.zeros(real_batch_size)], dim=0).to(self.device)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            auc, sp, eo = self.predict_fit(A_train, Z, group_labels, A_full_dense, val_mask, model)

            if auc > best_auc:
                num_patient_epochs = 0
                best_auc = auc
                best_sp = sp
                best_eo = eo
                best_model_state_dict = deepcopy(model.state_dict())
            else:
                num_patient_epochs += 1

            if num_patient_epochs == self.patience:
                break

        model.load_state_dict(best_model_state_dict)
        return best_auc, best_sp, best_eo, model

    def fit(self, A_train, A_full, X, s, Y, num_classes,
            train_mask, val_mask):
        """
        Parameters
        ----------
        A_train : dgl.sparse.SparseMatrix
            Training adjacency matrix.
        A_full : dgl.sparse.SparseMatrix
            Full adjacency matrix.
        X : torch.Tensor of shape (|V|, D)
            Binary node features.
        Y : torch.Tensor of shape (|V|,)
            Node labels.
        num_classes : int
            Number of node classes.
        train_mask : torch.Tensor of shape (|V|, |V|)
            Mask indicating training edges.
        val_mask : torch.Tensor of shape (|V|, |V|)
            Mask indicating validation edges.
        """
        self.num_classes = num_classes
        A_train, Z, A_full_dense = self.preprocess(
            A_train, A_full, X, s, Y)
        group_labels = Y if Y is not None else s

        config_list = self.get_config_list()

        best_auc = 0
        with tqdm(config_list) as tconfig:
            tconfig.set_description(
                f"Training GAE {self.num_gnn_layers}-layer discriminator")

            for config in tconfig:
                trial_auc, trial_sp, trial_eo, trial_model = self.fit_trial(A_train,
                                                                            Z,
                                                                            group_labels,
                                                                            A_full_dense,
                                                                            train_mask,
                                                                            val_mask,
                                                                            **config)

                if trial_auc > best_auc:
                    best_auc = trial_auc
                    best_sp = trial_sp
                    best_eo = trial_eo
                    best_model = trial_model
                    best_model_config = {
                        "in_size": Z.size(1),
                        "num_layers": config["num_layers"],
                        "hidden_size": config["hidden_size"],
                        "dropout": config["dropout"]
                    }

                tconfig.set_postfix(roc_auc=100. * best_auc)
                tconfig.set_postfix({'SP': best_sp, 'EO': best_eo})
                if trial_auc == 1.0:
                    break
        self.model = best_model
        self.best_model_config = best_model_config

   

    def save_model(self, model_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.best_model_config,
            "num_classes": self.num_classes
        }, model_path)

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        model = GAE(**state_dict["model_config"]).to(self.device)
        model.load_state_dict(state_dict["model_state_dict"])
        self.model = model
        self.num_classes = state_dict["num_classes"]

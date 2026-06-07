import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
class CN(nn.Module):
    def __init__(self, batch_size = 65536):
        super().__init__()

        self.best_threshold = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.batch_size = batch_size

    def fit(self, A_train, A_full, val_mask):
        A_train = A_train.to_dense()
        A_full = A_full.to_dense()

        val_src, val_dst = val_mask.nonzero().T
        label = A_full[val_src, val_dst]
        label[label != 0] = 1
        label = label.cpu()

        A_train[A_train != 0.] = 1.

        num_batches = len(val_src) // self.batch_size
        if len(val_src) % self.batch_size != 0:
            num_batches += 1

        start = 0
        pred_list = []
        for i in range(num_batches):
            end = start + self.batch_size

            batch_src = val_src[start:end]
            batch_dst = val_dst[start:end]
            batch_pred = (A_train[batch_src] * A_train[batch_dst]).sum(dim=-1)
            batch_pred = batch_pred.cpu()
            pred_list.append(batch_pred)

            start = end

        pred = torch.cat(pred_list)

        thresholds = pred.unique()
        acc_list = []
        for bar in thresholds:
            pred_bar = (pred >= bar).float()
            acc_bar = (pred_bar == label).float().mean()
            acc_list.append(acc_bar.item())

        self.best_threshold = nn.Parameter(
            thresholds[np.argmax(acc_list)], requires_grad=False)
    
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
    
        
    def predict(self, A_train, A_full, s, mask, Y=None):
        A_train = A_train.to_dense()
        A_full = A_full.to_dense()

        src, dst = mask.nonzero().T
        label = A_full[src, dst]
        label[label != 0] = 1
        label = label.cpu()

        group_labels = Y if Y is not None else s
        pair_same = (group_labels[src] == group_labels[dst]).cpu().numpy()
        A_train[A_train != 0.] = 1.

        num_batches = len(src) // self.batch_size
        if len(src) % self.batch_size != 0:
            num_batches += 1

        start = 0
        pred_list = []
        for i in range(num_batches):
            end = start + self.batch_size

            batch_src = src[start:end]
            batch_dst = dst[start:end]
            batch_pred = (A_train[batch_src] * A_train[batch_dst]).sum(dim=-1)

            batch_pred = batch_pred.cpu()
            batch_pred = (batch_pred >= self.best_threshold).float()
            pred_list.append(batch_pred)

            start = end

        pred = torch.cat(pred_list)
        with torch.no_grad():
            sp, eo = self.fairness_metrics(label.numpy(), pred.cpu().numpy(), pair_same)
        return (pred == label).float().mean().item(), sp, eo

class CNEvaluator:
    def __init__(self,
                 model_path,
                 A_train,
                 s,
                 Y,
                 A_full,
                 val_mask,
                 test_mask):
        self.real_A_train = A_train
        self.real_A_full = A_full
        self.real_s = s
        self.real_Y = Y
        self.real_test_mask = test_mask

        self.sample_sample_acc = []

        self.model_real = CN()
        if os.path.exists(model_path):
            self.model_real.load_state_dict(torch.load(model_path))
        else:
            self.model_real.fit(A_train, A_full, val_mask)
            torch.save(self.model_real.state_dict(), model_path)

        self.real_real_acc, self.real_real_sp, self.real_real_eo = self.model_real.predict(A_train, A_full, s, test_mask, Y=Y)

        self.real_sample_acc = []
        self.sample_real_acc = []
        self.sample_sample_acc = []
        
        self.real_sample_sp = []
        self.sample_real_sp = []
        self.sample_sample_sp = []
        
        self.real_sample_eo = []
        self.sample_real_eo = []
        self.sample_sample_eo = []

    def add_sample(self,
                   A_train,
                   s,
                   Y,
                   A_full,
                   val_mask,
                   test_mask):
        
        real_sample_acc, real_sample_sp, real_sample_eo = self.model_real.predict(A_train, A_full, s, test_mask, Y=Y)
        self.real_sample_acc.append(real_sample_acc)
        self.real_sample_sp.append(real_sample_sp)
        self.real_sample_eo.append(real_sample_eo)

        model_sample = CN()
        model_sample.fit(A_train, A_full, val_mask)
        
        sample_real_acc, sample_real_sp, sample_real_eo = model_sample.predict(
                self.real_A_train,
                self.real_A_full,
                self.real_s,
                self.real_test_mask,
                Y=self.real_Y)
        self.sample_real_acc.append(sample_real_acc)
        self.sample_real_sp.append(sample_real_sp)
        self.sample_real_eo.append(sample_real_eo)
        
        
        sample_sample_acc, sample_sample_sp, sample_sample_eo = model_sample.predict(A_train, A_full, s, test_mask, Y=Y)
        self.sample_sample_acc.append(sample_sample_acc)
        self.sample_sample_sp.append(sample_sample_sp)
        self.sample_sample_eo.append(sample_sample_eo)

    def summary(self):
        print(f"ACC/AUC(G|G): {self.real_real_acc}")
        print(f"SP(G|G): {self.real_real_sp}")
        print(f"EO(G|G): {self.real_real_eo}")
        
        mean_sample_real_acc = np.mean(self.sample_real_acc)
        mean_sample_real_sp = np.mean(self.sample_real_sp)
        mean_sample_real_eo = np.mean(self.sample_real_eo)
        print(f"ACC/AUC(G|G_hat): {mean_sample_real_acc}")
        print(f"SP(G|G_hat): {mean_sample_real_sp}")
        print(f"EO(G|G_hat): {mean_sample_real_eo}")

        mean_sample_sample_acc = np.mean(self.sample_sample_acc)
        mean_sample_sample_sp = np.mean(self.sample_sample_sp)
        mean_sample_sample_eo = np.mean(self.sample_sample_eo)
        print(f"ACC/AUC(G_hat|G_hat): {mean_sample_sample_acc}")
        print(f"SP(G_hat|G_hat): {mean_sample_sample_sp}")
        print(f"EO(G_hat|G_hat): {mean_sample_sample_eo}")
        
        mean_real_sample_acc = np.mean(self.real_sample_acc)
        mean_real_sample_sp = np.mean(self.real_sample_sp)
        mean_real_sample_eo = np.mean(self.real_sample_eo)
        print(f"ACC/AUC(G_hat|G): {mean_real_sample_acc}")
        print(f"SP(G_hat|G): {mean_real_sample_sp}")
        print(f"EO(G_hat|G): {mean_real_sample_eo}")

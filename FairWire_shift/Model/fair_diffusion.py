import dgl.sparse as dglsp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from .gnn import *

__all__ = ["ModelSync"]

class MarginalTransitionForX(nn.Module):
    """
    Parameters
    ----------
    device : torch.device
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
      """
    def __init__(self,
                 device,
                 X_marginal):
        super().__init__()

        num_attrs_X, num_classes_X = X_marginal.shape
        # (F, 2, 2)
        self.I_X = torch.eye(num_classes_X, device=device).unsqueeze(0).expand(
            num_attrs_X, num_classes_X, num_classes_X).clone()

        # (F, 2, 2)
        self.m_X = X_marginal.unsqueeze(1).expand(
            num_attrs_X, num_classes_X, -1).clone()
       
        self.I_X = nn.Parameter(self.I_X, requires_grad=False)
        self.m_X = nn.Parameter(self.m_X, requires_grad=False)
    def get_Q_bar_X(self, alpha_bar_t):
        """Compute the probability transition matrices for obtaining X^t.

        Parameters
        ----------
        alpha_bar_t : torch.Tensor of shape (1)
            A value in [0, 1].

        Returns
        -------
        Q_bar_t_X : torch.Tensor of shape (F, 2, 2)
            Transition matrix for corrupting node attributes at time step t.
        """
        Q_bar_t_X = alpha_bar_t * self.I_X + (1 - alpha_bar_t) * self.m_X

        return Q_bar_t_X


class MarginalTransitionForA(nn.Module):
    """
    Parameters
    ----------
    device : torch.device
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    num_classes_E : int
        Number of edge classes.
    """
    def __init__(self,
                 device,
                 E_marginal,
                 num_classes_E):
        super().__init__()

        # (2, 2)
        self.I_E = torch.eye(num_classes_E, device=device)
        # (2, 2)
        self.m_E = E_marginal.unsqueeze(0).expand(num_classes_E, -1).clone()
        self.I_E = nn.Parameter(self.I_E, requires_grad=False)
        self.m_E = nn.Parameter(self.m_E, requires_grad=False)
    def get_Q_bar_E(self, alpha_bar_t):
        """Compute the probability transition matrices for obtaining A^t.

        Parameters
        ----------
        alpha_bar_t : torch.Tensor of shape (1)
            A value in [0, 1].

        Returns
        -------
        Q_bar_t_E : torch.Tensor of shape (2, 2)
            Transition matrix for corrupting graph structure at time step t.
        """
        Q_bar_t_E = alpha_bar_t * self.I_E + (1 - alpha_bar_t) * self.m_E

        return Q_bar_t_E

class NoiseSchedule(nn.Module):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps.
    device : torch.device
    s : float
        Small constant for numerical stability.
    """
    def __init__(self, T, device, s=0.008):
        super().__init__()

        # Cosine schedule as proposed in
        # https://arxiv.org/abs/2102.09672
        num_steps = T + 2
        t = np.linspace(0, num_steps, num_steps)
        # Schedule for \bar{alpha}_t = alpha_1 * ... * alpha_t
        alpha_bars = np.cos(0.5 * np.pi * ((t / num_steps) + s) / (1 + s)) ** 2
        # Make the largest value 1.
        alpha_bars = alpha_bars / alpha_bars[0]
        alphas = alpha_bars[1:] / alpha_bars[:-1]

        self.betas = torch.from_numpy(1 - alphas).float().to(device)
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)

        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(self.alpha_bars, requires_grad=False)

class LossE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true_E, logit_E):
        """
        Parameters
        ----------
        true_E : torch.Tensor of shape (B, 2)
            One-hot encoding of the edge existence for a batch of node pairs.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence.

        Returns
        -------
        loss_E : torch.Tensor
            Scalar representing the loss for edge existence.
        """
        true_E = torch.argmax(true_E, dim=-1)    # (B)
        loss_E = F.cross_entropy(logit_E, true_E)

        return loss_E
    
class FairLossE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit_E, M_chi, M_omega):
        """
        Parameters
        ----------
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence.
        M_chi: torch.Tensor of shape (B)
            Weight mask for one pair group.
        M_omega: torch.Tensor of shape (B)
            Weight mask for the complementary pair group.

        Returns
        -------
        fair_loss_E : torch.Tensor
            Scalar representing the fairness loss between pair groups.
        """
        
        fair_loss_E = torch.pow(torch.sum(logit_E[:,1] * M_chi) - torch.sum(logit_E[:,1] * M_omega),2)

        return fair_loss_E
        

class BaseModel(nn.Module):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps - 1.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    num_nodes : int
        Number of nodes in the original graph.
    """
    def __init__(self,
                 T,
                 X_marginal,
                 s_marginal,
                 y_marginal,
                 E_marginal,
                 num_nodes):
        super().__init__()

        device = E_marginal.device
        # 2 for if edge exists or not.
        self.num_classes_E = 2
        self.num_attrs_X, self.num_classes_X = X_marginal.shape
        self.transition_X = MarginalTransitionForX(device, X_marginal)
        self.transition_A = MarginalTransitionForA(device, E_marginal, self.num_classes_E)

        self.T = T
        # Number of intermediate time steps to use for validation.
        self.num_denoise_match_samples = self.T
        self.noise_schedule = NoiseSchedule(T, device)

        self.num_nodes = num_nodes
        self.s_marginal = s_marginal
        self.y_marginal = y_marginal 
        self.X_marginal = X_marginal
        self.E_marginal = E_marginal

        self.loss_E = LossE()
        self.fair_loss_E = FairLossE()

    def sample_E(self, prob_E):
        """Sample a graph structure from prob_E.

        Parameters
        ----------
        prob_E : torch.Tensor of shape (|V|, |V|, 2)
            Probability distribution for edge existence.

        Returns
        -------
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled symmetric adjacency matrix.
        """
        # (|V|^2, 1)
        E_t = prob_E.reshape(-1, prob_E.size(-1)).multinomial(1)

        # (|V|, |V|)
        num_nodes = prob_E.size(0)
        E_t = E_t.reshape(num_nodes, num_nodes)
        # Make it symmetric for undirected graphs.
        src, dst = torch.triu_indices(
            num_nodes, num_nodes, device=E_t.device)
        E_t[dst, src] = E_t[src, dst]
        return E_t

    def sample_X(self, prob_X):
        """Sample node attributes from prob_X.

        Parameters
        ----------
        prob_X : torch.Tensor of shape (F, |V|, 2)
            Probability distributions for node attributes.

        Returns
        -------
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        """
        # (F * |V|)
        X_t = prob_X.reshape(-1, prob_X.size(-1)).multinomial(1)
        # (F, |V|)
        X_t = X_t.reshape(self.num_attrs_X, -1)
        # (|V|, 2 * F)
        X_t_one_hot = torch.cat([
            F.one_hot(X_t[i], num_classes=self.num_classes_X)
            for i in range(self.num_attrs_X)
        ], dim=1).float()
        return X_t_one_hot

    def get_adj(self, E_t):
        """
        Parameters
        ----------
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled symmetric adjacency matrix.

        Returns
        -------
        dglsp.SparseMatrix
            Row-normalized adjacency matrix.
        """
        # Row normalization.
        edges_t = E_t.nonzero().T
        num_nodes = E_t.size(0)
        A_t = dglsp.spmatrix(edges_t, shape=(num_nodes, num_nodes))
        D_t = dglsp.diag(A_t.sum(1)) ** -1
        return D_t @ A_t

    def _sp_eta_at_step(self, base_eta, t, schedule):
        """Return the post-hoc SP logit-shift strength for reverse step t."""
        base_eta = float(base_eta)
        if base_eta <= 0.0:
            return 0.0

        progress = float(t) / max(float(self.T), 1.0)
        progress = min(max(progress, 0.0), 1.0)
        schedule = (schedule or "constant").lower()

        if schedule == "constant":
            return base_eta
        if schedule == "early":
            return base_eta * progress
        if schedule == "late":
            return base_eta * (1.0 - progress)

        raise ValueError(
            f"Unknown sp_eta_schedule={schedule}. "
            "Expected one of: constant, early, late.")

    def _clear_score_sp_state(self):
        for name in (
            "_fair_edge_sensitive_mask",
            "_fair_N1",
            "_fair_N0",
            "_fair_score_h",
            "_fair_score_q",
            "_fair_score_R1",
            "_fair_score_R0",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _shift_binary_log_probs_from_pos_logit(self, pos_logit):
        """Return EDGE_fairness-style [log p(0), log p(1)] from z=logit1-logit0."""
        log_p1 = F.logsigmoid(pos_logit)
        log_p0 = F.logsigmoid(-pos_logit)
        return torch.stack([log_p0, log_p1], dim=1)

    def _init_score_sp_state(self, edge_group_labels, E_t):
        """Initialize the EDGE_fairness score-SP state for FairWire's full edge set."""
        device = E_t.device
        dtype = self.E_marginal.dtype
        num_edges = self.src.numel()

        if edge_group_labels is None:
            mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        else:
            labels = edge_group_labels.reshape(-1).to(device)
            mask = (labels[self.src] == labels[self.dst]).bool()

        mask_float = mask.to(dtype=dtype)
        inv_mask_float = (~mask).to(dtype=dtype)
        n1 = mask_float.sum()
        n0 = inv_mask_float.sum()

        denom = float(max(self.num_nodes * (self.num_nodes - 1), 1))
        rho = E_t.to(dtype=dtype).sum() / denom
        rho = rho.clamp(1e-4, 1.0 - 1e-4)
        rho_e = rho.expand(num_edges).clone()

        self._fair_edge_sensitive_mask = mask
        self._fair_N1 = n1
        self._fair_N0 = n0
        self._fair_score_h = torch.logit(rho_e)
        self._fair_score_q = rho_e.clone()
        self._fair_score_R1 = (self._fair_score_q * mask_float).sum()
        self._fair_score_R0 = (self._fair_score_q * inv_mask_float).sum()

    def _score_sp_guidance_terms(self, logit_E, edge_ids, sp_k):
        """Compute score-SP guidance terms without mutating sampler state."""
        device = logit_E.device
        dtype = logit_E.dtype
        edge_ids = edge_ids.to(device=device)

        mask_active = self._fair_edge_sensitive_mask[edge_ids].to(device=device)
        mask_float = mask_active.to(dtype=dtype)
        inv_mask_float = (~mask_active).to(dtype=dtype)

        h_prev = self._fair_score_h[edge_ids].to(device=device, dtype=dtype)
        q_prev = self._fair_score_q[edge_ids].to(device=device, dtype=dtype)
        n1 = self._fair_N1.to(device=device, dtype=dtype)
        n0 = self._fair_N0.to(device=device, dtype=dtype)
        r1 = self._fair_score_R1.to(device=device, dtype=dtype)
        r0 = self._fair_score_R0.to(device=device, dtype=dtype)

        k = torch.as_tensor(float(sp_k), device=device, dtype=dtype)
        z_raw = logit_E[:, 1] - logit_E[:, 0]

        h_cand = h_prev + k * (z_raw - h_prev)
        q_cand = torch.sigmoid(h_cand)

        delta_q_cand = q_cand - q_prev
        c1 = (delta_q_cand * mask_float).sum()
        c0 = (delta_q_cand * inv_mask_float).sum()
        safe_n1 = torch.where(n1 > 0, n1, torch.ones_like(n1))
        safe_n0 = torch.where(n0 > 0, n0, torch.ones_like(n0))

        r1_new = r1 + c1
        r0_new = r0 + c0
        delta_sp = (r1_new / safe_n1) - (r0_new / safe_n0)

        a_e = mask_float / safe_n1 - inv_mask_float / safe_n0
        step_scale = 0.5 * (n1 + n0)
        a_bar = step_scale * a_e
        grad_raw = delta_sp * a_bar * k * q_cand * (1.0 - q_cand)

        return {
            "edge_ids": edge_ids,
            "mask_float": mask_float,
            "inv_mask_float": inv_mask_float,
            "h_prev": h_prev,
            "q_prev": q_prev,
            "n1": n1,
            "n0": n0,
            "z_raw": z_raw,
            "c1": c1,
            "c0": c0,
            "delta_sp": delta_sp,
            "a_bar": a_bar,
            "step_scale": step_scale,
            "grad_raw": grad_raw,
        }

    def _estimate_score_sp_guidance_scale(self,
                                          edge_data_loader,
                                          pred_E_func,
                                          t_float,
                                          X_t_one_hot,
                                          A_t,
                                          s_0,
                                          Y_0,
                                          sp_k,
                                          device):
        """Estimate graph-level mean(abs(raw SP gradient)) for one reverse step."""
        dtype = self.E_marginal.dtype
        grad_abs_sum = torch.zeros((), device=device, dtype=dtype)
        active_count = 0
        start = 0

        for batch_edge_index in edge_data_loader:
            batch_edge_index = batch_edge_index.to(device)
            batch_dst, batch_src = batch_edge_index.T
            end = start + batch_edge_index.size(0)

            logit_E = pred_E_func(t_float,
                                  X_t_one_hot,
                                  A_t,
                                  s_0,
                                  Y_0,
                                  batch_src,
                                  batch_dst)
            edge_ids = torch.arange(start, end, device=device)
            terms = self._score_sp_guidance_terms(logit_E, edge_ids, sp_k)
            grad_abs_sum = grad_abs_sum + torch.nan_to_num(
                terms["grad_raw"].detach().abs(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).sum().to(dtype=dtype)
            active_count += int(edge_ids.numel())
            start = end

        if active_count <= 0:
            return torch.zeros((), device=device, dtype=dtype)
        return grad_abs_sum / float(active_count)

    def _apply_score_sp_guidance(self,
                                 logit_E,
                                 edge_ids,
                                 eta_t,
                                 sp_k,
                                 sp_shift_clip=None,
                                 sp_guidance_normalize=False,
                                 sp_guidance_scale=None):
        """Apply EDGE_fairness score-SP guidance to one active edge batch."""
        device = logit_E.device
        dtype = logit_E.dtype
        terms = self._score_sp_guidance_terms(logit_E, edge_ids, sp_k)

        eta = torch.as_tensor(float(eta_t), device=device, dtype=dtype)
        k = torch.as_tensor(float(sp_k), device=device, dtype=dtype)
        z_raw = terms["z_raw"]
        grad_raw = terms["grad_raw"]
        grad_e = grad_raw
        graph_mean_abs = None

        if sp_guidance_normalize:
            if sp_guidance_scale is None:
                graph_mean_abs = torch.nan_to_num(
                    grad_raw.detach().abs(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).mean()
            else:
                graph_mean_abs = torch.as_tensor(
                    sp_guidance_scale,
                    device=device,
                    dtype=dtype,
                )

            scale_floor = torch.tensor(1e-30, device=device, dtype=dtype)
            graph_scale = torch.maximum(graph_mean_abs, scale_floor)
            valid_scale = graph_mean_abs > 0
            grad_e = grad_raw / graph_scale
            grad_e = torch.where(valid_scale, grad_e, torch.zeros_like(grad_e))
            grad_e = torch.nan_to_num(grad_e, nan=0.0, posinf=0.0, neginf=0.0)

        z_guided = z_raw - eta * grad_e
        valid_guidance = bool((terms["n1"] > 0).item() and (terms["n0"] > 0).item())
        z_final = z_guided if valid_guidance else z_raw

        if sp_shift_clip is not None and sp_shift_clip > 0:
            clip = float(sp_shift_clip)
            delta_z = torch.clamp(z_final - z_raw, min=-clip, max=clip)
            z_final = z_raw + delta_z

        log_model_prob_edge = self._shift_binary_log_probs_from_pos_logit(z_final)

        h_new = terms["h_prev"] + k * (z_final - terms["h_prev"])
        q_new = torch.sigmoid(h_new)
        delta_q_new = q_new - terms["q_prev"]

        self._fair_score_h[terms["edge_ids"]] = h_new.to(dtype=self._fair_score_h.dtype)
        self._fair_score_q[terms["edge_ids"]] = q_new.to(dtype=self._fair_score_q.dtype)
        self._fair_score_R1 = self._fair_score_R1 + (
            delta_q_new * terms["mask_float"]).sum().to(dtype=self._fair_score_R1.dtype)
        self._fair_score_R0 = self._fair_score_R0 + (
            delta_q_new * terms["inv_mask_float"]).sum().to(dtype=self._fair_score_R0.dtype)

        trace = {
            "fair_score_sp_enabled": True,
            "fair_score_k": float(sp_k),
            "fair_score_delta_sp": terms["delta_sp"].detach(),
            "fair_score_C1": terms["c1"].detach(),
            "fair_score_C0": terms["c0"].detach(),
            "fair_score_mean_q_active_prev": terms["q_prev"].mean().item(),
            "fair_score_mean_q_active_new": q_new.mean().item(),
            "fair_score_step_scale_mean": terms["step_scale"].detach().item(),
            "fair_score_mean_abs_a_bar": terms["a_bar"].abs().mean().item(),
            "fair_score_mean_abs_logit_shift": (eta * grad_e).abs().mean().item(),
            "fair_score_guidance_normalize": bool(sp_guidance_normalize),
            "fair_score_raw_grad_abs_mean": grad_raw.detach().abs().mean().item(),
            "fair_score_grad_dir_abs_mean": grad_e.detach().abs().mean().item(),
            "valid_guidance": valid_guidance,
        }
        if graph_mean_abs is not None:
            graph_mean_abs_value = graph_mean_abs.detach().item()
            trace["fair_score_graph_mean_abs_grad_min"] = graph_mean_abs_value
            trace["fair_score_graph_mean_abs_grad_mean"] = graph_mean_abs_value
            trace["fair_score_graph_mean_abs_grad_max"] = graph_mean_abs_value
        return log_model_prob_edge, trace

    def denoise_match_Z(self,
                        Z_t_one_hot,
                        Q_t_Z,
                        Z_one_hot,
                        Q_bar_s_Z,
                        pred_Z):
        """Compute the denoising match term for Z given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        Z_t_one_hot : torch.Tensor of shape (B, C) or (A, B, C)
            One-hot encoding of the data sampled at time step t.
        Q_t_Z : torch.Tensor of shape (C, C) or (A, C, C)
            Transition matrix from time step t - 1 to t.
        Z_one_hot : torch.Tensor of shape (B, C) or (A, B, C)
            One-hot encoding of the original data.
        Q_bar_s_Z : torch.Tensor of shape (C, C) or (A, C, C)
            Transition matrix from timestep 0 to t-1.
        pred_Z : torch.Tensor of shape (B, C) or (A, B, C)
            Predicted probs for the original data.

        Returns
        -------
        float
            KL value.
        """
        # q(Z^{t-1}| Z, Z^t)
        left_term = Z_t_one_hot @ torch.transpose(Q_t_Z, -1, -2) # (B, C) or (A, B, C)
        right_term = Z_one_hot @ Q_bar_s_Z                       # (B, C) or (A, B, C)
        product = left_term * right_term                         # (B, C) or (A, B, C)
        denom = product.sum(dim=-1)                              # (B,) or (A, B)
        denom[denom == 0.] = 1
        prob_true = product / denom.unsqueeze(-1)                # (B, C) or (A, B, C)

        # q(Z^{t-1}| hat{p}^{Z}, Z^t)
        right_term = pred_Z @ Q_bar_s_Z                          # (B, C) or (A, B, C)
        product = left_term * right_term                         # (B, C) or (A, B, C)
        denom = product.sum(dim=-1)                              # (B,) or (A, B)
        denom[denom == 0.] = 1
        prob_pred = product / denom.unsqueeze(-1)                # (B, C) or (A, B, C)

        # KL(q(Z^{t-1}| hat{p}^{Z}, Z^t) || q(Z^{t-1}| Z, Z^t))
        kl = F.kl_div(input=prob_pred.log(), target=prob_true, reduction='none')
        return kl.clamp(min=0).mean().item()

    def denoise_match_E(self,
                        t_float,
                        logit_E,
                        E_t_one_hot,
                        E_one_hot):
        """Compute the denoising match term for edge prediction given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        logit_E : torch.Tensor of shape (B, 2)
            Predicted logits for the edge existence of a batch of node pairs.
        E_t_one_hot : torch.Tensor of shape (B, 2)
            One-hot encoding of sampled edge existence for the batch of
            node pairs.
        E_one_hot : torch.Tensor of shape (B, 2)
            One-hot encoding of the original edge existence for the batch of
            node pairs.

        Returns
        -------
        float
            KL value.
        """
        t = int(t_float.item() * self.T)
        s = t - 1

        alpha_bar_s = self.noise_schedule.alpha_bars[s]
        alpha_t = self.noise_schedule.alphas[t]

        Q_bar_s_E = self.transition_A.get_Q_bar_E(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_E = self.transition_A.get_Q_bar_E(alpha_t)

        pred_E = logit_E.softmax(dim=-1)

        return self.denoise_match_Z(E_t_one_hot,
                                    Q_t_E,
                                    E_one_hot,
                                    Q_bar_s_E,
                                    pred_E)

    def posterior(self,
                  Z_t,
                  Q_t,
                  Q_bar_s,
                  Q_bar_t,
                  prior):
        """Compute the posterior distribution for time step s, i.e., t - 1.

        Parameters
        ----------
        Z_t : torch.Tensor of shape (B, 2) or (F, |V|, 2)
            One-hot encoding of the sampled data at timestep t.
            B for batch size, C for number of classes, D for number
            of features.
        Q_t : torch.Tensor of shape (2, 2) or (F, 2, 2)
            The transition matrix from timestep t-1 to t.
        Q_bar_s : torch.Tensor of shape (2, 2) or (F, 2, 2)
            The transition matrix from timestep 0 to t-1.
        Q_bar_t : torch.Tensor of shape (2, 2) or (F, 2, 2)
            The transition matrix from timestep 0 to t.
        prior : torch.Tensor of shape (B, 2) or (F, |V|, 2)
            Reconstructed prior distribution.

        Returns
        -------
        prob : torch.Tensor of shape (B, 2) or (D, B, C)
            Posterior distribution.
        """
        # (B, 2) or (F, |V|, 2)
        left_term = Z_t @ torch.transpose(Q_t, -1, -2)
        # (B, 1, 2) or (F, |V|, 1, 2)
        left_term = left_term.unsqueeze(dim=-2)
        # (1, 2, 2) or (F, 1, 2, 2)
        right_term = Q_bar_s.unsqueeze(dim=-3)
        # (B, 2, 2) or (F, |V|, 2, 2)
        # Different from denoise_match_z, this function does not
        # compute (Z_t @ Q_t.T) * (Z_0 @ Q_bar_s) for a specific
        # Z_0, but compute for all possible values of Z_0.
        numerator = left_term * right_term

        # (2, B) or (F, 2, |V|)
        prod = Q_bar_t @ torch.transpose(Z_t, -1, -2)
        # (B, 2) or (F, |V|, 2)
        prod = torch.transpose(prod, -1, -2)
        # (B, 2, 1) or (F, |V|, 2, 1)
        denominator = prod.unsqueeze(-1)
        denominator[denominator == 0.] = 1.
        # (B, 2, 2) or (F, |V|, 2, 2)
        out = numerator / denominator

        # (B, 2, 2) or (F, |V|, 2, 2)
        prob = prior.unsqueeze(-1) * out
        # (B, 2) or (F, |V|, C)
        prob = prob.sum(dim=-2)

        return prob

    def sample_E_infer(self, prob_E):
        """Draw a sample from prob_E

        Parameters
        ----------
        prob_E : torch.Tensor of shape (B, 2)
            Probability distributions for edge existence.

        Returns
        -------
        E_t : torch.LongTensor of shape (|V|, |V|)
            Sampled adjacency matrix.
        """
        E_t_ = prob_E.multinomial(1).squeeze(-1)
        E_t = torch.zeros(self.num_nodes, self.num_nodes).long().to(E_t_.device)
        E_t[self.dst, self.src] = E_t_
        E_t[self.src, self.dst] = E_t_

        return E_t

    def get_E_t(self,
                device,
                edge_data_loader,
                pred_E_func,
                t_float,
                X_t_one_hot,
                s_0,
                E_t,
                Q_t_E,
                Q_bar_s_E,
                Q_bar_t_E,
                batch_size,
                Y_0,
                edge_group_labels=None,
                sp_shift=False,
                sp_eta=0.0,
                sp_k=1.0,
                sp_eta_schedule="constant",
                sp_shift_clip=None,
                sp_guidance_normalize=False,
                sp_stats=None):
        A_t = self.get_adj(E_t)
        E_prob = torch.zeros(len(self.src), self.num_classes_E).to(device)

        base_eta = float(sp_eta) if sp_eta is not None else 0.0
        sp_k_value = float(sp_k) if sp_k is not None else 1.0
        sp_shift_enabled = bool(sp_shift) and base_eta > 0.0 and sp_k_value > 0.0

        # Default sampler path: keep the original FairWire reverse edge sampler
        # when the post-hoc SP correction is disabled or eta is zero.
        if not sp_shift_enabled:
            start = 0
            for batch_edge_index in edge_data_loader:
                # (B, 2)
                batch_edge_index = batch_edge_index.to(device)
                batch_dst, batch_src = batch_edge_index.T
                # Reconstruct the edges.
                # (B, 2)
                batch_pred_E = pred_E_func(t_float,
                                           X_t_one_hot,
                                           A_t,
                                           s_0,
                                           Y_0,
                                           batch_src,
                                           batch_dst
                                           )

                batch_pred_E = batch_pred_E.softmax(dim=-1)

                # (B, 2)
                batch_E_t_one_hot = F.one_hot(
                    E_t[batch_src, batch_dst],
                    num_classes=self.num_classes_E).float()
                batch_E_prob_ = self.posterior(batch_E_t_one_hot, Q_t_E,
                                               Q_bar_s_E, Q_bar_t_E, batch_pred_E)

                end = start + batch_edge_index.size(0)
                E_prob[start: end] = batch_E_prob_
                start = end

            E_t = self.sample_E_infer(E_prob)

            return A_t, E_t

        if not hasattr(self, "_fair_score_q"):
            labels = edge_group_labels if edge_group_labels is not None else s_0
            self._init_score_sp_state(labels, E_t)

        t_int = int(round(float(t_float.item() * self.T)))
        eta_t = self._sp_eta_at_step(base_eta, t_int, sp_eta_schedule)

        n_same = self._fair_N1.to(device=device)
        n_diff = self._fair_N0.to(device=device)
        has_both_groups = bool((n_same > 0).item() and (n_diff > 0).item())
        sp_step_stats = None
        batch_traces = []
        if sp_stats is not None:
            sp_step_stats = {
                "t": int(t_int),
                "eta_t": float(eta_t),
                "sp_k": float(sp_k_value),
                "sp_guidance_normalize": bool(sp_guidance_normalize),
                "n_same": int(n_same.detach().cpu().item()),
                "n_diff": int(n_diff.detach().cpu().item()),
                "shift_applied": bool(has_both_groups),
            }

        guidance_scale = None
        if sp_guidance_normalize:
            guidance_scale = self._estimate_score_sp_guidance_scale(
                edge_data_loader,
                pred_E_func,
                t_float,
                X_t_one_hot,
                A_t,
                s_0,
                Y_0,
                sp_k_value,
                device,
            )
            if sp_step_stats is not None:
                sp_step_stats["fair_score_graph_mean_abs_grad_mean"] = float(
                    guidance_scale.detach().cpu().item())

        start = 0
        for batch_edge_index in edge_data_loader:
            # (B, 2)
            batch_edge_index = batch_edge_index.to(device)
            batch_dst, batch_src = batch_edge_index.T
            end = start + batch_edge_index.size(0)
            # Reconstruct the edges.
            # (B, 2)
            logit_E = pred_E_func(t_float,
                                  X_t_one_hot,
                                  A_t,
                                  s_0,
                                  Y_0,
                                  batch_src,
                                  batch_dst
                                  )

            edge_ids = torch.arange(start, end, device=device)
            log_model_prob_E, trace = self._apply_score_sp_guidance(
                logit_E,
                edge_ids,
                eta_t,
                sp_k_value,
                sp_shift_clip,
                sp_guidance_normalize=sp_guidance_normalize,
                sp_guidance_scale=guidance_scale,
            )
            batch_pred_E = log_model_prob_E.exp()
            if sp_step_stats is not None:
                batch_traces.append(trace)

            # (B, 2)
            batch_E_t_one_hot = F.one_hot(
                E_t[batch_src, batch_dst],
                num_classes=self.num_classes_E).float()
            batch_E_prob_ = self.posterior(batch_E_t_one_hot, Q_t_E,
                                           Q_bar_s_E, Q_bar_t_E, batch_pred_E)

            E_prob[start: end] = batch_E_prob_
            start = end

        if sp_step_stats is not None:
            safe_n_same = torch.where(n_same > 0, n_same, torch.ones_like(n_same))
            safe_n_diff = torch.where(n_diff > 0, n_diff, torch.ones_like(n_diff))
            if has_both_groups:
                mean_same_after = self._fair_score_R1.to(device=device) / safe_n_same
                mean_diff_after = self._fair_score_R0.to(device=device) / safe_n_diff
                delta_sp_after = mean_same_after - mean_diff_after
            else:
                mean_same_after = torch.zeros((), device=device)
                mean_diff_after = torch.zeros((), device=device)
                delta_sp_after = torch.zeros((), device=device)

            delta_sp_after_value = float(delta_sp_after.detach().cpu().item())
            if batch_traces:
                delta_vals = [
                    float(t["fair_score_delta_sp"].detach().cpu().item())
                    for t in batch_traces
                ]
                mean_abs_shift_vals = [
                    float(t["fair_score_mean_abs_logit_shift"])
                    for t in batch_traces
                ]
                sp_step_stats.update({
                    "delta_sp_before": float(sum(delta_vals) / len(delta_vals)),
                    "abs_delta_sp_before": float(
                        sum(abs(v) for v in delta_vals) / len(delta_vals)),
                    "fair_score_delta_sp_first": float(delta_vals[0]),
                    "fair_score_delta_sp_last": float(delta_vals[-1]),
                    "fair_score_mean_abs_logit_shift": float(
                        sum(mean_abs_shift_vals) / len(mean_abs_shift_vals)),
                    "fair_score_step_scale_mean": float(
                        sum(float(t["fair_score_step_scale_mean"])
                            for t in batch_traces) / len(batch_traces)),
                    "fair_score_mean_abs_a_bar": float(
                        sum(float(t["fair_score_mean_abs_a_bar"])
                            for t in batch_traces) / len(batch_traces)),
                })
            else:
                sp_step_stats.update({
                    "delta_sp_before": 0.0,
                    "abs_delta_sp_before": 0.0,
                })
            sp_step_stats.update({
                "mean_same_after_prior": float(mean_same_after.detach().cpu().item()),
                "mean_diff_after_prior": float(mean_diff_after.detach().cpu().item()),
                "delta_sp_after_prior": delta_sp_after_value,
                "abs_delta_sp_after_prior": abs(delta_sp_after_value),
            })
            sp_stats.append(sp_step_stats)

        E_t = self.sample_E_infer(E_prob)

        return A_t, E_t

class LossX(nn.Module):
    """
    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X):
        super().__init__()

        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

    def forward(self, true_X, logit_X):
        """
        Parameters
        ----------
        true_X : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        """
        true_X = true_X.transpose(0, 1)               # (|V|, F, 2)
        # v1x1, v1x2, ..., v1xd, v2x1, ...
        true_X = true_X.reshape(-1, true_X.size(-1))  # (|V| * F, 2)

        # v1x1, v1x2, ..., v1xd, v2x1, ...
        logit_X = logit_X.reshape(true_X.size(0), -1) # (|V| * F, 2)

        true_X = torch.argmax(true_X, dim=-1)         # (|V| * F)
        loss_X = F.cross_entropy(logit_X, true_X)

        return loss_X
    
class FairLossX(nn.Module):
    """
    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    num_classes_X : int
        Number of classes for each node attribute.
    """
    def __init__(self,
                 num_attrs_X,
                 num_classes_X):
        super().__init__()

        self.num_attrs_X = num_attrs_X
        self.num_classes_X = num_classes_X

    def forward(self, true_X, logit_X, p_values):
        """
        Parameters
        ----------
        true_X : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        """
        true_X = true_X.transpose(0, 1)               # (|V|, F, 2)
        fair_loss_X = 0
        for i in range(self.num_attrs_X):
            fair_loss_X += p_values[i] * F.cross_entropy(logit_X[:,i,:], torch.argmax(true_X[:,i,:], dim=-1))

        return fair_loss_X

class ModelSync(BaseModel):
    """
    Parameters
    ----------
    T : int
        Number of diffusion time steps - 1.
    X_marginal : torch.Tensor of shape (F, 2)
        X_marginal[f, :] is the marginal distribution of the f-th node attribute.
    Y_marginal : torch.Tensor of shape (C)
        Marginal distribution of the node labels.
    E_marginal : torch.Tensor of shape (2)
        Marginal distribution of the edge existence.
    gnn_X_config : dict
        Configuration of the GNN for reconstructing node attributes.
    gnn_E_config : dict
        Configuration of the GNN for reconstructing edges.
    num_nodes : int
        Number of nodes in the original graph.
    """
    def __init__(self,
                 T,
                 X_marginal,
                 E_marginal,
                 s_marginal,
                 y_marginal,
                 y_cond_s_marginal,
                 gnn_X_config,
                 gnn_E_config,
                 num_nodes,
                 p_values):
        super().__init__(T=T,
                         X_marginal=X_marginal,
                         s_marginal=s_marginal,
                         y_marginal=y_marginal,
                         E_marginal=E_marginal,
                         num_nodes=num_nodes)
        self.y_marginal = y_marginal
        self.s_marginal = s_marginal
        self.y_cond_s_marginal = y_cond_s_marginal
        if y_marginal is not None:
            self.num_classes_y = len(y_marginal)
        else:
            self.num_classes_y = None
        self.num_classes_s = len(s_marginal)
        self.graph_encoder = GNN(num_attrs_X=self.num_attrs_X,
                                 num_classes_X=self.num_classes_X,
                                 num_classes_E=self.num_classes_E,
                                 num_classes_s=self.num_classes_s,
                                 num_classes_Y=self.num_classes_y,
                                 gnn_X_config=gnn_X_config,
                                 gnn_E_config=gnn_E_config)
        
        self.loss_X = LossX(self.num_attrs_X, self.num_classes_X)
        self.fair_loss_X = FairLossX(self.num_attrs_X, self.num_classes_X)
        self.p_values = p_values
    def apply_noise(self, X_one_hot_3d, E_one_hot, t=None):
        """Corrupt G and sample G^t.

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the sampled node attributes.
        """
        if t is None:
            # Sample a timestep t uniformly.                                                                                                                                                                                                               
            # Note that the notation is slightly inconsistent with the paper.                                                                                                                                                                                          # t=0 corresponds to t=1 in the paper, where corruption has already taken place.                                                                                                                                                               
            t = torch.randint(low=0, high=self.T + 1, size=(1,),
                              device=E_one_hot.device)

        alpha_bar_t = self.noise_schedule.alpha_bars[t]
        # Sample A^t.                                                                                                                                                                                                                                      
        Q_bar_t_E = self.transition_A.get_Q_bar_E(alpha_bar_t) # (2, 2)                                                                                                                                                                                    
        prob_E = E_one_hot @ Q_bar_t_E                       # (|V|, |V|, 2)                                                                                                                                                                               
        E_t = self.sample_E(prob_E)                          # (|V|, |V|)                                                                                                                                                                                  

        # Sample X^t.
        if X_one_hot_3d!=None:
            Q_bar_t_X = self.transition_X.get_Q_bar_X(alpha_bar_t) # (F, 2, 2)                                                                                                                                                                                         # Compute matrix multiplication over the first batch dimension.                                                                                                                                                                                
            prob_X = torch.bmm(X_one_hot_3d, Q_bar_t_X)          # (F, |V|, 2)                                                                                                                                                                             
            X_t_one_hot = self.sample_X(prob_X)
        else:
            X_t_one_hot=None

        t_float = t / self.T

        return t_float, X_t_one_hot, E_t

    def log_p_t(self,
                X_one_hot_3d,
                E_one_hot,
                batch_src,
                batch_dst,
                batch_E_one_hot,
                s,
                y,
                is_diffuse_X=True,
                t=None):
        """Obtain G^t and compute log p(G | G^t, Y, t).

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels of the real graph.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            E_one_hot[batch_dst, batch_src].
        t : torch.LongTensor of shape (1), optional
            If specified, a time step will be enforced rather than sampled.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        loss_E : torch.Tensor
            Scalar representing the loss for edge existence.
        """
        #s_one_hot = F.one_hot(s, num_classes=self.num_classes_s).double()
        t_float, X_t_one_hot, E_t = self.apply_noise(X_one_hot_3d, E_one_hot, t)

        #intra = torch.matmul(s_one_hot, torch.transpose(s_one_hot, 0, 1))
        #print('total intra', torch.sum(E_t[intra>0]))
        #print('total inter', torch.sum(E_t) - torch.sum(E_t[intra>0]))
        
        A_t = self.get_adj(E_t)
        if not is_diffuse_X:
            X_t_one_hot = X_one_hot_3d.transpose(0, 1) 
            X_t_one_hot = X_t_one_hot.reshape(X_t_one_hot.size(0), -1)

        logit_X, logit_E = self.graph_encoder(t_float,
                                              X_t_one_hot,
                                              A_t,
                                              s,
                                              y,
                                              batch_src,
                                              batch_dst)
        loss_X = self.loss_X(X_one_hot_3d, logit_X)
        fair_loss_X = self.fair_loss_X(X_one_hot_3d, logit_X, self.p_values)
        loss_E = self.loss_E(batch_E_one_hot, logit_E)
        batch_pred_E = logit_E.softmax(dim=-1)
        group_labels = y if y is not None else s
        pair_same_mask = (group_labels[batch_src] == group_labels[batch_dst]).float()
        pair_diff_mask = 1.0 - pair_same_mask
        fair_loss_E = self.fair_loss_E(
            batch_pred_E,
            pair_same_mask / pair_same_mask.sum().clamp_min(1.0),
            pair_diff_mask / pair_diff_mask.sum().clamp_min(1.0))
        return loss_X, fair_loss_X, loss_E, fair_loss_E

    def denoise_match_X(self,
                        t_float,
                        logit_X,
                        X_t_one_hot,
                        X_one_hot_3d):
        """Compute the denoising match term for node attribute prediction given a
        sampled t, i.e., the KL divergence between q(D^{t-1}| D, D^t) and
        q(D^{t-1}| hat{p}^{D}, D^t).

        Parameters
        ----------
        t_float : torch.Tensor of shape (1)
            Sampled timestep divided by self.T.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.
        X_t_one_hot : torch.Tensor of shape (|V|, 2 * F)
            One-hot encoding of the node attributes sampled at time step t.
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.

        Returns
        -------
        float
            KL value for node attributes.
        """
        t = int(t_float.item() * self.T)
        s = t - 1

        alpha_bar_s = self.noise_schedule.alpha_bars[s]
        alpha_t = self.noise_schedule.alphas[t]

        Q_bar_s_X = self.transition_X.get_Q_bar_X(alpha_bar_s)
        # Note that computing Q_bar_t from alpha_bar_t is the same
        # as computing Q_t from alpha_t.
        Q_t_X = self.transition_X.get_Q_bar_X(alpha_t)

        # (|V|, F, 2)
        pred_X = logit_X.softmax(dim=-1)
        # (F, |V|, 2)
        pred_X = torch.transpose(pred_X, 0, 1)

        num_nodes = X_t_one_hot.size(0)
        # (|V|, F, 2)
        X_t_one_hot = X_t_one_hot.reshape(num_nodes, self.num_attrs_X, -1)
        # (F, |V|, 2)
        X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

        return self.denoise_match_Z(X_t_one_hot,
                                    Q_t_X,
                                    X_one_hot_3d,
                                    Q_bar_s_X,
                                    pred_X)

    @torch.no_grad()
    def val_step(self,
                 X_one_hot_3d,
                 E_one_hot,
                 s,
                 y,
                 batch_src,
                 batch_dst,
                 batch_E_one_hot,
                 is_diff_X=True):
        """Perform a validation step.

        Parameters
        ----------
        X_one_hot_3d : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute
            in the real graph.
        E_one_hot : torch.Tensor of shape (|V|, |V|, 2)
            - E_one_hot[:, :, 0] indicates the absence of an edge in the real graph.
            - E_one_hot[:, :, 1] is the adjacency matrix of the real graph.
        s : torch.Tensor of shape (|V|)
            Categorical sensitive attribute labels of the real graph.
        Y : torch.Tensor of shape (|V|)
            Categorical node labels of the real graph.
        batch_src : torch.LongTensor of shape (B)
            Source node IDs for a batch of candidate edges (node pairs).
        batch_dst : torch.LongTensor of shape (B)
            Destination node IDs for a batch of candidate edges (node pairs).
        batch_E_one_hot : torch.Tensor of shape (B, 2)
            E_one_hot[batch_dst, batch_src].

        Returns
        -------
        denoise_match_E : float
            Denoising matching term for edge.
        denoise_match_X : float
            Denoising matching term for node attributes.
        log_p_0_E : float
            Reconstruction term for edge.
        log_p_0_X : float
            Reconstruction term for node attributes.
        """
        device = E_one_hot.device
        if is_diff_X:
            denoise_match_X = []
        denoise_match_E = []

        # t=0 is handled separately.
        for t_sample in range(1, self.T + 1):
            t = torch.LongTensor([t_sample]).to(device)
            t_float, X_t_one_hot, E_t = self.apply_noise(
                X_one_hot_3d, E_one_hot, t)
            if not is_diff_X:
                X_t_one_hot = X_one_hot_3d.transpose(0, 1) 
                X_t_one_hot = X_t_one_hot.reshape(X_t_one_hot.size(0), -1)
                
            A_t = self.get_adj(E_t)
            logit_X, logit_E = self.graph_encoder(t_float,
                                                  X_t_one_hot,
                                                  A_t,
                                                  s,
                                                  y,
                                                  batch_src,
                                                  batch_dst)

            E_t_one_hot = F.one_hot(E_t[batch_src, batch_dst],
                                    num_classes=self.num_classes_E).float()
            denoise_match_E_t = self.denoise_match_E(t_float,
                                                     logit_E,
                                                     E_t_one_hot,
                                                     batch_E_one_hot)
            denoise_match_E.append(denoise_match_E_t)
            if is_diff_X:
                denoise_match_X_t = self.denoise_match_X(t_float,
                                                         logit_X,
                                                         X_t_one_hot,
                                                         X_one_hot_3d)
                denoise_match_X.append(denoise_match_X_t)

        denoise_match_E = float(np.mean(denoise_match_E)) * self.T
        if is_diff_X:
            denoise_match_X = float(np.mean(denoise_match_X)) * self.T

        # t=0
        t_0 = torch.LongTensor([0]).to(device)
        loss_X, fair_loss_X, loss_E, fair_loss_E = self.log_p_t(X_one_hot_3d=X_one_hot_3d,
                                                                E_one_hot=E_one_hot,
                                                                s=s,
                                                                y=y,
                                                                batch_src=batch_src,
                                                                batch_dst=batch_dst,
                                                                batch_E_one_hot=batch_E_one_hot,
                                                                t=t_0,
                                                                is_diffuse_X=is_diff_X)
        log_p_0_E = loss_E.item()
        log_p_0_X = loss_X.item()

        return denoise_match_E, denoise_match_X,\
            log_p_0_E, fair_loss_E, log_p_0_X, fair_loss_X

    @torch.no_grad()
    def sample(self,
               is_diff_X=True,
               batch_size=32768,
               num_workers=4,
               sp_shift=False,
               sp_eta=0.0,
               sp_k=1.0,
               sp_eta_schedule="constant",
               sp_shift_clip=None,
               sp_guidance_normalize=False,
               return_sp_stats=False):
        """Sample a graph.

        Parameters
        ----------
        batch_size : int
            Batch size for edge prediction.
        num_workers : int
            Number of subprocesses for data loading in edge prediction.
        sp_shift : bool
            If True, apply a post-training edge-only SP logit shift while
            sampling. This does not update parameters or alter X generation.
        sp_eta : float
            Strength of the post-hoc SP logit shift. A value of 0 keeps the
            original sampler path.
        sp_k : float
            Multiplicative scale for the SP logit-shift gradient. This is the
            FairWire_shift counterpart to EDGE_fairness fair_score_k.
        sp_eta_schedule : str
            One of "constant", "early", or "late".
        sp_shift_clip : float, optional
            Optional absolute clamp for each edge-existence logit shift.
        sp_guidance_normalize : bool
            If True, divide the SP raw guidance by the graph-level
            mean(abs(raw guidance)) at each reverse step.
        return_sp_stats : bool
            If True, append per-step SP shift diagnostics to the return value.

        Returns
        -------
        X_t_one_hot : torch.Tensor of shape (F, |V|, 2)
            One-hot encoding of the generated node attributes.
        Y_0_one_hot : torch.Tensor of shape (|V|, C)
            One-hot encoding of the generated node labels.
        E_t : torch.LongTensor of shape (|V|, |V|)
            Adjacency matrix of the generated graph.
        """
        device = self.X_marginal.device
        dst, src = torch.triu_indices(self.num_nodes, self.num_nodes,
                                      offset=1, device=device)
        self.last_sp_shift_stats = []
        self._clear_score_sp_state()
        # (|E|)
        self.dst = dst
        # (|E|)
        self.src = src
        # (|E|, 2)
        edge_index = torch.stack([dst, src], dim=1).to("cpu")
        data_loader = DataLoader(edge_index, batch_size=batch_size,
                                 num_workers=num_workers)

        # Sample G^T from prior distribution.
        # (|V|, C)
        s_prior = self.s_marginal[None, :].expand(self.num_nodes, -1)
        s_0 = s_prior.multinomial(1).reshape(-1)
        y_0 = torch.zeros(len(s_0), device = s_0.device)
        if self.y_cond_s_marginal is not None:
            for k in range(self.s_marginal.size(-1)):
                y_prior = self.y_cond_s_marginal[:, k].expand(sum(s_0==k), -1)
                y_0_k = y_prior.multinomial(1).reshape(-1)
                y_0[s_0==k] = y_0_k.float()
            y_0 = torch.LongTensor(y_0.cpu().numpy()).to(s_0.device)
        else:
            y_0 = None
        
        

        # (|V|, |V|, 2)
        E_prior = self.E_marginal[None, None, :].expand(
            self.num_nodes, self.num_nodes, -1)
        # (|V|, |V|)
        E_t = self.sample_E(E_prior)
        edge_group_labels = y_0 if y_0 is not None else s_0
        if bool(sp_shift) and float(sp_eta or 0.0) > 0.0 and float(sp_k or 0.0) > 0.0:
            self._init_score_sp_state(edge_group_labels, E_t)

        # (F, |V|, 2)
        X_prior = self.X_marginal[:, None, :].expand(-1, self.num_nodes, -1) # You should change this for different initializations of X
        # (|V|, 2F)
        X_t_one_hot = self.sample_X(X_prior)

        # Iteratively sample p(D^s | D^t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(list(reversed(range(0, self.T)))):
            t = s + 1

            # Note that computing Q_bar_t from alpha_bar_t is the same
            # as computing Q_t from alpha_t.
            alpha_t = self.noise_schedule.alphas[t]
            alpha_bar_s = self.noise_schedule.alpha_bars[s]
            alpha_bar_t = self.noise_schedule.alpha_bars[t]

            Q_t_E = self.transition_A.get_Q_bar_E(alpha_t)
            Q_bar_s_E = self.transition_A.get_Q_bar_E(alpha_bar_s)
            Q_bar_t_E = self.transition_A.get_Q_bar_E(alpha_bar_t)

            t_float = torch.tensor([t / self.T]).to(device)

            A_t, E_s = self.get_E_t(device,
                                    data_loader,
                                    self.graph_encoder.pred_E,
                                    t_float,
                                    X_t_one_hot,
                                    s_0,
                                    E_t,
                                    Q_t_E,
                                    Q_bar_s_E,
                                    Q_bar_t_E,
                                    batch_size,
                                    y_0,
                                    edge_group_labels=edge_group_labels,
                                    sp_shift=sp_shift,
                                    sp_eta=sp_eta,
                                    sp_k=sp_k,
                                    sp_eta_schedule=sp_eta_schedule,
                                    sp_shift_clip=sp_shift_clip,
                                    sp_guidance_normalize=sp_guidance_normalize,
                                    sp_stats=self.last_sp_shift_stats)

            # (|V|, F, 2)
            pred_X = self.graph_encoder.pred_X(t_float,
                                               X_t_one_hot,
                                               A_t,
                                               s_0,
                                               y_0)
            pred_X = pred_X.softmax(dim=-1)
            # (F, |V|, 2)
            pred_X = torch.transpose(pred_X, 0, 1)
            if is_diff_X:
                # (|V|, F, 2)
                X_t_one_hot = X_t_one_hot.reshape(self.num_nodes, self.num_attrs_X, -1)
                # (F, |V|, 2)
                X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)

                # (F, |V|, 2)
                Q_t_X = self.transition_X.get_Q_bar_X(alpha_t)
                Q_bar_s_X = self.transition_X.get_Q_bar_X(alpha_bar_s)
                Q_bar_t_X = self.transition_X.get_Q_bar_X(alpha_bar_t)

                X_prob = self.posterior(X_t_one_hot, Q_t_X,
                                        Q_bar_s_X, Q_bar_t_X, pred_X)
                X_t_one_hot = self.sample_X(X_prob)
            E_t = E_s
        X_t_one_hot = X_t_one_hot.reshape(self.num_nodes, self.num_attrs_X, -1)
        # (F, |V|, 2)
        X_t_one_hot = torch.transpose(X_t_one_hot, 0, 1)
        s_0_one_hot = F.one_hot(s_0, num_classes=self.num_classes_s).float()
        if y_0 is not None:
            y_0_one_hot = F.one_hot(y_0, num_classes=self.num_classes_y).float()
        else:
            y_0_one_hot = None
        node_orig_id = torch.arange(self.num_nodes, device=device)
        if return_sp_stats:
            return X_t_one_hot, s_0_one_hot, y_0_one_hot, E_t, node_orig_id, self.last_sp_shift_stats
        return X_t_one_hot, s_0_one_hot, y_0_one_hot, E_t, node_orig_id

            

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
eps = 1e-8


def _inverse_softplus(x):
    x = torch.as_tensor(x).clamp(min=1e-8, max=80.0)
    return x + torch.log(-torch.expm1(-x))

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
                 num_nodes,
                 fair_label_attr="y",
                 fair_score_eta=0.0,
                 fair_score_k=0.15,
                 fair_score_eta_scale=1.0,
                 fair_score_controller_train=False,
                 fair_score_learn_k=True,
                 fair_score_learn_eta=True,
                 fair_score_guidance_normalize=True,
                 fair_score_fair_loss_weight=1.0,
                 fair_score_k_tracking_loss_weight=1.0,
                 fair_score_utility_loss_weight=1.0):
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
        self.fair_label_attr = fair_label_attr
        self.fair_score_eta = float(fair_score_eta)
        self.fair_score_k = float(fair_score_k)
        self.fair_score_eta_scale = max(float(fair_score_eta_scale), 1e-8)
        self.fair_score_controller_train = bool(fair_score_controller_train)
        self.fair_score_learn_k = bool(fair_score_learn_k)
        self.fair_score_learn_eta = bool(fair_score_learn_eta)
        self.fair_score_guidance_normalize = bool(fair_score_guidance_normalize)
        self.fair_score_fair_loss_weight = float(fair_score_fair_loss_weight)
        self.fair_score_k_tracking_loss_weight = float(fair_score_k_tracking_loss_weight)
        self.fair_score_utility_loss_weight = float(fair_score_utility_loss_weight)
        self._init_fair_controller_params()

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

    def _sample_categorical_indices_with_optional_gumbel(self, prob, gumbel_noise=None):
        logits = torch.log(prob.clamp(min=1e-30))
        if gumbel_noise is None:
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        else:
            gumbel_noise = gumbel_noise.to(device=logits.device, dtype=logits.dtype)
            if gumbel_noise.shape != logits.shape:
                raise ValueError(
                    f"stored gumbel noise shape {tuple(gumbel_noise.shape)} "
                    f"does not match categorical logits shape {tuple(logits.shape)}"
                )
        return (logits + gumbel_noise).argmax(dim=-1), gumbel_noise

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

    def _init_fair_controller_params(self):
        if not self.fair_score_controller_train:
            self.register_parameter("fair_score_k_raw", None)
            self.register_parameter("fair_score_eta_raw", None)
            return

        raw_k_scalar = torch.logit(torch.tensor(float(self.fair_score_k)).clamp(1e-4, 1.0 - 1e-4))
        raw_k = raw_k_scalar.repeat(self.T)

        # EDGE convention: fair_score_eta is the base eta, and the learned
        # per-step parameter is a positive multiplier initialized at 1.
        raw_eta = torch.zeros(self.T)

        self.fair_score_k_raw = nn.Parameter(raw_k)
        self.fair_score_eta_raw = nn.Parameter(raw_eta)

    def _get_effective_fair_score_k(self, t_index=None):
        if not self.fair_score_controller_train:
            return self.fair_score_k

        effective_k = torch.sigmoid(self.fair_score_k_raw)
        if t_index is None:
            return effective_k

        index = torch.as_tensor(t_index, device=effective_k.device, dtype=torch.long)
        index = index.clamp(0, self.T - 1)
        return effective_k.index_select(0, index.reshape(-1)).reshape(index.shape)

    def _get_effective_fair_score_eta(self, t_index=None):
        if not self.fair_score_controller_train:
            return self.fair_score_eta

        base_eta = torch.as_tensor(
            float(self.fair_score_eta),
            device=self.fair_score_eta_raw.device,
            dtype=self.fair_score_eta_raw.dtype,
        )
        denom = F.softplus(torch.zeros((), device=self.fair_score_eta_raw.device, dtype=self.fair_score_eta_raw.dtype))
        effective_eta = base_eta * F.softplus(self.fair_score_eta_raw) / denom
        if t_index is None:
            return effective_eta

        index = torch.as_tensor(t_index, device=effective_eta.device, dtype=torch.long)
        index = index.clamp(0, self.T - 1)
        return effective_eta.index_select(0, index.reshape(-1)).reshape(index.shape)

    def _sp_eta_at_step(self, base_eta, t, schedule):
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
            "_fair_edge_group_labels",
            "_fair_N1",
            "_fair_N0",
            "_fair_score_h",
            "_fair_score_q",
            "_fair_score_R1",
            "_fair_score_R0",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _select_fair_edge_group_labels(self, s_0, y_0):
        label_attr = str(getattr(self, "fair_label_attr", "y") or "y").lower()
        if label_attr in {"s", "sens", "sensitive", "sensitive_attr"}:
            return s_0
        if label_attr == "y":
            return y_0 if y_0 is not None else s_0
        return y_0 if y_0 is not None else s_0

    def _shift_binary_log_probs_from_pos_logit(self, pos_logit):
        log_p1 = F.logsigmoid(pos_logit)
        log_p0 = F.logsigmoid(-pos_logit)
        return torch.stack([log_p0, log_p1], dim=1)

    def _init_score_sp_state(self, edge_group_labels, E_t):
        device = E_t.device
        dtype = self.E_marginal.dtype
        num_edges = self.src.numel()

        if edge_group_labels is None:
            mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
            labels = None
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
        rho_edge = rho.expand(num_edges).clone()

        self._fair_edge_sensitive_mask = mask
        self._fair_edge_group_labels = labels.detach().clone() if labels is not None else None
        self._fair_N1 = n1
        self._fair_N0 = n0
        self._fair_score_h = torch.logit(rho_edge)
        self._fair_score_q = rho_edge.clone()
        self._fair_score_R1 = (self._fair_score_q * mask_float).sum()
        self._fair_score_R0 = (self._fair_score_q * inv_mask_float).sum()

    def _build_controller_replay_header(self, s_0=None, y_0=None, edge_group_labels=None, E_init=None, X_init=None):
        def _clone_or_none(value):
            if value is None:
                return None
            return value.detach().cpu().clone()

        return {
            "s_0": _clone_or_none(s_0),
            "y_0": _clone_or_none(y_0),
            "edge_group_labels": _clone_or_none(edge_group_labels),
            "E_init": _clone_or_none(E_init),
            "X_init": _clone_or_none(X_init),
            "h_init": self._fair_score_h.detach().cpu().clone(),
            "q_init": self._fair_score_q.detach().cpu().clone(),
            "R1_init": self._fair_score_R1.detach().cpu().clone(),
            "R0_init": self._fair_score_R0.detach().cpu().clone(),
            "N1": self._fair_N1.detach().cpu().clone(),
            "N0": self._fair_N0.detach().cpu().clone(),
            "full_mask": self._fair_edge_sensitive_mask.detach().cpu().clone(),
            "num_nodes": int(self.num_nodes),
            "num_full_edges": int(self.src.numel()),
            "steps": [],
        }

    def _compute_fair_controller_guidance(
        self,
        z_active,
        h_active,
        R1,
        R0,
        N1,
        N0,
        mask_active,
        k_active,
    ):
        dtype = z_active.dtype
        device = z_active.device
        mask_float = mask_active.to(device=device, dtype=dtype)
        inv_mask_float = (~mask_active).to(device=device, dtype=dtype)
        k_active = torch.as_tensor(k_active, dtype=dtype, device=device)

        h_pre = h_active + k_active * (z_active - h_active)
        q_prev = torch.sigmoid(h_active)
        q_pre = torch.sigmoid(h_pre)
        delta_q_pre = q_pre - q_prev

        safe_N1 = torch.where(N1 > 0, N1, torch.ones_like(N1))
        safe_N0 = torch.where(N0 > 0, N0, torch.ones_like(N0))
        R1_pre = R1 + (delta_q_pre * mask_float).sum()
        R0_pre = R0 + (delta_q_pre * inv_mask_float).sum()
        delta_pre = R1_pre / safe_N1 - R0_pre / safe_N0

        a_e = mask_float / safe_N1 - inv_mask_float / safe_N0

        if self.fair_score_guidance_normalize:
            grad_raw = delta_pre * a_e * k_active * q_pre * (1.0 - q_pre)
            grad_scale = torch.ones_like(grad_raw)
            if grad_raw.numel() > 0:
                mean_abs = grad_raw.detach().abs().mean()
                scale_floor = torch.tensor(1e-30, device=device, dtype=dtype)
                scale = torch.maximum(mean_abs, scale_floor)
                grad_scale = scale.expand_as(grad_raw)
                grad = grad_raw / grad_scale
                grad = torch.where(mean_abs > 0, grad, torch.zeros_like(grad))
            else:
                grad = grad_raw
        else:
            step_scale = 0.5 * (N1 + N0)
            a_bar = step_scale * a_e
            grad_raw = delta_pre * a_bar * k_active * q_pre * (1.0 - q_pre)
            grad_scale = torch.ones_like(grad_raw)
            grad = grad_raw
            valid_guidance = (N1 > 0) & (N0 > 0)
            grad = torch.where(valid_guidance, grad, torch.zeros_like(grad))

        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        zero = grad_raw.sum() * 0.0
        diagnostics = {
            "h_pre": h_pre.detach(),
            "q_prev": q_prev.detach(),
            "q_pre": q_pre.detach(),
            "delta_q_pre": delta_q_pre.detach(),
            "R1_pre": R1_pre.detach(),
            "R0_pre": R0_pre.detach(),
            "delta_pre": delta_pre.detach(),
            "a_e": a_e.detach(),
            "grad_raw": grad_raw.detach(),
            "grad_scale": grad_scale.detach(),
            "grad_raw_abs_mean": grad_raw.detach().abs().mean() if grad_raw.numel() > 0 else zero.detach(),
            "grad_dir_abs_mean": grad.detach().abs().mean() if grad.numel() > 0 else zero.detach(),
        }
        return grad.detach(), diagnostics

    def _apply_score_sp_guidance(self,
                                 logit_E,
                                 edge_ids,
                                 eta_t,
                                 sp_k,
                                 sp_shift_clip=None):
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

        eta = torch.as_tensor(eta_t, device=device, dtype=dtype)
        k = torch.as_tensor(sp_k, device=device, dtype=dtype)
        z_raw = logit_E[:, 1] - logit_E[:, 0]
        k_active = k.expand_as(z_raw)

        grad_dir, diagnostics = self._compute_fair_controller_guidance(
            z_active=z_raw,
            h_active=h_prev,
            R1=r1,
            R0=r0,
            N1=n1,
            N0=n0,
            mask_active=mask_active,
            k_active=k_active,
        )
        # EDGE order: shift z first, then update h using the shifted logit.
        z_guided = z_raw - eta * grad_dir

        if sp_shift_clip is not None and sp_shift_clip > 0:
            clip = float(sp_shift_clip)
            delta_z = torch.clamp(z_guided - z_raw, min=-clip, max=clip)
            z_guided = z_raw + delta_z

        log_model_prob_edge = self._shift_binary_log_probs_from_pos_logit(z_guided)

        h_new = h_prev + k * (z_guided - h_prev)
        q_new = torch.sigmoid(h_new)
        delta_q_new = q_new - q_prev

        self._fair_score_h[edge_ids] = h_new.to(dtype=self._fair_score_h.dtype)
        self._fair_score_q[edge_ids] = q_new.to(dtype=self._fair_score_q.dtype)
        self._fair_score_R1 = self._fair_score_R1 + (
            delta_q_new * mask_float).sum().to(dtype=self._fair_score_R1.dtype)
        self._fair_score_R0 = self._fair_score_R0 + (
            delta_q_new * inv_mask_float).sum().to(dtype=self._fair_score_R0.dtype)

        trace = {
            "fair_score_sp_enabled": True,
            "fair_score_k": float(k.detach().cpu()),
            "fair_score_eta": float(eta.detach().cpu()),
            "fair_score_delta_sp": diagnostics["delta_pre"],
            "fair_score_mean_q_active_prev": q_prev.mean().item(),
            "fair_score_mean_q_active_new": q_new.mean().item(),
            "fair_score_mean_abs_logit_shift": (eta * grad_dir).detach().abs().mean().item(),
            "valid_guidance": bool((n1 > 0).item() and (n0 > 0).item()),
        }
        return log_model_prob_edge, trace

    def compute_fair_controller_loss_from_replay(self, replay):
        device = next(self.parameters()).device
        h = replay["h_init"].to(device=device)
        dtype = h.dtype
        R1 = replay["R1_init"].to(device=device, dtype=dtype)
        R0 = replay["R0_init"].to(device=device, dtype=dtype)
        N1 = replay["N1"].to(device=device, dtype=dtype)
        N0 = replay["N0"].to(device=device, dtype=dtype)
        full_mask = replay["full_mask"].to(device=device, dtype=torch.bool)

        k_all = self._get_effective_fair_score_k().to(device=device, dtype=dtype)
        eta_all = self._get_effective_fair_score_eta().to(device=device, dtype=dtype)
        zero = (k_all.sum() + eta_all.sum()) * 0.0

        k_tracking_loss = zero
        utility_loss = zero
        mean_abs_shift_sum = zero
        mean_abs_shift_count = 0
        raw_abs_sum = zero
        raw_abs_count = 0
        delta_pre_abs_sum = zero
        delta_pre_count = 0
        fair_guidance_eta_sum = zero
        fair_guidance_k_sum = zero
        fair_guidance_active_count = 0
        eta_seen = []
        k_seen = []

        for step in replay.get("steps", []):
            z_raw = step["z_raw"].to(device=device, dtype=dtype)
            if "edge_ids" in step:
                edge_ids = step["edge_ids"].to(device=device, dtype=torch.long)
            else:
                edge_ids = torch.arange(z_raw.numel(), device=device, dtype=torch.long)
            t_index = int(step["t_index"])

            if edge_ids.numel() == 0:
                continue

            k_t = self._get_effective_fair_score_k(t_index=t_index).to(device=device, dtype=dtype)
            eta_t = self._get_effective_fair_score_eta(t_index=t_index).to(device=device, dtype=dtype)
            k_seen.append(k_t.detach().reshape(1))
            eta_seen.append(eta_t.detach().reshape(1))

            h_active = h.index_select(0, edge_ids)
            mask_active = full_mask.index_select(0, edge_ids)
            k_active = k_t.expand_as(z_raw)
            eta_active = eta_t.expand_as(z_raw)

            grad_dir, diagnostics = self._compute_fair_controller_guidance(
                z_active=z_raw,
                h_active=h_active,
                R1=R1,
                R0=R0,
                N1=N1,
                N0=N0,
                mask_active=mask_active,
                k_active=k_active.detach(),
            )

            # EDGE order: shift z first, then update h using z_guided.
            z_guided = z_raw - eta_active * grad_dir
            h_new_fair = h_active + k_active.detach() * (z_guided - h_active)
            q_prev = torch.sigmoid(h_active)
            q_new_fair = torch.sigmoid(h_new_fair)
            delta_q = q_new_fair - q_prev

            mask_float = mask_active.to(dtype=dtype)
            inv_mask_float = (~mask_active).to(dtype=dtype)
            R1 = R1 + (delta_q * mask_float).sum()
            R0 = R0 + (delta_q * inv_mask_float).sum()
            h = h.index_copy(0, edge_ids, h_new_fair)

            pi_raw = torch.sigmoid(z_raw.detach())
            h_track = h_active.detach() + k_active * (z_raw.detach() - h_active.detach())
            k_tracking_loss = k_tracking_loss + F.binary_cross_entropy_with_logits(
                h_track,
                pi_raw,
            )

            h_unshifted = h_active + k_active.detach() * (z_raw - h_active)
            h_unshifted_ref = h_unshifted.detach()
            pi_h_unshifted = torch.sigmoid(h_unshifted_ref)
            utility_step = (
                pi_h_unshifted * (F.logsigmoid(h_unshifted_ref) - F.logsigmoid(h_new_fair))
                + (1.0 - pi_h_unshifted) * (
                    F.logsigmoid(-h_unshifted_ref) - F.logsigmoid(-h_new_fair)
                )
            ).mean()
            utility_loss = utility_loss + utility_step

            shift_abs = (eta_active * grad_dir).abs()
            mean_abs_shift_sum = mean_abs_shift_sum + shift_abs.sum()
            mean_abs_shift_count += int(shift_abs.numel())
            raw_abs_sum = raw_abs_sum + diagnostics["grad_raw"].abs().sum()
            raw_abs_count += int(diagnostics["grad_raw"].numel())
            delta_pre_abs_sum = delta_pre_abs_sum + diagnostics["delta_pre"].abs()
            delta_pre_count += 1
            fair_guidance_eta_sum = fair_guidance_eta_sum + eta_active.detach().sum()
            fair_guidance_k_sum = fair_guidance_k_sum + k_active.detach().sum()
            fair_guidance_active_count += int(edge_ids.numel())

        safe_N1 = torch.where(N1 > 0, N1, torch.ones_like(N1))
        safe_N0 = torch.where(N0 > 0, N0, torch.ones_like(N0))
        delta_final = R1 / safe_N1 - R0 / safe_N0
        valid_graph = (N1 > 0) & (N0 > 0)
        fair_loss = 0.5 * delta_final.pow(2) if valid_graph else zero
        delta_final_abs = delta_final.abs() if valid_graph else zero.detach()

        loss = (
            self.fair_score_fair_loss_weight * fair_loss
            + self.fair_score_k_tracking_loss_weight * k_tracking_loss
            + self.fair_score_utility_loss_weight * utility_loss
        )

        eta_stats = torch.cat(eta_seen) if eta_seen else eta_all.detach().reshape(-1)
        k_stats = torch.cat(k_seen) if k_seen else k_all.detach().reshape(-1)
        mean_abs_shift = mean_abs_shift_sum / float(mean_abs_shift_count) if mean_abs_shift_count else zero
        raw_abs_mean = raw_abs_sum / float(raw_abs_count) if raw_abs_count else zero
        delta_pre_abs_mean = delta_pre_abs_sum / float(delta_pre_count) if delta_pre_count else zero
        if fair_guidance_active_count > 0:
            fair_guidance_eta_mean = fair_guidance_eta_sum / float(fair_guidance_active_count)
            fair_guidance_k_mean = fair_guidance_k_sum / float(fair_guidance_active_count)
        else:
            fair_guidance_eta_mean = zero
            fair_guidance_k_mean = zero
        t0 = 0
        tmid = self.T // 2
        tlast = self.T - 1
        stats = {
            "fair_controller_fair_loss": float(fair_loss.detach().cpu()),
            "fair_controller_k_tracking_loss": float(k_tracking_loss.detach().cpu()),
            "fair_controller_utility_loss": float(utility_loss.detach().cpu()),
            "fair_controller_delta_final_abs_mean": float(delta_final_abs.detach().cpu()),
            "fair_controller_k_t0": float(k_all.detach()[t0].cpu()),
            "fair_controller_k_tmid": float(k_all.detach()[tmid].cpu()),
            "fair_controller_k_tlast": float(k_all.detach()[tlast].cpu()),
            "fair_controller_eta_t0": float(eta_all.detach()[t0].cpu()),
            "fair_controller_eta_tmid": float(eta_all.detach()[tmid].cpu()),
            "fair_controller_eta_tlast": float(eta_all.detach()[tlast].cpu()),
            "fair_controller_eta_mean": float(eta_stats.mean().detach().cpu()),
            "fair_controller_eta_min": float(eta_stats.min().detach().cpu()),
            "fair_controller_eta_max": float(eta_stats.max().detach().cpu()),
            "fair_controller_k_mean": float(k_stats.mean().detach().cpu()),
            "fair_controller_k_min": float(k_stats.min().detach().cpu()),
            "fair_controller_k_max": float(k_stats.max().detach().cpu()),
            "fair_controller_mean_abs_shift": float(mean_abs_shift.detach().cpu()),
            "fair_guidance_raw_abs_mean": float(raw_abs_mean.detach().cpu()),
            "fair_guidance_shift_abs_mean": float(mean_abs_shift.detach().cpu()),
            "fair_guidance_delta_pre_abs_mean": float(delta_pre_abs_mean.detach().cpu()),
            "fair_guidance_eta_mean": float(fair_guidance_eta_mean.detach().cpu()),
            "fair_guidance_k_mean": float(fair_guidance_k_mean.detach().cpu()),
        }
        return loss, stats

    def freeze_for_fair_controller_training(self):
        if not self.fair_score_controller_train:
            raise ValueError("fair_score_controller_train must be True for controller training.")
        if self.fair_score_k_raw is None or self.fair_score_eta_raw is None:
            raise ValueError("fair controller parameters are not initialized.")

        for param in self.parameters():
            param.requires_grad_(False)
        controller_params = []
        if self.fair_score_learn_k:
            self.fair_score_k_raw.requires_grad_(True)
            controller_params.append(self.fair_score_k_raw)
        if self.fair_score_learn_eta:
            self.fair_score_eta_raw.requires_grad_(True)
            controller_params.append(self.fair_score_eta_raw)
        if len(controller_params) == 0:
            raise ValueError("At least one of fair_score_learn_k/fair_score_learn_eta must be True.")
        self.eval()
        print(
            f"[controller] fair_score_k_raw shape: {tuple(self.fair_score_k_raw.shape)} "
            f"| trainable={self.fair_score_learn_k}"
        )
        print(
            f"[controller] fair_score_eta_raw shape: {tuple(self.fair_score_eta_raw.shape)} "
            f"| trainable={self.fair_score_learn_eta}"
        )
        print(f"[controller] learn_k={self.fair_score_learn_k} | learn_eta={self.fair_score_learn_eta}")
        return controller_params

    def get_fair_controller_state_dict(self):
        state = {
            "fair_score_k_mode": "per_step_sigmoid",
            "fair_score_eta_mode": "per_step_multiplier_softplus",
            "fair_score_eta_base": self.fair_score_eta,
            "fair_score_eta_scale": self.fair_score_eta_scale,
            "fair_score_learn_k": self.fair_score_learn_k,
            "fair_score_learn_eta": self.fair_score_learn_eta,
            "fair_label_attr": self.fair_label_attr,
            "num_timesteps": self.T,
            "fair_score_k_raw": self.fair_score_k_raw.detach().cpu().clone(),
            "fair_score_eta_raw": self.fair_score_eta_raw.detach().cpu().clone(),
            "fair_score_k_schedule": self._get_effective_fair_score_k().detach().cpu().clone(),
            "fair_score_eta_schedule": self._get_effective_fair_score_eta().detach().cpu().clone(),
        }
        return state

    def load_fair_controller_state_dict(self, state_dict, strict=True):
        if "controller" in state_dict:
            state_dict = state_dict["controller"]
        missing = []
        eta_mode = state_dict.get("fair_score_eta_mode")
        for name in ("fair_score_k_raw", "fair_score_eta_raw"):
            value = state_dict.get(name)
            param = getattr(self, name, None)
            if value is None:
                missing.append(name)
                continue
            if param is None:
                raise ValueError("Instantiate ModelSync with fair_score_controller_train=True before loading a controller.")
            value = value.to(device=param.device, dtype=param.dtype)
            if name == "fair_score_eta_raw" and eta_mode == "per_step_softplus":
                old_effective_eta = F.softplus(value)
                base_eta = max(float(self.fair_score_eta), 1e-8)
                denom = F.softplus(torch.zeros((), device=param.device, dtype=param.dtype))
                multiplier = old_effective_eta * denom / base_eta
                value = _inverse_softplus(multiplier).to(device=param.device, dtype=param.dtype)
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match current shape {tuple(param.shape)}.")
            param.data.copy_(value)
        if strict and missing:
            raise KeyError(f"Missing fair controller parameter(s): {missing}")

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

    def sample_E_infer(self, prob_E, gumbel_noise=None, return_gumbel=False):
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
        if gumbel_noise is None and not return_gumbel:
            E_t_ = prob_E.multinomial(1).squeeze(-1)
            edge_sample_gumbel = None
        else:
            E_t_, edge_sample_gumbel = self._sample_categorical_indices_with_optional_gumbel(
                prob_E,
                gumbel_noise=gumbel_noise,
            )
        E_t = torch.zeros(self.num_nodes, self.num_nodes).long().to(E_t_.device)
        E_t[self.dst, self.src] = E_t_
        E_t[self.src, self.dst] = E_t_

        if return_gumbel:
            return E_t, edge_sample_gumbel
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
                sp_stats=None,
                use_fair_controller=False,
                record_controller_replay=False,
                controller_replay=None,
                controller_replay_step=None):
        A_t = self.get_adj(E_t)
        E_prob = torch.zeros(len(self.src), self.num_classes_E).to(device)
        playback_controller_replay = controller_replay_step is not None

        t_int = int(round(float(t_float.item() * self.T)))
        t_index = max(0, min(self.T - 1, t_int - 1))
        if playback_controller_replay and int(controller_replay_step.get("t_index", t_index)) != int(t_index):
            raise ValueError(
                f"controller replay step order mismatch: replay t_index={controller_replay_step.get('t_index')} "
                f"current t_index={t_index}"
            )
        if use_fair_controller:
            eta_t = self._get_effective_fair_score_eta(t_index=t_index)
            sp_k_value = self._get_effective_fair_score_k(t_index=t_index)
            eta_check = float(eta_t.detach().cpu())
            k_check = float(sp_k_value.detach().cpu())
        else:
            base_eta = float(sp_eta) if sp_eta is not None else 0.0
            eta_t = self._sp_eta_at_step(base_eta, t_int, sp_eta_schedule)
            sp_k_value = float(sp_k) if sp_k is not None else 1.0
            eta_check = float(eta_t)
            k_check = float(sp_k_value)

        sp_shift_enabled = bool(sp_shift) and eta_check > 0.0 and k_check > 0.0 and not record_controller_replay
        if (sp_shift_enabled or record_controller_replay or playback_controller_replay) and not hasattr(self, "_fair_score_q"):
            labels = edge_group_labels if edge_group_labels is not None else s_0
            self._init_score_sp_state(labels, E_t)

        n_same = getattr(self, "_fair_N1", torch.zeros((), device=device)).to(device=device)
        n_diff = getattr(self, "_fair_N0", torch.zeros((), device=device)).to(device=device)
        has_both_groups = bool((n_same > 0).item() and (n_diff > 0).item())
        sp_step_stats = None
        batch_traces = []
        if sp_stats is not None and (sp_shift_enabled or record_controller_replay):
            sp_step_stats = {
                "t": int(t_int),
                "t_index": int(t_index),
                "eta_t": float(eta_check),
                "sp_k": float(k_check),
                "n_same": int(n_same.detach().cpu().item()),
                "n_diff": int(n_diff.detach().cpu().item()),
                "shift_applied": bool(sp_shift_enabled and has_both_groups),
            }

        replay_step = None
        if record_controller_replay:
            replay_step = {
                "t": int(t_int),
                "t_index": int(t_index),
                "edge_ids": torch.arange(len(self.src), dtype=torch.long),
                "z_raw": torch.empty(len(self.src), dtype=self.E_marginal.dtype),
                "edge_sample_gumbel": torch.empty(len(self.src), self.num_classes_E, dtype=self.E_marginal.dtype),
            }

        start = 0
        replay_z_raw = None
        if playback_controller_replay:
            replay_z_raw = controller_replay_step["z_raw"].to(device=device, dtype=E_prob.dtype)
            if replay_z_raw.numel() != len(self.src):
                raise ValueError(
                    f"controller replay z_raw length {replay_z_raw.numel()} does not match "
                    f"num full edges {len(self.src)}"
                )
            if "edge_sample_gumbel" not in controller_replay_step:
                raise ValueError("controller replay is missing edge_sample_gumbel; re-record it with the current code.")

        for batch_edge_index in edge_data_loader:
            # (B, 2)
            batch_edge_index = batch_edge_index.to(device)
            batch_dst, batch_src = batch_edge_index.T
            end = start + batch_edge_index.size(0)
            if playback_controller_replay:
                batch_z_raw = replay_z_raw[start:end]
                logit_E = torch.stack([torch.zeros_like(batch_z_raw), batch_z_raw], dim=1)
            else:
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

            if record_controller_replay:
                z_raw = (logit_E[:, 1] - logit_E[:, 0]).detach().cpu()
                replay_step["z_raw"][start:end] = z_raw.to(dtype=replay_step["z_raw"].dtype)

            if sp_shift_enabled:
                edge_ids = torch.arange(start, end, device=device)
                log_model_prob_E, trace = self._apply_score_sp_guidance(
                    logit_E,
                    edge_ids,
                    eta_t,
                    sp_k_value,
                    sp_shift_clip,
                )
                batch_pred_E = log_model_prob_E.exp()
                if sp_step_stats is not None:
                    batch_traces.append(trace)
            else:
                batch_pred_E = logit_E.softmax(dim=-1)

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
            if has_both_groups and hasattr(self, "_fair_score_R1"):
                mean_same_after = self._fair_score_R1.to(device=device) / safe_n_same
                mean_diff_after = self._fair_score_R0.to(device=device) / safe_n_diff
                delta_sp_after = mean_same_after - mean_diff_after
            else:
                mean_same_after = torch.zeros((), device=device)
                mean_diff_after = torch.zeros((), device=device)
                delta_sp_after = torch.zeros((), device=device)

            if batch_traces:
                delta_vals = [
                    float(trace["fair_score_delta_sp"].detach().cpu().item())
                    for trace in batch_traces
                ]
                shift_vals = [
                    float(trace["fair_score_mean_abs_logit_shift"])
                    for trace in batch_traces
                ]
                sp_step_stats.update({
                    "delta_sp_before": float(sum(delta_vals) / len(delta_vals)),
                    "abs_delta_sp_before": float(sum(abs(v) for v in delta_vals) / len(delta_vals)),
                    "fair_score_delta_sp_first": float(delta_vals[0]),
                    "fair_score_delta_sp_last": float(delta_vals[-1]),
                    "fair_score_mean_abs_logit_shift": float(sum(shift_vals) / len(shift_vals)),
                })
            else:
                sp_step_stats.update({
                    "delta_sp_before": 0.0,
                    "abs_delta_sp_before": 0.0,
                    "fair_score_mean_abs_logit_shift": 0.0,
                })
            sp_step_stats.update({
                "mean_same_after_prior": float(mean_same_after.detach().cpu().item()),
                "mean_diff_after_prior": float(mean_diff_after.detach().cpu().item()),
                "delta_sp_after_prior": float(delta_sp_after.detach().cpu().item()),
                "abs_delta_sp_after_prior": abs(float(delta_sp_after.detach().cpu().item())),
            })
            sp_stats.append(sp_step_stats)

        stored_gumbel = None
        if playback_controller_replay:
            stored_gumbel = controller_replay_step["edge_sample_gumbel"]
        need_edge_sample_gumbel = record_controller_replay or playback_controller_replay
        if need_edge_sample_gumbel:
            E_t, edge_sample_gumbel = self.sample_E_infer(
                E_prob,
                gumbel_noise=stored_gumbel,
                return_gumbel=True,
            )
        else:
            E_t = self.sample_E_infer(E_prob)
            edge_sample_gumbel = None
        if replay_step is not None:
            replay_step["edge_sample_gumbel"] = edge_sample_gumbel.detach().cpu().to(
                dtype=replay_step["edge_sample_gumbel"].dtype
            )
            replay_step["edge_next"] = E_t[self.src, self.dst].detach().cpu().to(torch.int8)
        if replay_step is not None and controller_replay is not None:
            controller_replay["steps"].append(replay_step)

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
                 p_values,
                 fair_label_attr="y",
                 fair_score_eta=0.0,
                 fair_score_k=0.15,
                 fair_score_eta_scale=1.0,
                 fair_score_controller_train=False,
                 fair_score_learn_k=True,
                 fair_score_learn_eta=True,
                 fair_score_guidance_normalize=True,
                 fair_score_fair_loss_weight=1.0,
                 fair_score_k_tracking_loss_weight=1.0,
                 fair_score_utility_loss_weight=1.0):
        super().__init__(T=T,
                         X_marginal=X_marginal,
                         s_marginal=s_marginal,
                         y_marginal=y_marginal,
                         E_marginal=E_marginal,
                         num_nodes=num_nodes,
                         fair_label_attr=fair_label_attr,
                         fair_score_eta=fair_score_eta,
                         fair_score_k=fair_score_k,
                         fair_score_eta_scale=fair_score_eta_scale,
                         fair_score_controller_train=fair_score_controller_train,
                         fair_score_learn_k=fair_score_learn_k,
                         fair_score_learn_eta=fair_score_learn_eta,
                         fair_score_guidance_normalize=fair_score_guidance_normalize,
                         fair_score_fair_loss_weight=fair_score_fair_loss_weight,
                         fair_score_k_tracking_loss_weight=fair_score_k_tracking_loss_weight,
                         fair_score_utility_loss_weight=fair_score_utility_loss_weight)
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
               sp_eta=None,
               sp_k=None,
               sp_eta_schedule="constant",
               sp_shift_clip=None,
               return_sp_stats=False,
               fair_score_sp=None,
               fair_score_eta=None,
               fair_score_k=None,
               return_controller_replay=False,
               controller_replay=None):
        """Sample a graph.

        Parameters
        ----------
        batch_size : int
            Batch size for edge prediction.
        num_workers : int
            Number of subprocesses for data loading in edge prediction.

        Returns
        -------
        X_t_one_hot : torch.Tensor of shape (F, |V|, 2)
            One-hot encoding of the generated node attributes.
        Y_0_one_hot : torch.Tensor of shape (|V|, C)
            One-hot encoding of the generated node labels.
        E_t : torch.LongTensor of shape (|V|, |V|)
            Adjacency matrix of the generated graph.
        """
        if return_controller_replay and controller_replay is not None:
            raise ValueError("return_controller_replay and controller_replay playback cannot be used together.")

        device = self.X_marginal.device
        playback_replay = controller_replay
        playback_controller_replay = controller_replay is not None
        dst, src = torch.triu_indices(self.num_nodes, self.num_nodes,
                                      offset=1, device=device)
        if fair_score_sp is not None:
            sp_shift = fair_score_sp
        if fair_score_eta is not None:
            sp_eta = fair_score_eta
        if fair_score_k is not None:
            sp_k = fair_score_k
        if sp_eta is None:
            sp_eta = self.fair_score_eta
        if sp_k is None:
            sp_k = self.fair_score_k

        use_fair_controller = bool(sp_shift and self.fair_score_controller_train)
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

        if playback_controller_replay:
            s_0 = controller_replay["s_0"].to(device=device, dtype=torch.long)
            y_replay = controller_replay.get("y_0", None)
            y_0 = y_replay.to(device=device, dtype=torch.long) if y_replay is not None else None
            E_t = controller_replay["E_init"].to(device=device, dtype=torch.long)
            X_t_one_hot = controller_replay["X_init"].to(device=device, dtype=self.X_marginal.dtype)
            edge_group_labels = controller_replay["edge_group_labels"].to(device=device, dtype=torch.long)
        else:
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
            edge_group_labels = self._select_fair_edge_group_labels(s_0, y_0)

        record_controller_replay_out = None
        if return_controller_replay or playback_controller_replay:
            self._init_score_sp_state(edge_group_labels, E_t)
        elif bool(sp_shift):
            self._init_score_sp_state(edge_group_labels, E_t)

        if not playback_controller_replay:
            # (F, |V|, 2)
            X_prior = self.X_marginal[:, None, :].expand(-1, self.num_nodes, -1) # You should change this for different initializations of X
            # (|V|, 2F)
            X_t_one_hot = self.sample_X(X_prior)

        if return_controller_replay:
            record_controller_replay_out = self._build_controller_replay_header(
                s_0=s_0,
                y_0=y_0,
                edge_group_labels=edge_group_labels,
                E_init=E_t,
                X_init=X_t_one_hot,
            )
        replay_steps = playback_replay.get("steps", []) if playback_controller_replay else None
        if playback_controller_replay and len(replay_steps) != self.T:
            raise ValueError(f"controller replay has {len(replay_steps)} steps, expected {self.T}.")

        # Iteratively sample p(D^s | D^t) for t = 1, ..., T, with s = t - 1.
        for replay_i, s in enumerate(tqdm(list(reversed(range(0, self.T))))):
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
            replay_step = replay_steps[replay_i] if playback_controller_replay else None

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
                                    sp_stats=self.last_sp_shift_stats,
                                    use_fair_controller=use_fair_controller,
                                    record_controller_replay=return_controller_replay,
                                    controller_replay=record_controller_replay_out,
                                    controller_replay_step=replay_step)

            if is_diff_X:
                if playback_controller_replay:
                    if "X_next" not in replay_step:
                        raise ValueError("controller replay is missing X_next; re-record it with the current code.")
                    X_t_one_hot = replay_step["X_next"].to(
                        device=device,
                        dtype=self.X_marginal.dtype,
                    )
                else:
                    # (|V|, F, 2)
                    pred_X = self.graph_encoder.pred_X(t_float,
                                                       X_t_one_hot,
                                                       A_t,
                                                       s_0,
                                                       y_0)
                    pred_X = pred_X.softmax(dim=-1)
                    # (F, |V|, 2)
                    pred_X = torch.transpose(pred_X, 0, 1)
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

                if return_controller_replay:
                    record_controller_replay_out["steps"][-1]["X_next"] = X_t_one_hot.detach().cpu().clone()
            elif return_controller_replay:
                record_controller_replay_out["steps"][-1]["X_next"] = X_t_one_hot.detach().cpu().clone()
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
        if return_controller_replay:
            return X_t_one_hot, s_0_one_hot, y_0_one_hot, E_t, node_orig_id, record_controller_replay_out
        if return_sp_stats:
            return X_t_one_hot, s_0_one_hot, y_0_one_hot, E_t, node_orig_id, self.last_sp_shift_stats
        return X_t_one_hot, s_0_one_hot, y_0_one_hot, E_t, node_orig_id

            

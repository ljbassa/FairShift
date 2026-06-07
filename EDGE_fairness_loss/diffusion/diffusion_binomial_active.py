import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric as pyg
from diffusion.diffusion_base import cosine_beta_schedule, log_1_min_a, log_categorical, extract
from diffusion.diffusion_binomial_vanilla import BinomialDiffusionVanilla
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def _inverse_softplus(x):
    x = torch.as_tensor(x).clamp(min=1e-8, max=80.0)
    return x + torch.log(-torch.expm1(-x))


class BinomialDiffusionActive(BinomialDiffusionVanilla):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, 
                 denoise_fn, timesteps=1000, loss_type='vb_kl', parametrization='x0',
                 final_prob_node=None, final_prob_edge=None, sample_time_method='importance', 
                 noise_schedule=cosine_beta_schedule, device='cuda',
                 # active snode predict
                 predict_s=False, sampling_stage="stage1_base",
                 active_method="topk", active_ratio=0.05, active_threshold=0.5,
                 s_loss_weight=1.0, ratio_loss_weight=0.1, s_pos_weight_cap=50.0,
                 fair_score_eta=0.0, fair_score_k=0.15, fair_score_eta_scale=1.0,
                 fair_label_attr="y",
                 fair_score_controller_train=False, controller_pretrained_ckpt=None,
                 controller_epochs=1000, controller_lr=1e-3,
                 controller_replay_num_samples=1, controller_replay_refresh=100,
                 fair_score_train_loss_weight=1.0,
                 fair_score_fair_loss_weight=1.0,
                 fair_score_k_tracking_loss_weight=1.0,
                 fair_score_utility_loss_weight=1.0,
                 fair_score_guidance_normalize=True,
                 ):
        super(BinomialDiffusionActive, self).__init__(num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps,
                 loss_type, parametrization, final_prob_node, final_prob_edge, sample_time_method, noise_schedule, device)
        self.predict_s = predict_s
        self.sampling_stage = sampling_stage
        self.active_method = active_method
        self.active_ratio = active_ratio
        self.active_threshold = active_threshold
        self.s_loss_weight = s_loss_weight
        self.ratio_loss_weight = ratio_loss_weight
        self.s_pos_weight_cap = s_pos_weight_cap
        self.fair_score_eta = fair_score_eta
        self.fair_score_k = fair_score_k
        self.fair_score_eta_scale = max(float(fair_score_eta_scale), 1e-8)
        self.fair_label_attr = fair_label_attr
        self.fair_score_controller_train = fair_score_controller_train
        self.controller_pretrained_ckpt = controller_pretrained_ckpt
        self.controller_epochs = controller_epochs
        self.controller_lr = controller_lr
        self.controller_replay_num_samples = controller_replay_num_samples
        self.controller_replay_refresh = controller_replay_refresh
        self.fair_score_train_loss_weight = fair_score_train_loss_weight
        self.fair_score_fair_loss_weight = fair_score_fair_loss_weight
        self.fair_score_k_tracking_loss_weight = fair_score_k_tracking_loss_weight
        self.fair_score_utility_loss_weight = fair_score_utility_loss_weight
        self.fair_score_guidance_normalize = fair_score_guidance_normalize
        self._init_controller_guidance_params()

    def _init_controller_guidance_params(self):
        if not self.fair_score_controller_train:
            self.register_parameter("fair_score_k_raw", None)
            self.register_parameter("fair_score_eta_raw", None)
            return

        raw_k_scalar = torch.logit(torch.tensor(float(self.fair_score_k)).clamp(1e-4, 1.0 - 1e-4))

        raw_k = raw_k_scalar.repeat(self.num_timesteps)
        # In controller mode, fair_score_eta is the base eta value.
        # fair_score_eta_raw learns a per-step positive multiplier initialized at 1.
        # fair_score_eta_scale is ignored in controller mode.
        raw_eta = torch.zeros(self.num_timesteps)

        self.fair_score_k_raw = torch.nn.Parameter(raw_k)
        self.fair_score_eta_raw = torch.nn.Parameter(raw_eta)

        assert self.fair_score_k_raw.ndim == 1
        assert self.fair_score_k_raw.numel() == self.num_timesteps
        assert self.fair_score_eta_raw.ndim == 1
        assert self.fair_score_eta_raw.numel() == self.num_timesteps

    def _get_effective_fair_score_k(self, *args, **kwargs):
        if not self.fair_score_controller_train:
            return self.fair_score_k

        # Per-step controller convention:
        # t=0 is the final reverse step A^1 -> A^0.
        # t=self.num_timesteps-1 is the first reverse step A^T -> A^{T-1}.
        effective_k_all = torch.sigmoid(self.fair_score_k_raw)
        t_graph = kwargs.get("t_graph", None)
        if t_graph is None:
            return effective_k_all

        t_index = torch.as_tensor(t_graph, device=self.fair_score_k_raw.device, dtype=torch.long)
        t_index = t_index.clamp(0, self.num_timesteps - 1)
        return effective_k_all.index_select(0, t_index.reshape(-1)).reshape(t_index.shape)

    def _get_effective_fair_score_eta(self, *args, **kwargs):
        if not self.fair_score_controller_train:
            return self.fair_score_eta

        # Per-step controller convention:
        # t=0 is the final reverse step A^1 -> A^0.
        # t=self.num_timesteps-1 is the first reverse step A^T -> A^{T-1}.
        base_eta = torch.as_tensor(
            float(self.fair_score_eta),
            device=self.fair_score_eta_raw.device,
            dtype=self.fair_score_eta_raw.dtype,
        )
        denom = F.softplus(torch.zeros((), device=self.fair_score_eta_raw.device, dtype=self.fair_score_eta_raw.dtype))
        effective_eta_all = base_eta * F.softplus(self.fair_score_eta_raw) / denom
        t_graph = kwargs.get("t_graph", None)
        if t_graph is None:
            return effective_eta_all

        t_index = torch.as_tensor(t_graph, device=self.fair_score_eta_raw.device, dtype=torch.long)
        t_index = t_index.clamp(0, self.num_timesteps - 1)
        return effective_eta_all.index_select(0, t_index.reshape(-1)).reshape(t_index.shape)

    def _get_effective_k(self, *args, **kwargs):
        return self._get_effective_fair_score_k(*args, **kwargs)

    def _get_effective_eta(self, *args, **kwargs):
        return self._get_effective_fair_score_eta(*args, **kwargs)

    def _get_full_edge_sensitive_mask(self, batched_graph):
        if hasattr(batched_graph, self.fair_label_attr):
            node_labels = getattr(batched_graph, self.fair_label_attr)
            if node_labels is None:
                return None
            node_labels = node_labels.reshape(-1)
            src = batched_graph.full_edge_index[0]
            dst = batched_graph.full_edge_index[1]
            return (node_labels[src] == node_labels[dst]).bool()
        return None

    def _init_score_sp_state_if_needed(self, batched_graph):
        device = batched_graph.full_edge_index.device
        num_full_edges = batched_graph.full_edge_index.size(1)
        num_graphs = batched_graph.num_graphs

        mask = self._get_full_edge_sensitive_mask(batched_graph)
        if mask is None:
            mask = torch.zeros(num_full_edges, dtype=torch.bool, device=device)

        self._fair_edge_sensitive_mask = mask
        self._fair_edge_batch = batched_graph.batch[batched_graph.full_edge_index[0]]

        self._fair_N1 = scatter(
            self._fair_edge_sensitive_mask.float(),
            self._fair_edge_batch,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )
        self._fair_N0 = scatter(
            (~self._fair_edge_sensitive_mask).float(),
            self._fair_edge_batch,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )

        if hasattr(batched_graph, "degree") and batched_graph.degree is not None:
            nodes_per_graph = batched_graph.nodes_per_graph.float()
            sum_deg = scatter(
                batched_graph.degree.float(),
                batched_graph.batch,
                dim=0,
                dim_size=num_graphs,
                reduce='sum',
            )
            rho_graph = sum_deg / (nodes_per_graph * (nodes_per_graph - 1) + eps)
        else:
            rho_graph = torch.full((num_graphs,), 1e-3, device=device)

        rho_graph = rho_graph.clamp(1e-4, 1.0 - 1e-4)
        rho_edge = rho_graph[self._fair_edge_batch]
        self._fair_score_h = torch.logit(rho_edge)
        self._fair_score_q = rho_edge.clone()
        self._fair_score_R1 = scatter(
            self._fair_score_q * self._fair_edge_sensitive_mask.float(),
            self._fair_edge_batch,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )
        self._fair_score_R0 = scatter(
            self._fair_score_q * (~self._fair_edge_sensitive_mask).float(),
            self._fair_edge_batch,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )

    def _build_controller_replay_header(self, batched_graph):
        return {
            "initial_log_node_attr_t": batched_graph.log_node_attr_t.detach().clone(),
            "initial_log_full_edge_attr_t": batched_graph.log_full_edge_attr_t.detach().clone(),
            "h_init": self._fair_score_h.detach().clone(),
            "q_init": self._fair_score_q.detach().clone(),
            "R1_init": self._fair_score_R1.detach().clone(),
            "R0_init": self._fair_score_R0.detach().clone(),
            "N1": self._fair_N1.detach().clone(),
            "N0": self._fair_N0.detach().clone(),
            "full_mask": self._fair_edge_sensitive_mask.detach().clone(),
            "full_batch": self._fair_edge_batch.detach().clone(),
            "num_graphs": int(batched_graph.num_graphs),
            "num_full_edges": int(batched_graph.full_edge_index.size(1)),
            "steps": [],
        }

    def _restore_controller_replay_initial_state(self, batched_graph, replay):
        for replay_name, graph_name in (
            ("initial_log_node_attr_t", "log_node_attr_t"),
            ("initial_log_full_edge_attr_t", "log_full_edge_attr_t"),
        ):
            if replay_name not in replay:
                raise ValueError(
                    f"controller replay is missing {replay_name}; "
                    "re-record the replay with the current code."
                )
            value = replay[replay_name].to(
                device=batched_graph.full_edge_index.device,
                dtype=getattr(batched_graph, graph_name).dtype,
            )
            current = getattr(batched_graph, graph_name)
            if value.shape != current.shape:
                raise ValueError(
                    f"controller replay {replay_name} shape {tuple(value.shape)} "
                    f"does not match sampled graph {graph_name} shape {tuple(current.shape)}"
                )
            setattr(batched_graph, graph_name, value.clone())

    def _restore_controller_replay_fair_state(self, replay):
        device = self.device
        self._fair_score_h = replay["h_init"].to(device=device).clone()
        self._fair_score_q = replay["q_init"].to(device=device).clone()
        self._fair_score_R1 = replay["R1_init"].to(device=device).clone()
        self._fair_score_R0 = replay["R0_init"].to(device=device).clone()
        self._fair_N1 = replay["N1"].to(device=device).clone()
        self._fair_N0 = replay["N0"].to(device=device).clone()
        self._fair_edge_sensitive_mask = replay["full_mask"].to(device=device, dtype=torch.bool).clone()
        self._fair_edge_batch = replay["full_batch"].to(device=device, dtype=torch.long).clone()

    def _log_sample_categorical_with_optional_gumbel(self, logits, num_classes, gumbel_noise=None):
        if gumbel_noise is None:
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        else:
            gumbel_noise = gumbel_noise.to(device=logits.device, dtype=logits.dtype)
            if gumbel_noise.shape != logits.shape:
                raise ValueError(
                    f"stored gumbel noise shape {tuple(gumbel_noise.shape)} "
                    f"does not match logits shape {tuple(logits.shape)}"
                )
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = torch.log(F.one_hot(sample, num_classes).float().clamp(min=1e-30))
        return log_sample, gumbel_noise

    def _compute_fair_controller_guidance(
        self,
        z_active,
        h_active,
        R1,
        R0,
        N1,
        N0,
        batch_active,
        mask_active,
        k_active,
    ):
        num_graphs = R1.size(0)
        k_active = torch.as_tensor(k_active, dtype=z_active.dtype, device=z_active.device)
        mask_float = mask_active.float()
        inv_mask_float = (~mask_active).float()

        h_pre = h_active + k_active * (z_active - h_active)
        q_prev = torch.sigmoid(h_active)
        q_pre = torch.sigmoid(h_pre)
        delta_q_pre = q_pre - q_prev

        dR1 = scatter(
            delta_q_pre * mask_float,
            batch_active,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )
        dR0 = scatter(
            delta_q_pre * inv_mask_float,
            batch_active,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )
        R1_pre = R1 + dR1
        R0_pre = R0 + dR0

        safe_N1 = torch.where(N1 > 0, N1, torch.ones_like(N1))
        safe_N0 = torch.where(N0 > 0, N0, torch.ones_like(N0))
        delta_pre = R1_pre / safe_N1 - R0_pre / safe_N0

        a_e = mask_float / safe_N1[batch_active] - inv_mask_float / safe_N0[batch_active]

        if not self.fair_score_guidance_normalize:
            step_scale_graph = 0.5 * (N1 + N0)
            step_scale_active = step_scale_graph.index_select(0, batch_active)
            a_bar = step_scale_active * a_e
            grad_raw = (
                delta_pre.index_select(0, batch_active)
                * a_bar
                * k_active
                * q_pre
                * (1.0 - q_pre)
            )
            valid_guidance = (N1 > 0) & (N0 > 0)
            grad_raw = torch.where(
                valid_guidance.index_select(0, batch_active),
                grad_raw,
                torch.zeros_like(grad_raw),
            )
            zero = grad_raw.sum() * 0.0
            grad = torch.nan_to_num(grad_raw, nan=0.0, posinf=0.0, neginf=0.0)
            grad_raw_abs_mean = grad_raw.detach().abs().mean() if grad_raw.numel() > 0 else zero.detach()
            grad_dir_abs_mean = grad.detach().abs().mean() if grad.numel() > 0 else zero.detach()
            aux = {
                "h_pre": h_pre.detach(),
                "q_prev": q_prev.detach(),
                "q_pre": q_pre.detach(),
                "delta_q_pre": delta_q_pre.detach(),
                "R1_pre": R1_pre.detach(),
                "R0_pre": R0_pre.detach(),
                "delta_pre": delta_pre.detach(),
                "a_e": a_e.detach(),
                "a_bar": a_bar.detach(),
                "step_scale_active": step_scale_active.detach(),
                "grad_raw": grad_raw.detach(),
                "grad_scale": torch.ones_like(grad).detach(),
                "grad_raw_abs_mean": grad_raw_abs_mean,
                "grad_dir_abs_mean": grad_dir_abs_mean,
                "graph_mean_abs_min": zero.detach(),
                "graph_mean_abs_mean": zero.detach(),
                "graph_mean_abs_max": zero.detach(),
            }
            return grad.detach(), aux

        grad_raw = delta_pre.index_select(0, batch_active) * a_e * k_active * q_pre * (1.0 - q_pre)
        zero = grad_raw.sum() * 0.0
        grad = grad_raw
        grad_scale = torch.ones_like(grad)
        graph_mean_abs = torch.zeros((num_graphs,), device=grad.device, dtype=grad.dtype)
        if grad.numel() > 0:
            grad_abs_sum = scatter(
                grad.detach().abs(),
                batch_active,
                dim=0,
                dim_size=num_graphs,
                reduce='sum',
            )
            active_count = scatter(
                torch.ones_like(grad),
                batch_active,
                dim=0,
                dim_size=num_graphs,
                reduce='sum',
            ).clamp_min(1.0)

            graph_mean_abs = grad_abs_sum / active_count

            # Use a very small floor only to avoid division by exact zero.
            # Do not use 1e-8 because raw SP gradients are often much smaller than that.
            scale_floor = torch.tensor(1e-30, device=grad.device, dtype=grad.dtype)
            graph_scale = torch.maximum(graph_mean_abs, scale_floor)

            grad_scale = graph_scale[batch_active]
            grad = grad / grad_scale

            # If a graph has exactly zero gradient, keep its guidance at zero.
            valid_scale = graph_mean_abs > 0
            grad = torch.where(
                valid_scale.index_select(0, batch_active),
                grad,
                torch.zeros_like(grad),
            )

            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        grad_raw_abs_mean = grad_raw.detach().abs().mean() if grad_raw.numel() > 0 else zero.detach()
        grad_dir_abs_mean = grad.detach().abs().mean() if grad.numel() > 0 else zero.detach()
        if graph_mean_abs.numel() > 0:
            graph_mean_abs_min = graph_mean_abs.detach().min()
            graph_mean_abs_mean = graph_mean_abs.detach().mean()
            graph_mean_abs_max = graph_mean_abs.detach().max()
        else:
            graph_mean_abs_min = zero.detach()
            graph_mean_abs_mean = zero.detach()
            graph_mean_abs_max = zero.detach()

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
            "grad_raw_abs_mean": grad_raw_abs_mean,
            "grad_dir_abs_mean": grad_dir_abs_mean,
            "graph_mean_abs_min": graph_mean_abs_min,
            "graph_mean_abs_mean": graph_mean_abs_mean,
            "graph_mean_abs_max": graph_mean_abs_max,
        }
        return grad.detach(), diagnostics

    def compute_fair_controller_loss_from_replay(self, replay):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = replay["h_init"].device

        h = replay["h_init"].to(device)
        dtype = h.dtype
        q = torch.sigmoid(h)
        R1 = replay["R1_init"].to(device=device, dtype=dtype)
        R0 = replay["R0_init"].to(device=device, dtype=dtype)
        N1 = replay["N1"].to(device=device, dtype=dtype)
        N0 = replay["N0"].to(device=device, dtype=dtype)
        num_graphs = int(replay.get("num_graphs", R1.size(0)))

        k_all = self._get_effective_fair_score_k().to(device=device, dtype=dtype)
        eta_all = self._get_effective_fair_score_eta().to(device=device, dtype=dtype)
        zero = (k_all.sum() + eta_all.sum()) * 0.0
        k_tracking_loss = zero
        utility_loss = zero
        mean_abs_shift_sum = zero
        mean_abs_shift_count = 0
        fair_guidance_raw_abs_sum = zero
        fair_guidance_raw_count = 0
        fair_guidance_delta_pre_abs_sum = zero
        fair_guidance_delta_pre_count = 0
        fair_guidance_eta_sum = zero
        fair_guidance_k_sum = zero
        fair_guidance_active_count = 0
        k_stat_values = []
        eta_stat_values = []

        for step in replay.get("steps", []):
            idx_active = step["active_edge_indices"].to(device=device, dtype=torch.long)
            z_raw = step["z_raw"].to(device=device, dtype=dtype)
            batch_active = step["batch_active"].to(device=device, dtype=torch.long)
            mask_active = step["mask_active"].to(device=device, dtype=torch.bool)
            t_graph = step["t_graph"].to(device=device, dtype=torch.long)

            k_graph = self._get_effective_fair_score_k(t_graph=t_graph).to(device=device, dtype=dtype)
            eta_graph = self._get_effective_fair_score_eta(t_graph=t_graph).to(device=device, dtype=dtype)
            assert k_graph.shape == t_graph.shape
            assert eta_graph.shape == t_graph.shape
            k_stat_values.append(k_graph.detach())
            eta_stat_values.append(eta_graph.detach())

            if idx_active.numel() == 0:
                continue

            k_active = k_graph.index_select(0, batch_active)
            eta_active = eta_graph.index_select(0, batch_active)
            h_active = h.index_select(0, idx_active)

            grad_dir, guidance_diagnostics = self._compute_fair_controller_guidance(
                z_active=z_raw,
                h_active=h_active,
                R1=R1,
                R0=R0,
                N1=N1,
                N0=N0,
                batch_active=batch_active,
                mask_active=mask_active,
                k_active=k_active.detach(),
            )

            z_guided = z_raw - eta_active * grad_dir
            shift_abs = (eta_active * grad_dir).abs()
            fair_guidance_raw_abs_sum = fair_guidance_raw_abs_sum + guidance_diagnostics["grad_raw"].abs().sum()
            fair_guidance_raw_count += int(guidance_diagnostics["grad_raw"].numel())
            fair_guidance_delta_pre_abs_sum = fair_guidance_delta_pre_abs_sum + guidance_diagnostics["delta_pre"].abs().sum()
            fair_guidance_delta_pre_count += int(guidance_diagnostics["delta_pre"].numel())
            fair_guidance_eta_sum = fair_guidance_eta_sum + eta_active.detach().sum()
            fair_guidance_k_sum = fair_guidance_k_sum + k_active.detach().sum()
            fair_guidance_active_count += int(idx_active.numel())

            h_new_fair = h_active + k_active.detach() * (z_guided - h_active)
            q_prev = torch.sigmoid(h_active)
            q_new_fair = torch.sigmoid(h_new_fair)

            delta_q = q_new_fair - q_prev
            R1 = R1 + scatter(
                delta_q * mask_active.float(),
                batch_active,
                dim=0,
                dim_size=num_graphs,
                reduce='sum',
            )
            R0 = R0 + scatter(
                delta_q * (~mask_active).float(),
                batch_active,
                dim=0,
                dim_size=num_graphs,
                reduce='sum',
            )

            h = h.index_copy(0, idx_active, h_new_fair)
            q = q.index_copy(0, idx_active, q_new_fair)

            # Preserve the Stage-1 link probability for the k-controller:
            # BCEWithLogits(h_track, sigmoid(z_raw)) is minimized when h_track == z_raw.
            pi_raw = torch.sigmoid(z_raw.detach())
            h_active_track = h_active.detach()
            h_track = h_active_track + k_active * (z_raw.detach() - h_active_track)
            k_tracking_loss = k_tracking_loss + F.binary_cross_entropy_with_logits(
                h_track,
                pi_raw,
            )

            # Utility is measured on the h-state transition, not directly on z.
            # z_raw and h_active differ at intermediate steps, so compare the
            # unshifted h update against the shifted h update.
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

            mean_abs_shift_sum = mean_abs_shift_sum + shift_abs.sum()
            mean_abs_shift_count += int(idx_active.numel())

        safe_N1 = torch.where(N1 > 0, N1, torch.ones_like(N1))
        safe_N0 = torch.where(N0 > 0, N0, torch.ones_like(N0))
        delta_final = R1 / safe_N1 - R0 / safe_N0
        valid_graph = (N1 > 0) & (N0 > 0)
        if valid_graph.any():
            fair_loss = 0.5 * delta_final[valid_graph].pow(2).mean()
            delta_final_abs_mean = delta_final[valid_graph].abs().mean()
        else:
            fair_loss = zero
            delta_final_abs_mean = zero.detach()

        loss = (
            self.fair_score_fair_loss_weight * fair_loss
            + self.fair_score_k_tracking_loss_weight * k_tracking_loss
            + self.fair_score_utility_loss_weight * utility_loss
        )

        if k_stat_values:
            k_stats = torch.cat([v.reshape(-1) for v in k_stat_values])
        else:
            k_stats = k_all.detach().reshape(-1)
        if eta_stat_values:
            eta_stats = torch.cat([v.reshape(-1) for v in eta_stat_values])
        else:
            eta_stats = eta_all.detach().reshape(-1)

        if mean_abs_shift_count > 0:
            mean_abs_shift = mean_abs_shift_sum / float(mean_abs_shift_count)
        else:
            mean_abs_shift = zero
        if fair_guidance_raw_count > 0:
            fair_guidance_raw_abs_mean = fair_guidance_raw_abs_sum / float(fair_guidance_raw_count)
        else:
            fair_guidance_raw_abs_mean = zero
        if fair_guidance_delta_pre_count > 0:
            fair_guidance_delta_pre_abs_mean = fair_guidance_delta_pre_abs_sum / float(fair_guidance_delta_pre_count)
        else:
            fair_guidance_delta_pre_abs_mean = zero
        if fair_guidance_active_count > 0:
            fair_guidance_eta_mean = fair_guidance_eta_sum / float(fair_guidance_active_count)
            fair_guidance_k_mean = fair_guidance_k_sum / float(fair_guidance_active_count)
        else:
            fair_guidance_eta_mean = zero
            fair_guidance_k_mean = zero

        t0 = 0
        tmid = self.num_timesteps // 2
        tlast = self.num_timesteps - 1
        k_schedule = k_all.detach().reshape(-1)
        eta_schedule = eta_all.detach().reshape(-1)
        stats = {
            "fair_controller_fair_loss": float(fair_loss.detach().cpu()),
            "fair_controller_k_tracking_loss": float(k_tracking_loss.detach().cpu()),
            "fair_controller_utility_loss": float(utility_loss.detach().cpu()),
            "fair_controller_delta_final_abs_mean": float(delta_final_abs_mean.detach().cpu()),
            "fair_controller_k_t0": float(k_schedule[t0].cpu()),
            "fair_controller_k_tmid": float(k_schedule[tmid].cpu()),
            "fair_controller_k_tlast": float(k_schedule[tlast].cpu()),
            "fair_controller_eta_t0": float(eta_schedule[t0].cpu()),
            "fair_controller_eta_tmid": float(eta_schedule[tmid].cpu()),
            "fair_controller_eta_tlast": float(eta_schedule[tlast].cpu()),
            "fair_controller_eta_mean": float(eta_stats.mean().detach().cpu()),
            "fair_controller_eta_min": float(eta_stats.min().detach().cpu()),
            "fair_controller_eta_max": float(eta_stats.max().detach().cpu()),
            "fair_controller_k_mean": float(k_stats.mean().detach().cpu()),
            "fair_controller_k_min": float(k_stats.min().detach().cpu()),
            "fair_controller_k_max": float(k_stats.max().detach().cpu()),
            "fair_controller_mean_abs_shift": float(mean_abs_shift.detach().cpu()),
            "fair_guidance_raw_abs_mean": float(fair_guidance_raw_abs_mean.detach().cpu()),
            "fair_guidance_shift_abs_mean": float(mean_abs_shift.detach().cpu()),
            "fair_guidance_delta_pre_abs_mean": float(fair_guidance_delta_pre_abs_mean.detach().cpu()),
            "fair_guidance_eta_mean": float(fair_guidance_eta_mean.detach().cpu()),
            "fair_guidance_k_mean": float(fair_guidance_k_mean.detach().cpu()),
        }
        return loss, stats

    def freeze_for_fair_controller_training(self):
        if not self.fair_score_controller_train:
            raise ValueError("fair_score_controller_train must be True for fair controller training.")
        if self.fair_score_k_raw is None:
            raise ValueError("fair_score_k_raw must not be None for fair controller training.")
        if self.fair_score_eta_raw is None:
            raise ValueError("fair_score_eta_raw must not be None for fair controller training.")
        if tuple(self.fair_score_k_raw.shape) != (self.num_timesteps,):
            raise ValueError(
                f"fair_score_k_raw must have shape [{self.num_timesteps}], "
                f"got {tuple(self.fair_score_k_raw.shape)}."
            )
        if tuple(self.fair_score_eta_raw.shape) != (self.num_timesteps,):
            raise ValueError(
                f"fair_score_eta_raw must have shape [{self.num_timesteps}], "
                f"got {tuple(self.fair_score_eta_raw.shape)}."
            )

        for param in self.parameters():
            param.requires_grad = False

        controller_params = []
        for name in ("fair_score_k_raw", "fair_score_eta_raw"):
            param = getattr(self, name, None)
            if param is not None:
                param.requires_grad = True
                controller_params.append(param)

        if len(controller_params) != 2:
            raise ValueError(
                f"Expected exactly two trainable fair-score controller parameters, got {len(controller_params)}."
            )

        self.train()
        self._denoise_fn.eval()
        print(f"[controller] trainable fair_score_k_raw shape: {tuple(self.fair_score_k_raw.shape)}")
        print(f"[controller] trainable fair_score_eta_raw shape: {tuple(self.fair_score_eta_raw.shape)}")
        return controller_params

    def get_fair_controller_state_dict(self):
        assert self.fair_score_k_raw is None or self.fair_score_k_raw.numel() == self.num_timesteps
        assert self.fair_score_eta_raw is None or self.fair_score_eta_raw.numel() == self.num_timesteps
        state = {
            "fair_score_k_mode": "per_step_sigmoid",
            "fair_score_eta_mode": "per_step_multiplier_softplus",
            "fair_score_eta_base": self.fair_score_eta,
            "fair_score_eta_scale": self.fair_score_eta_scale,
            "num_timesteps": self.num_timesteps,
        }
        for name in ("fair_score_k_raw", "fair_score_eta_raw"):
            param = getattr(self, name, None)
            state[name] = param.detach().clone() if param is not None else None
        return state

    def load_fair_controller_state_dict(self, state_dict, strict=True):
        if "controller" in state_dict:
            state_dict = state_dict["controller"]
        missing = []
        for name in ("fair_score_k_raw", "fair_score_eta_raw"):
            value = state_dict.get(name, None)
            param = getattr(self, name, None)
            if value is None:
                missing.append(name)
                if strict:
                    raise KeyError(f"Missing fair controller parameter: {name}")
                continue
            if param is None:
                missing.append(name)
                if strict:
                    raise KeyError(f"Current model does not have fair controller parameter: {name}")
                continue

            value = torch.as_tensor(value)
            if tuple(value.shape) != tuple(param.shape):
                if value.numel() == 1 and param.numel() == self.num_timesteps:
                    if strict:
                        raise ValueError(
                            f"{name} has old global scalar shape {tuple(value.shape)}, "
                            f"but current per-step controller expects shape {tuple(param.shape)}. "
                            "Old global scalar controller checkpoints are incompatible in strict mode."
                        )
                    print(
                        f"[WARN] Broadcasting old global scalar {name} checkpoint value to "
                        f"{self.num_timesteps} timesteps because strict=False."
                    )
                    value = value.reshape(1).expand_as(param)
                else:
                    raise ValueError(
                        f"Shape mismatch for {name}: checkpoint shape {tuple(value.shape)} vs "
                        f"current shape {tuple(param.shape)}."
                    )

            with torch.no_grad():
                param.copy_(value.to(device=param.device, dtype=param.dtype))

        if strict and missing:
            raise KeyError(f"Missing fair controller parameter(s): {missing}")
        return missing

    # 모델 기반 active 선택 함수
    def _p_sample_and_set_actives_model(self, batched_graph, t_node, t_edge):
        # (1) s logits 예측 (x 조건 포함, active GT 미사용)
        s_logits = self._denoise_fn(batched_graph, t_node, t_edge, mode="s_only")
        s_prob = torch.sigmoid(s_logits)

        # (2) graph별 top-k/threshold/bernoulli로 active_mask 구성
        active_mask = torch.zeros_like(s_prob, dtype=torch.bool)
        offset = 0
        for n in batched_graph.nodes_per_graph.tolist():
            prob_g = s_prob[offset:offset+n]

            if self.active_method == "topk":
                k = max(1, int(round(self.active_ratio * n)))
                idx = torch.topk(prob_g, k=k, largest=True).indices + offset
                active_mask[idx] = True
            elif self.active_method == "threshold":
                active_mask[offset:offset+n] = (prob_g >= self.active_threshold)
            else:  # bernoulli
                active_mask[offset:offset+n] = torch.bernoulli(prob_g).bool()

            offset += n

        batched_graph.active_node_indices = active_mask.nonzero(as_tuple=True)[0]

        # (3) active-active pair만 edge 후보로
        src = batched_graph.full_edge_index[0]
        dst = batched_graph.full_edge_index[1]
        batched_graph.active_edge_indices = (active_mask[src] & active_mask[dst]).nonzero(as_tuple=True)[0]

    def sample_increment(self, num_samples):
        # some bugs are in here, do not use for now.
        raise NotImplementedError
        batched_graph = self.initial_graph_sampler.sample(num_samples)
        batched_graph.to(self.device)

        num_nodes = batched_graph.nodes_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)

        edge_attr_t = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices_t = edge_attr_t.nonzero(as_tuple=True)[0]
        batched_graph.edge_index_t = batched_graph.full_edge_index[:, is_edge_indices_t]
        print()
        for t in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = None
            # p_sample variants
            degree_t = self._compute_degree(torch.ones_like(batched_graph.edge_index_t[0]), batched_graph.edge_index_t, batched_graph.num_nodes)
            log_model_prob_active = self._q_posterior_actives(batched_graph.degree, degree_t, t_node)
            active_node_masks = self.log_sample_categorical(log_model_prob_active, num_classes=2).argmax(1).bool()
            batched_graph.active_node_indices = active_node_masks.nonzero(as_tuple=True)[0]
            batched_graph.active_edge_indices = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]]).nonzero(as_tuple=True)[0] 
            
            if batched_graph.active_edge_indices.size(0) == 0:
                continue

            _, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)

            assert log_model_prob_edge.size(0) == batched_graph.active_edge_indices.size(0)

            log_out_edge_active = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)

            row = batched_graph.full_edge_index[0].index_select(0, batched_graph.active_edge_indices[log_out_edge_active.argmax(-1).bool()])
            col = batched_graph.full_edge_index[1].index_select(0, batched_graph.active_edge_indices[log_out_edge_active.argmax(-1).bool()])
            sampled_edge_indices_tmin1 = torch.stack((row,col))
            edge_index_t = torch.cat((batched_graph.edge_index_t, sampled_edge_indices_tmin1), dim=-1)
            batched_graph.edge_index_t = pyg.utils.coalesce(edge_index_t, reduce='max') 
        print()

        batched_graph.edge_index = batched_graph.edge_index_t 
        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs)
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']
        return batched_graph

    def _apply_sampling_stage(self, batched_graph, log_model_prob_edge, t_node, t_edge):
        """
        Stage hook for future multi-stage sampling.

        Stage 1 intentionally leaves the model probabilities untouched so the
        diffusion only learns to reconstruct the observed data distribution.
        A future stage 2 can override this hook with fairness-aware sampling
        without disturbing the base sampling loop or delta tracing.
        """
        if self.sampling_stage in ("stage1_base", "stage1", "base"):
            return log_model_prob_edge, {}
        raise NotImplementedError(
            f"sampling_stage={self.sampling_stage!r} is a reserved skeleton and has no implementation yet."
        )
         
    @torch.no_grad()
    def p_sample(
        self,
        batched_graph,
        t_node,
        t_edge,
        return_soft: bool = False,
        keep_zeros: bool = False,
        delta_as_sparse_adj: bool = False,
        undirected: bool = True,
        record_controller_replay: bool = False,
        controller_replay_step=None,
    ):
        """
        diffusion_active 전용 p_sample.
        항상 3개를 return:
        (log_out_node, log_out_edge, trace)

        trace에는 step 내에서 "active edge 후보"들 중 변화한 edge의 정보가 들어감.
        """

        playback_controller_replay = controller_replay_step is not None
        fair_state_needed = bool(
            record_controller_replay
            or playback_controller_replay
            or getattr(self, "_fair_guidance_active", False)
        )
        apply_fair_guidance = bool(getattr(self, "_fair_guidance_active", False))
        if fair_state_needed and not hasattr(self, "_fair_edge_batch"):
            self._init_score_sp_state_if_needed(batched_graph)

        # ---------- helper (함수 내부 local helper) ----------
        def _delta_edge_list_to_sparse_adj(edge_pairs, delta, num_nodes, undirected=True):
            device = edge_pairs.device
            if edge_pairs.numel() == 0:
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=delta.dtype, device=device),
                    size=(num_nodes, num_nodes),
                    device=device,
                ).coalesce()
            idx = edge_pairs
            val = delta
            if undirected:
                idx = torch.cat([idx, idx.flip(0)], dim=1)
                val = torch.cat([val, val], dim=0)
            return torch.sparse_coo_tensor(idx, val, size=(num_nodes, num_nodes), device=device).coalesce()
        # -----------------------------------------------------

        # 1) active node/edge 설정
        if playback_controller_replay:
            replay_active_edges = controller_replay_step["active_edge_indices"].to(
                device=batched_graph.full_edge_index.device,
                dtype=torch.long,
            )
            batched_graph.active_edge_indices = replay_active_edges
            if "active_node_indices" in controller_replay_step:
                batched_graph.active_node_indices = controller_replay_step["active_node_indices"].to(
                    device=batched_graph.full_edge_index.device,
                    dtype=torch.long,
                )
            elif replay_active_edges.numel() > 0:
                active_pairs = batched_graph.full_edge_index.index_select(1, replay_active_edges)
                batched_graph.active_node_indices = torch.unique(active_pairs.reshape(-1))
            else:
                batched_graph.active_node_indices = torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=batched_graph.full_edge_index.device,
                )
        elif self.predict_s:
            self._p_sample_and_set_actives_model(batched_graph, t_node, t_edge)
        else:
            self._p_sample_and_set_actives(batched_graph, t_node)

        assert hasattr(batched_graph, "active_node_indices")
        assert hasattr(batched_graph, "active_edge_indices")
        active_edge_indices_all = batched_graph.active_edge_indices.detach().clone()
        if playback_controller_replay:
            t_graph = controller_replay_step["t_graph"].to(device=t_node.device, dtype=torch.long)
            if t_graph.shape != (batched_graph.num_graphs,):
                raise ValueError(
                    f"controller replay t_graph shape {tuple(t_graph.shape)} "
                    f"does not match num_graphs={batched_graph.num_graphs}"
                )
        else:
            t_graph = t_node.new_full((batched_graph.num_graphs,), int(t_node[0].item()))

        # 2) active edge가 없으면: 상태 유지 + empty trace 반환
        if batched_graph.active_edge_indices.numel() == 0:
            if playback_controller_replay and "log_node_attr_tmin1" in controller_replay_step:
                log_node_empty = controller_replay_step["log_node_attr_tmin1"].to(
                    device=batched_graph.log_node_attr_t.device,
                    dtype=batched_graph.log_node_attr_t.dtype,
                )
            else:
                log_node_empty = batched_graph.log_node_attr_t
            empty = {
                "edge_pairs": batched_graph.full_edge_index.new_empty((2, 0)),
                "delta_hard": torch.empty((0,), dtype=torch.int8, device=batched_graph.full_edge_index.device),
                "active_edge_indices": batched_graph.active_edge_indices,
            }
            if record_controller_replay:
                empty["controller_replay_step"] = {
                    "t_graph": t_graph.detach().clone(),
                    "active_node_indices": batched_graph.active_node_indices.detach().clone(),
                    "active_edge_indices": active_edge_indices_all,
                    "batch_active": self._fair_edge_batch.index_select(0, active_edge_indices_all).detach().clone(),
                    "mask_active": self._fair_edge_sensitive_mask.index_select(0, active_edge_indices_all).detach().clone(),
                    "z_raw": torch.empty((0,), dtype=torch.float32, device=batched_graph.full_edge_index.device),
                    "y_next": torch.empty((0,), dtype=torch.long, device=batched_graph.full_edge_index.device),
                    "prev_label": torch.empty((0,), dtype=torch.long, device=batched_graph.full_edge_index.device),
                    "edge_sample_gumbel": torch.empty((0, self.num_edge_classes), dtype=torch.float32, device=batched_graph.full_edge_index.device),
                    "log_node_attr_tmin1": log_node_empty.detach().clone(),
                }
            if return_soft:
                empty["delta_soft"] = torch.empty((0,), dtype=torch.float32, device=batched_graph.full_edge_index.device)
            if delta_as_sparse_adj:
                # 글로벌 배치 기준 (N_total x N_total) sparse
                empty["delta_adj"] = _delta_edge_list_to_sparse_adj(
                    empty["edge_pairs"],
                    empty["delta_soft"] if return_soft else empty["delta_hard"],
                    num_nodes=int(batched_graph.num_nodes),
                    undirected=undirected,
                )
            return log_node_empty, batched_graph.log_full_edge_attr_t, empty

        # 3) 현재 x_t에서 active edge들의 이전 상태(0/1) 저장
        prev_active = batched_graph.log_full_edge_attr_t.index_select(
            0, batched_graph.active_edge_indices
        ).argmax(-1)  # [E_active] in {0,1}

        # 4) 모델 확률 계산. Controller replay playback은 Stage-1이 저장한
        # active set과 raw logits를 그대로 사용하고, denoiser를 다시 호출하지 않는다.
        if playback_controller_replay:
            log_model_prob_node = None
            z_raw = controller_replay_step["z_raw"].to(
                device=batched_graph.full_edge_index.device,
                dtype=batched_graph.log_full_edge_attr_t.dtype,
            )
            if z_raw.numel() != batched_graph.active_edge_indices.numel():
                raise ValueError(
                    f"controller replay z_raw length {z_raw.numel()} does not match "
                    f"active edges {batched_graph.active_edge_indices.numel()}"
                )
            log_model_prob_edge = torch.stack(
                [F.logsigmoid(-z_raw), F.logsigmoid(z_raw)],
                dim=1,
            )
            stage_trace = {"controller_replay_playback": True}
        else:
            log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)
            assert log_model_prob_edge.size(0) == batched_graph.active_edge_indices.size(0)
            z_raw = (log_model_prob_edge[:, 1] - log_model_prob_edge[:, 0]).detach()

            log_model_prob_edge, stage_trace = self._apply_sampling_stage(
                batched_graph=batched_graph,
                log_model_prob_edge=log_model_prob_edge,
                t_node=t_node,
                t_edge=t_edge,
            )

        if apply_fair_guidance:
            idx_active = batched_graph.active_edge_indices
            batch_active = self._fair_edge_batch.index_select(0, idx_active)
            mask_active = self._fair_edge_sensitive_mask.index_select(0, idx_active)
            h_prev = self._fair_score_h.index_select(0, idx_active)
            q_prev = torch.sigmoid(h_prev)

            k_graph = torch.as_tensor(
                self._get_effective_fair_score_k(t_graph=t_graph),
                dtype=z_raw.dtype,
                device=z_raw.device,
            )
            eta_graph = torch.as_tensor(
                self._get_effective_fair_score_eta(t_graph=t_graph),
                dtype=z_raw.dtype,
                device=z_raw.device,
            )
            if self.fair_score_controller_train:
                assert k_graph.numel() == batched_graph.num_graphs
                assert eta_graph.numel() == batched_graph.num_graphs
                k_active = k_graph.reshape(-1).index_select(0, batch_active)
                eta_active = eta_graph.reshape(-1).index_select(0, batch_active)
            else:
                if k_graph.dim() == 0:
                    k_active = k_graph.expand_as(z_raw)
                else:
                    k_active = k_graph.reshape(-1).index_select(0, batch_active)
                if eta_graph.dim() == 0:
                    eta_active = eta_graph.expand_as(z_raw)
                else:
                    eta_active = eta_graph.reshape(-1).index_select(0, batch_active)

            grad_dir, guidance_diagnostics = self._compute_fair_controller_guidance(
                z_active=z_raw,
                h_active=h_prev,
                R1=self._fair_score_R1,
                R0=self._fair_score_R0,
                N1=self._fair_N1,
                N0=self._fair_N0,
                batch_active=batch_active,
                mask_active=mask_active,
                k_active=k_active,
            )
            z_final = z_raw - eta_active * grad_dir
            shift_abs = (eta_active * grad_dir).detach().abs()
            shift_abs_mean = shift_abs.mean() if shift_abs.numel() > 0 else z_raw.new_tensor(0.0)
            if hasattr(self, "_fair_guidance_sample_shift_abs_sum"):
                self._fair_guidance_sample_shift_abs_sum = self._fair_guidance_sample_shift_abs_sum + shift_abs.sum()
                self._fair_guidance_sample_shift_abs_count += int(shift_abs.numel())
            log_model_prob_edge = torch.stack(
                [F.logsigmoid(-z_final), F.logsigmoid(z_final)],
                dim=1,
            )

            h_new = h_prev + k_active * (z_final - h_prev)
            q_new = torch.sigmoid(h_new)
            delta_q_new = q_new - q_prev
            self._fair_score_h[idx_active] = h_new
            self._fair_score_q[idx_active] = q_new
            self._fair_score_R1 += scatter(
                delta_q_new * mask_active.float(),
                batch_active,
                dim=0,
                dim_size=batched_graph.num_graphs,
                reduce='sum',
            )
            self._fair_score_R0 += scatter(
                delta_q_new * (~mask_active).float(),
                batch_active,
                dim=0,
                dim_size=batched_graph.num_graphs,
                reduce='sum',
            )
            stage_trace.update({
                "fair_guidance_grad_dir": grad_dir.detach(),
                "fair_guidance_delta_pre": guidance_diagnostics["delta_pre"],
                "fair_guidance_grad_scale": guidance_diagnostics["grad_scale"],
                "fair_guidance_raw_abs_mean": guidance_diagnostics["grad_raw_abs_mean"],
                "fair_guidance_shift_abs_mean": shift_abs_mean.detach(),
                "fair_guidance_delta_pre_abs_mean": guidance_diagnostics["delta_pre"].detach().abs().mean(),
                "fair_guidance_eta_mean": eta_active.detach().mean(),
                "fair_guidance_k_mean": k_active.detach().mean(),
            })

        # 5) 샘플링해서 x_{t-1} 생성 (active edge만)
        if playback_controller_replay:
            if "log_node_attr_tmin1" in controller_replay_step:
                log_out_node = controller_replay_step["log_node_attr_tmin1"].to(
                    device=batched_graph.log_node_attr_t.device,
                    dtype=batched_graph.log_node_attr_t.dtype,
                )
            else:
                log_out_node = batched_graph.log_node_attr_t
            stored_gumbel = controller_replay_step.get("edge_sample_gumbel", None)
        else:
            log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)
            stored_gumbel = None
        log_out_edge_active, edge_sample_gumbel = self._log_sample_categorical_with_optional_gumbel(
            log_model_prob_edge,
            self.num_edge_classes,
            gumbel_noise=stored_gumbel,
        )

        # 전체 edge state는 clone해서 덮어쓰기 (in-place 부작용 줄이기)
        log_out_edge = batched_graph.log_full_edge_attr_t.clone()
        log_out_edge[batched_graph.active_edge_indices] = log_out_edge_active

        # 6) delta 계산: next - prev in {-1,0,1}
        next_active = log_out_edge_active.argmax(-1)                # [E_active]
        delta_hard = (next_active - prev_active).to(torch.int8)     # [-1,0,1]

        # 7) 변화 없는 0을 유지할지 말지
        mask = torch.ones_like(delta_hard, dtype=torch.bool) if keep_zeros else (delta_hard != 0)

        # [ADD] step-level counts (wandb metric 만들 때 유용)
        n_active_total = int(batched_graph.active_edge_indices.numel())
        dh_kept = delta_hard[mask]
        n_changed = int(dh_kept.numel())
        n_add = int((dh_kept == 1).sum().item())
        n_remove = int((dh_kept == -1).sum().item())

        # active pair들 (global node index)
        edge_pairs_all = batched_graph.full_edge_index.index_select(1, batched_graph.active_edge_indices)  # [2, E_active]

        trace = {
            "edge_pairs": edge_pairs_all[:, mask],                             # [2, K]
            "delta_hard": delta_hard[mask],                                    # [K]
            "active_edge_indices": batched_graph.active_edge_indices[mask],    # [K]
            # [ADD]
            "n_active_total": n_active_total,
            "n_changed": n_changed,
            "n_add": n_add,
            "n_remove": n_remove,
        }
        trace.update(stage_trace)

        if record_controller_replay:
            trace["controller_replay_step"] = {
                "t_graph": t_graph.detach().clone(),
                "active_node_indices": batched_graph.active_node_indices.detach().clone(),
                "active_edge_indices": active_edge_indices_all,
                "batch_active": self._fair_edge_batch.index_select(0, active_edge_indices_all).detach().clone(),
                "mask_active": self._fair_edge_sensitive_mask.index_select(0, active_edge_indices_all).detach().clone(),
                "z_raw": z_raw.detach().clone(),
                "y_next": next_active.detach().clone(),
                "prev_label": prev_active.detach().clone(),
                "edge_sample_gumbel": edge_sample_gumbel.detach().clone(),
                "log_node_attr_tmin1": log_out_node.detach().clone(),
            }

        # 8) soft delta (loss 설계에 더 유리한 포맷)
        if return_soft:
            # (C=2 가정, class 1이 'edge 존재')
            p_edge = log_model_prob_edge[:, 1].exp()               # [E_active]
            delta_soft = (p_edge - prev_active.float())            # [-1,1]
            trace["delta_soft"] = delta_soft[mask]                 # [K]

        # 9) sparse matrix 형태로도 제공(원하면)
        if delta_as_sparse_adj:
            chosen = trace["delta_soft"] if return_soft else trace["delta_hard"]
            trace["delta_adj"] = _delta_edge_list_to_sparse_adj(
                trace["edge_pairs"],
                chosen,
                num_nodes=int(batched_graph.num_nodes),
                undirected=undirected,
            )

        return log_out_node, log_out_edge, trace


    
    def _compute_degree(self, full_edge_attr, full_edge_index, num_nodes):
        degree = scatter(full_edge_attr, full_edge_index[0], dim=0, dim_size=num_nodes) +\
                scatter(full_edge_attr, full_edge_index[1], dim=0, dim_size=num_nodes) 
        return degree

    def _q_posterior_actives(self, degree_start, degree_t, t_node):
        tmin1 = t_node - 1
        tmin1 = torch.where(tmin1 < 0, torch.zeros_like(tmin1), tmin1)

        log_beta_t = extract(self.log_1_min_alpha, t_node, degree_start.shape)
        
        log_cumprod_alpha_t_min_1 = extract(self.log_cumprod_alpha, tmin1, degree_start.shape)

        log_1_min_cumprod_alpha_t = extract(self.log_1_min_cumprod_alpha, t_node, degree_start.shape)

        logprob_edge_t = log_beta_t + log_cumprod_alpha_t_min_1 - log_1_min_cumprod_alpha_t

        logit_edge_t = logprob_edge_t - log_1_min_a(logprob_edge_t)

        n_trials = torch.max(degree_start-degree_t, torch.zeros_like(degree_start))
        
        logprob_node_nochange_t = torch.distributions.Binomial(total_count=n_trials, logits=logit_edge_t).log_prob(torch.zeros_like(degree_start))
        logprob_node_change_t = log_1_min_a(logprob_node_nochange_t)

        unnorm_log_probs = torch.stack([logprob_node_nochange_t, logprob_node_change_t], dim=1)
        log_node_change_given_dt_given_dstart = unnorm_log_probs - unnorm_log_probs.logsumexp(1, keepdim=True) 

        return log_node_change_given_dt_given_dstart

    def _p_sample_and_set_actives(self, batched_graph, t_node):
        if self.parametrization == 'xt_prescribed_st':
            degree_t = self._compute_degree(batched_graph.log_full_edge_attr_t.argmax(1), batched_graph.full_edge_index, batched_graph.num_nodes)
            log_model_prob_active = self._q_posterior_actives(batched_graph.degree, degree_t, t_node)
            active_node_masks = self.log_sample_categorical(log_model_prob_active, num_classes=2).argmax(1).bool()
            batched_graph.active_node_indices = active_node_masks.nonzero(as_tuple=True)[0]
            batched_graph.active_edge_indices = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
                active_node_masks[batched_graph.full_edge_index[1]]).nonzero(as_tuple=True)[0] 
        elif self.parametrization == 'xt_st': 
            pass #TODO
        else:
            raise NotImplementedError

    def _q_set_actives(self, batched_graph):
        degree_tmin1 = self._compute_degree(batched_graph.log_full_edge_attr_tmin1.argmax(1), batched_graph.full_edge_index, batched_graph.num_nodes)
        degree_t = self._compute_degree(batched_graph.log_full_edge_attr_t.argmax(1), batched_graph.full_edge_index, batched_graph.num_nodes)
       
        # set up active node indices, if K nodes are active, the length of active_nodes_indices is K
        active_node_masks = degree_tmin1 > degree_t
        batched_graph.active_node_indices = active_node_masks.nonzero(as_tuple=True)[0]
        # set up active edge indices, if K nodes are active, the length of active_edges_indices is K * (K-1) // 2
        batched_graph.active_edge_indices = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]]).nonzero(as_tuple=True)[0]
        batched_graph.edge_predict_masks = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]])

    def _predict_xtmin1_given_xt_st(self, batched_graph, t_node, t_edge):
        out_node, out_edge = self._denoise_fn(batched_graph, t_node, t_edge)

        assert out_node.size(1) == self.num_node_classes
        assert out_edge.size(1) == self.num_edge_classes

        log_pred_node = F.log_softmax(out_node, dim=1)
        log_pred_edge = F.log_softmax(out_edge, dim=1)
        return log_pred_node, log_pred_edge
    

    def _compute_MC_KL_joint(self, batched_graph, t, t_node, t_edge, return_log_model_prob_edge=False):
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        active_edge_attr_tmin1 = batched_graph.log_full_edge_attr_tmin1.index_select(0, batched_graph.active_edge_indices)
        
        loss_node = 0#scatter(loss_node, batched_graph.batch, dim=-1, reduce='sum')


        cross_ent_edge = -log_categorical(active_edge_attr_tmin1, log_model_prob_edge)
       


        cross_ent_edge = scatter(cross_ent_edge, batched_graph.batch[batched_graph.full_edge_index[0].index_select(0, batched_graph.active_edge_indices)], dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)

        # recover constant term
        num_actives_edge_per_graphs = scatter(torch.ones_like(batched_graph.active_edge_indices), batched_graph.batch[batched_graph.full_edge_index[0].index_select(0, batched_graph.active_edge_indices)], dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)
        num_nodes_per_graphs = scatter(torch.ones(batched_graph.num_nodes, device=self.device), batched_graph.batch)
        num_nodes_per_graphs*(num_nodes_per_graphs-1)//2
        num_inactive_edges_per_graph = num_nodes_per_graphs*(num_nodes_per_graphs-1)//2 - num_actives_edge_per_graphs
        cross_ent_edge += 6.9078e-29 * num_inactive_edges_per_graph
        ent_edge = 6.9078e-29 * batched_graph.edges_per_graph
        loss_edge = cross_ent_edge + ent_edge

        loss = loss_node + loss_edge

        if return_log_model_prob_edge:
            return loss, log_model_prob_edge
        return loss

    def _p_pred(self, batched_graph, t_node, t_edge):
        if self.parametrization in ['x0', 'xt']:
            return super(BinomialDiffusionActive, self)._p_pred(batched_graph, t_node, t_edge)
        elif self.parametrization == 'xt_prescribed_st':
            log_model_pred_node, log_model_pred_edge = self._predict_xtmin1_given_xt_st(batched_graph, t_node=t_node, t_edge=t_edge) 
            return log_model_pred_node, log_model_pred_edge
        elif self.parametrization == 'xt_st':
            pass # TODO

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0]# + batched_graph.node_attr.shape[0]

    def _eval_loss(self, batched_graph):
        if self.loss_type in ['vb_kl', 'vb_ce_xt']:
            # this is the same as vanilla since variable st is not introduced
            return super(BinomialDiffusionActive, self)._eval_loss(batched_graph)
        else:
            b = batched_graph.num_graphs
            if self.loss_type == 'vb_ce_xt_kl_st':
                pass

            elif self.loss_type == 'vb_ce_xt_ce_st':
                pass
            
            elif self.loss_type == 'vb_ce_xt_prescribred_st':
                t, pt =  self._sample_time(b, self.device, self.sample_time_method)

                t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
                t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

                self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)
                self._q_set_actives(batched_graph)

                kl = self._compute_MC_KL_joint(batched_graph, t, t_node, t_edge)

                ce_prior = self._kl_prior(batched_graph=batched_graph)
                # Upweigh loss term of the kl
                vb_loss = kl / pt + ce_prior

                # =========================
                # [ADD] s-head pseudo-label 학습
                # =========================
                if self.predict_s:
                    # (A) pseudo label s*: q_set_actives가 만든 active_node_indices
                    s_target = torch.zeros(batched_graph.num_nodes, device=self.device)
                    s_target[batched_graph.active_node_indices] = 1.0

                    # (B) 예측 s_logits (중요: mode='s_only' -> active GT를 입력으로 쓰지 않음)
                    s_logits = self._denoise_fn(batched_graph, t_node, t_edge, mode="s_only")

                    # (C) class imbalance 보정 (pos가 희소함)
                    pos = s_target.sum()
                    neg = s_target.numel() - pos
                    pos_weight = (neg / (pos + 1e-8)).clamp(max=self.s_pos_weight_cap)

                    s_loss = F.binary_cross_entropy_with_logits(
                        s_logits, s_target, pos_weight=pos_weight
                    )

                    # (D) active 비율을 teacher에 맞추는 작은 regularizer(선택)
                    ratio_pred = torch.sigmoid(s_logits).mean()
                    ratio_tgt = s_target.mean()
                    ratio_loss = (ratio_pred - ratio_tgt).pow(2)

                    vb_loss = vb_loss + self.s_loss_weight * s_loss + self.ratio_loss_weight * ratio_loss


                batched_graph.num_entries = self._calc_num_entries(batched_graph)

                return -vb_loss 

            else:
                raise ValueError()

    def _train_loss(self, batched_graph):
        if self.loss_type in ['vb_kl', 'vb_ce_xt']:
            return super(BinomialDiffusionActive, self)._train_loss(batched_graph)
        else:
            b = batched_graph.num_graphs
            if self.loss_type == 'vb_ce_xt_kl_st':
                pass # TODO

            elif self.loss_type == 'vb_ce_xt_ce_st':
                pass # TODO
            
            elif self.loss_type == 'vb_ce_xt_prescribred_st':
                t, pt =  self._sample_time(b, self.device, self.sample_time_method)

                t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
                t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
                self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)

                self._q_set_actives(batched_graph)

                kl = self._compute_MC_KL_joint(batched_graph, t, t_node, t_edge)

                Lt2 = kl.pow(2)
                Lt2_prev = self.Lt_history.gather(dim=0, index=t)
                new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
                self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
                self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

                ce_prior = self._kl_prior(batched_graph=batched_graph)# TODO replaced it back to _ce_prior
                # Upweigh loss term of the kl
                vb_loss = kl / pt + ce_prior

                # =========================
                # [ADD] s-head pseudo-label 학습
                # =========================
                if self.predict_s:
                    # (A) pseudo label s*: q_set_actives가 만든 active_node_indices
                    s_target = torch.zeros(batched_graph.num_nodes, device=self.device)
                    s_target[batched_graph.active_node_indices] = 1.0

                    # (B) 예측 s_logits (중요: mode='s_only' -> active GT를 입력으로 쓰지 않음)
                    s_logits = self._denoise_fn(batched_graph, t_node, t_edge, mode="s_only")

                    # (C) class imbalance 보정 (pos가 희소함)
                    pos = s_target.sum()
                    neg = s_target.numel() - pos
                    pos_weight = (neg / (pos + 1e-8)).clamp(max=self.s_pos_weight_cap)

                    s_loss = F.binary_cross_entropy_with_logits(
                        s_logits, s_target, pos_weight=pos_weight
                    )

                    # (D) active 비율을 teacher에 맞추는 작은 regularizer(선택)
                    ratio_pred = torch.sigmoid(s_logits).mean()
                    ratio_tgt = s_target.mean()
                    ratio_loss = (ratio_pred - ratio_tgt).pow(2)

                    vb_loss = vb_loss + self.s_loss_weight * s_loss + self.ratio_loss_weight * ratio_loss


                batched_graph.num_entries = self._calc_num_entries(batched_graph)

                return -vb_loss
            else:
                raise ValueError()
    

    def delta_edge_list_to_sparse_adj(self, edge_pairs, delta, num_nodes, undirected=True):
        # edge_pairs: [2,K], delta: [K] (int8 or float)
        if edge_pairs.numel() == 0:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=edge_pairs.device),
                torch.empty((0,), dtype=delta.dtype, device=edge_pairs.device),
                size=(num_nodes, num_nodes),
                device=edge_pairs.device,
            ).coalesce()

        idx = edge_pairs
        val = delta
        if undirected:
            idx = torch.cat([idx, idx.flip(0)], dim=1)
            val = torch.cat([val, val], dim=0)

        return torch.sparse_coo_tensor(idx, val, size=(num_nodes, num_nodes), device=edge_pairs.device).coalesce()

    @torch.no_grad()
    def sample(self, num_samples,
            return_edge_deltas=False,
            return_soft=False,
            keep_zeros=False,
            delta_as_sparse_adj=False,
            undirected=True,
            return_controller_replay: bool = False,
            controller_replay=None):
        if return_controller_replay and controller_replay is not None:
            raise ValueError("return_controller_replay and controller_replay playback cannot be used together")
        playback_controller_replay = controller_replay is not None
        playback_replay = controller_replay
        if playback_controller_replay:
            replay_num_graphs = int(playback_replay.get("num_graphs", num_samples))
            if int(num_samples) != replay_num_graphs:
                raise ValueError(
                    f"num_samples={num_samples} does not match controller replay num_graphs={replay_num_graphs}"
                )
            if int(playback_replay.get("num_full_edges", -1)) <= 0:
                raise ValueError("controller replay is missing num_full_edges")

        batched_graph = self.initial_graph_sampler.sample(num_samples).to(self.device)
        num_nodes = batched_graph.nodes_per_graph.sum()
        num_edges = batched_graph.edges_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)
        if playback_controller_replay:
            if int(batched_graph.num_graphs) != int(playback_replay["num_graphs"]):
                raise ValueError(
                    f"sampled graph num_graphs={batched_graph.num_graphs} does not match "
                    f"controller replay num_graphs={playback_replay['num_graphs']}"
                )
            if int(batched_graph.full_edge_index.size(1)) != int(playback_replay["num_full_edges"]):
                raise ValueError(
                    f"sampled graph full edges={batched_graph.full_edge_index.size(1)} does not match "
                    f"controller replay num_full_edges={playback_replay['num_full_edges']}"
                )
            self._restore_controller_replay_initial_state(batched_graph, playback_replay)

        # Replay is always recorded from the unguided Stage-1 trajectory.
        # Normal sample() calls still apply the learned controller guidance.
        self._fair_guidance_active = bool(
            self.fair_score_controller_train
            and not return_controller_replay
        )
        controller_replay = None
        if return_controller_replay or self._fair_guidance_active:
            self._init_score_sp_state_if_needed(batched_graph)
            self._fair_guidance_sample_shift_abs_sum = torch.zeros((), device=self.device)
            self._fair_guidance_sample_shift_abs_count = 0
            if playback_controller_replay:
                self._restore_controller_replay_fair_state(playback_replay)
        if return_controller_replay:
            controller_replay = self._build_controller_replay_header(batched_graph)
        replay_steps = None
        if playback_controller_replay:
            replay_steps = playback_replay.get("steps", [])
            if len(replay_steps) != self.num_timesteps:
                raise ValueError(
                    f"controller replay has {len(replay_steps)} steps, expected {self.num_timesteps}"
                )

        edge_deltas = [] if return_edge_deltas else None

        for replay_i, t in enumerate(reversed(range(0, self.num_timesteps))):
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long)
            replay_step = replay_steps[replay_i] if playback_controller_replay else None
            if replay_step is not None:
                replay_t = int(replay_step["t_graph"].reshape(-1)[0].item())
                if replay_t != int(t):
                    raise ValueError(
                        f"controller replay step order mismatch: got t={replay_t}, expected t={int(t)}"
                    )

            if return_edge_deltas:
                log_node_attr_tmin1, log_full_edge_attr_tmin1, trace = self.p_sample(
                    batched_graph,
                    t_node,
                    t_edge,
                    return_soft=return_soft,
                    keep_zeros=keep_zeros,
                    record_controller_replay=return_controller_replay,
                    controller_replay_step=replay_step,
                )
                controller_step = trace.pop("controller_replay_step", None)
                if return_controller_replay and controller_step is not None:
                    controller_replay["steps"].append(controller_step)
                trace["t"] = int(t)

                if delta_as_sparse_adj:
                    # 글로벌 배치 기준 N_total x N_total sparse delta matrix
                    trace["delta_adj"] = self.delta_edge_list_to_sparse_adj(
                        trace["edge_pairs"],
                        trace["delta_soft"] if return_soft else trace["delta_hard"],
                        num_nodes=int(batched_graph.num_nodes),
                        undirected=undirected
                    )

                edge_deltas.append(trace)
            else:
                log_node_attr_tmin1, log_full_edge_attr_tmin1, trace = self.p_sample(
                    batched_graph,
                    t_node,
                    t_edge,
                    return_soft=return_soft,
                    keep_zeros=keep_zeros,
                    record_controller_replay=return_controller_replay,
                    controller_replay_step=replay_step,
                )
                if return_controller_replay:
                    controller_step = trace.pop("controller_replay_step", None)
                    if controller_step is not None:
                        controller_replay["steps"].append(controller_step)

            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            batched_graph.log_node_attr_t = log_node_attr_tmin1

        # 아래는 원래 DiffusionBase.sample() 후처리 그대로 (edge_index 생성 등)
        edge_attr = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr.nonzero(as_tuple=True)[0]
        batched_graph.edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        batched_graph.edge_attr = edge_attr[is_edge_indices]
        batched_graph.node_attr = batched_graph.log_node_attr_t.argmax(-1)

        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs)
        edge_slice = torch.nn.functional.pad(edge_slice, (1, 0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']

        if self._fair_guidance_active and hasattr(self, "_fair_guidance_sample_shift_abs_sum"):
            shift_count = int(getattr(self, "_fair_guidance_sample_shift_abs_count", 0))
            if shift_count > 0:
                mean_abs_shift = self._fair_guidance_sample_shift_abs_sum / float(shift_count)
            else:
                mean_abs_shift = torch.zeros((), device=self.device)
            self._last_fair_guidance_mean_abs_shift = float(mean_abs_shift.detach().cpu())
            print(f"[fair guidance] mean_abs_shift={self._last_fair_guidance_mean_abs_shift:.8g}")

        if return_edge_deltas:
            if return_controller_replay:
                return batched_graph, edge_deltas, controller_replay
            return batched_graph, edge_deltas
        if return_controller_replay:
            return batched_graph, controller_replay
        return batched_graph

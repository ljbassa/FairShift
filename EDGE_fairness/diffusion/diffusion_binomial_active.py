import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch_scatter import scatter
import torch_geometric as pyg
from diffusion.diffusion_base import cosine_beta_schedule, log_1_min_a, log_add_exp, log_categorical, index_to_log_onehot, extract
from diffusion.diffusion_binomial_vanilla import BinomialDiffusionVanilla
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


class BinomialDiffusionActive(BinomialDiffusionVanilla):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, 
                 denoise_fn, timesteps=1000, loss_type='vb_kl', parametrization='x0',
                 final_prob_node=None, final_prob_edge=None, sample_time_method='importance', 
                 noise_schedule=cosine_beta_schedule, device='cuda',
                 # active snode predict
                 predict_s=False, active_method="topk", active_ratio=0.05, active_threshold=0.5,
                 s_loss_weight=1.0, ratio_loss_weight=0.1, s_pos_weight_cap=50.0,
                 # fairness score guidance
                 fair_score_sp=False, fair_score_eta=0.0, fair_score_k=0.15,
                 fair_score_apply_sample=True, fair_label_attr="y",
                 fair_score_guidance_normalize=False,
                 fair_sensitive_value=None, fair_edge_sensitive_mode="either"
                 ):
        super(BinomialDiffusionActive, self).__init__(num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps,
                 loss_type, parametrization, final_prob_node, final_prob_edge, sample_time_method, noise_schedule, device)
        self.predict_s = predict_s
        self.active_method = active_method
        self.active_ratio = active_ratio
        self.active_threshold = active_threshold
        self.s_loss_weight = s_loss_weight
        self.ratio_loss_weight = ratio_loss_weight
        self.s_pos_weight_cap = s_pos_weight_cap

        # fairness
        self.fair_score_sp = fair_score_sp
        self.fair_score_eta = fair_score_eta
        self.fair_score_k = fair_score_k
        self.fair_score_apply_sample = fair_score_apply_sample
        self.fair_label_attr = fair_label_attr
        self.fair_score_guidance_normalize = fair_score_guidance_normalize

    
    def _get_full_edge_sensitive_mask(self, batched_graph):
        """Returns a bool tensor [E_full] for same-label edges."""
        if hasattr(batched_graph, self.fair_label_attr):
            node_labels = getattr(batched_graph, self.fair_label_attr)
            if hasattr(node_labels, "reshape"):
                node_labels = node_labels.reshape(-1)
            src = batched_graph.full_edge_index[0]
            dst = batched_graph.full_edge_index[1]
            return (node_labels[src] == node_labels[dst]).bool()
        return None

    def _shift_binary_log_probs_from_pos_logit(self, pos_logit):
        """Return [E_active, 2] log-probabilities using logsigmoid."""
        log_p1 = F.logsigmoid(pos_logit)
        log_p0 = F.logsigmoid(-pos_logit)
        return torch.stack([log_p0, log_p1], dim=1)

    def _init_score_sp_state_if_needed(self, batched_graph):
        """Initialize and cache score-SP state for sampling."""
        if not self.fair_score_sp:
            return

        device = batched_graph.full_edge_index.device
        E_full = batched_graph.full_edge_index.size(1)
        B = batched_graph.num_graphs

        # 1) Edge sensitivity mask
        mask = self._get_full_edge_sensitive_mask(batched_graph)
        if mask is None:
            mask = torch.zeros(E_full, dtype=torch.bool, device=device)
        self._fair_edge_sensitive_mask = mask

        # 2) Edge batch indices
        self._fair_edge_batch = batched_graph.batch[batched_graph.full_edge_index[0]]

        # 3) N1, N0 (counts per graph)
        n1 = scatter(self._fair_edge_sensitive_mask.float(), self._fair_edge_batch, dim=0, dim_size=B, reduce='sum')
        n0 = scatter((~self._fair_edge_sensitive_mask).float(), self._fair_edge_batch, dim=0, dim_size=B, reduce='sum')
        self._fair_N1 = n1
        self._fair_N0 = n0

        # 4) Score state (h, q) initialization
        # Use graph-level density prior rho_b
        if hasattr(batched_graph, "degree") and batched_graph.degree is not None:
            # rho_b = sum(degree) / (n*(n-1))
            nodes_per_graph = batched_graph.nodes_per_graph.float()
            sum_deg = scatter(batched_graph.degree.float(), batched_graph.batch, dim=0, dim_size=B, reduce='sum')
            rho_b = sum_deg / (nodes_per_graph * (nodes_per_graph - 1) + eps)
        else:
            rho_b = torch.full((B,), 1e-3, device=device)
        
        rho_b = rho_b.clamp(1e-4, 1.0 - 1e-4)
        
        # Expand rho_b to all full edges
        rho_e = rho_b[self._fair_edge_batch]
        self._fair_score_h = torch.logit(rho_e)
        self._fair_score_q = rho_e.clone()

        # 5) R1, R0 (sums per graph)
        self._fair_score_R1 = scatter(self._fair_score_q * self._fair_edge_sensitive_mask.float(), self._fair_edge_batch, dim=0, dim_size=B, reduce='sum')
        self._fair_score_R0 = scatter(self._fair_score_q * (~self._fair_edge_sensitive_mask).float(), self._fair_edge_batch, dim=0, dim_size=B, reduce='sum')

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
    ):
        """
        diffusion_active 전용 p_sample.
        항상 3개를 return:
        (log_out_node, log_out_edge, trace)

        trace에는 step 내에서 "active edge 후보"들 중 변화한 edge의 정보가 들어감.
        """

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
        if self.predict_s:
            self._p_sample_and_set_actives_model(batched_graph, t_node, t_edge)
        else:
            self._p_sample_and_set_actives(batched_graph, t_node)

        assert hasattr(batched_graph, "active_node_indices")
        assert hasattr(batched_graph, "active_edge_indices")

        # 2) active edge가 없으면: 상태 유지 + empty trace 반환
        if batched_graph.active_edge_indices.numel() == 0:
            empty = {
                "edge_pairs": batched_graph.full_edge_index.new_empty((2, 0)),
                "delta_hard": torch.empty((0,), dtype=torch.int8, device=batched_graph.full_edge_index.device),
                "active_edge_indices": batched_graph.active_edge_indices,
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
            return batched_graph.log_node_attr_t, batched_graph.log_full_edge_attr_t, empty

        # 3) 현재 x_t에서 active edge들의 이전 상태(0/1) 저장
        prev_active = batched_graph.log_full_edge_attr_t.index_select(
            0, batched_graph.active_edge_indices
        ).argmax(-1)  # [E_active] in {0,1}

        # 4) 모델 확률 계산
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)
        assert log_model_prob_edge.size(0) == batched_graph.active_edge_indices.size(0)

        # [FAIRNESS] Score-SP Guidance
        fair_trace = {}
        if self.fair_score_sp and self.fair_score_apply_sample:
            # z_raw = log p(1) - log p(0)
            z_raw = log_model_prob_edge[:, 1] - log_model_prob_edge[:, 0]
            
            # Active indices and metadata
            idx_active = batched_graph.active_edge_indices
            batch_active = self._fair_edge_batch[idx_active]
            mask_active = self._fair_edge_sensitive_mask[idx_active]
            
            # Previous score state for active edges
            h_prev = self._fair_score_h[idx_active]
            q_prev = self._fair_score_q[idx_active]
            
            # Candidate score update (constant k)
            h_cand = h_prev + self.fair_score_k * (z_raw - h_prev)
            q_cand = torch.sigmoid(h_cand)
            
            # Corrections for per-graph sums
            delta_q_cand = q_cand - q_prev
            c1 = scatter(delta_q_cand * mask_active.float(), batch_active, dim=0, dim_size=batched_graph.num_graphs, reduce='sum')
            c0 = scatter(delta_q_cand * (~mask_active).float(), batch_active, dim=0, dim_size=batched_graph.num_graphs, reduce='sum')
            
            # Current global score-SP
            r1_new = self._fair_score_R1 + c1
            r0_new = self._fair_score_R0 + c0
            
            # Avoid division by zero
            safe_n1 = torch.where(self._fair_N1 > 0, self._fair_N1, torch.ones_like(self._fair_N1))
            safe_n0 = torch.where(self._fair_N0 > 0, self._fair_N0, torch.ones_like(self._fair_N0))
            
            delta_sp = (r1_new / safe_n1) - (r0_new / safe_n0)
            
            # Gradient of Delta_sp w.r.t z_e
            # a_e = 1[S=1]/N1 - 1[S=0]/N0
            a_e = (mask_active.float() / safe_n1[batch_active]) - ((~mask_active).float() / safe_n0[batch_active])
            
            # sample-time step scale: s_b = (N1 + N0) / 2  (full-edge 기준)
            step_scale_graph = 0.5 * (self._fair_N1 + self._fair_N0)
            step_scale_active = step_scale_graph[batch_active]

            # scaled coefficient
            a_bar = step_scale_active * a_e

            # grad_e = Delta_sp * s_b * a_e * k * q_cand * (1 - q_cand)
            grad_e = delta_sp[batch_active] * a_bar * self.fair_score_k * q_cand * (1.0 - q_cand)
            grad_raw = grad_e

            if self.fair_score_guidance_normalize:
                grad_abs_sum = scatter(
                    grad_raw.detach().abs(),
                    batch_active,
                    dim=0,
                    dim_size=batched_graph.num_graphs,
                    reduce='sum',
                )
                active_count = scatter(
                    torch.ones_like(grad_raw),
                    batch_active,
                    dim=0,
                    dim_size=batched_graph.num_graphs,
                    reduce='sum',
                ).clamp_min(1.0)
                graph_mean_abs = grad_abs_sum / active_count

                scale_floor = torch.tensor(1e-30, device=grad_raw.device, dtype=grad_raw.dtype)
                graph_scale = torch.maximum(graph_mean_abs, scale_floor)
                grad_e = grad_raw / graph_scale[batch_active]

                valid_scale = graph_mean_abs > 0
                grad_e = torch.where(
                    valid_scale[batch_active],
                    grad_e,
                    torch.zeros_like(grad_e),
                )
                grad_e = torch.nan_to_num(grad_e, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                graph_mean_abs = None

            # Guided logit
            z_guided = z_raw - self.fair_score_eta * grad_e
            
            # No-op for graphs with N1=0 or N0=0
            valid_guidance = (self._fair_N1 > 0) & (self._fair_N0 > 0)
            z_final = torch.where(valid_guidance[batch_active], z_guided, z_raw)
            
            # Rebuild log_model_prob_edge
            log_model_prob_edge = self._shift_binary_log_probs_from_pos_logit(z_final)
            
            # Update cached score state using z_final
            h_new = h_prev + self.fair_score_k * (z_final - h_prev)
            q_new = torch.sigmoid(h_new)
            delta_q_new = q_new - q_prev
            
            self._fair_score_h[idx_active] = h_new
            self._fair_score_q[idx_active] = q_new
            
            # Update R1, R0
            dr1 = scatter(delta_q_new * mask_active.float(), batch_active, dim=0, dim_size=batched_graph.num_graphs, reduce='sum')
            dr0 = scatter(delta_q_new * (~mask_active).float(), batch_active, dim=0, dim_size=batched_graph.num_graphs, reduce='sum')
            self._fair_score_R1 += dr1
            self._fair_score_R0 += dr0
            
            # Fill fair_trace
            fair_trace["fair_score_sp_enabled"] = True
            fair_trace["fair_score_k"] = self.fair_score_k
            fair_trace["fair_score_delta_sp"] = delta_sp.detach()
            fair_trace["fair_score_C1"] = c1.detach()
            fair_trace["fair_score_C0"] = c0.detach()
            fair_trace["fair_score_mean_q_active_prev"] = q_prev.mean().item()
            fair_trace["fair_score_mean_q_active_new"] = q_new.mean().item()
            fair_trace["fair_score_step_scale_mean"] = step_scale_active.mean().item()
            fair_trace["fair_score_mean_abs_a_bar"] = a_bar.abs().mean().item()
            fair_trace["fair_score_mean_abs_logit_shift"] = (self.fair_score_eta * grad_e).abs().mean().item()
            fair_trace["fair_score_guidance_normalize"] = bool(self.fair_score_guidance_normalize)
            fair_trace["fair_score_raw_grad_abs_mean"] = grad_raw.detach().abs().mean().item()
            fair_trace["fair_score_grad_dir_abs_mean"] = grad_e.detach().abs().mean().item()
            if graph_mean_abs is not None:
                fair_trace["fair_score_graph_mean_abs_grad_min"] = graph_mean_abs.detach().min().item()
                fair_trace["fair_score_graph_mean_abs_grad_mean"] = graph_mean_abs.detach().mean().item()
                fair_trace["fair_score_graph_mean_abs_grad_max"] = graph_mean_abs.detach().max().item()

        # 5) 샘플링해서 x_{t-1} 생성 (active edge만)
        log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)
        log_out_edge_active = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)

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
        trace.update(fair_trace)

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
            undirected=True):
        batched_graph = self.initial_graph_sampler.sample(num_samples).to(self.device)
        num_nodes = batched_graph.nodes_per_graph.sum()
        num_edges = batched_graph.edges_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)

        if self.fair_score_sp:
            self._init_score_sp_state_if_needed(batched_graph)

        edge_deltas = [] if return_edge_deltas else None

        for t in reversed(range(0, self.num_timesteps)):
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long)

            if return_edge_deltas:
                log_node_attr_tmin1, log_full_edge_attr_tmin1, trace = self.p_sample(
                    batched_graph, t_node, t_edge, return_soft=return_soft, keep_zeros=keep_zeros
                )
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
                log_node_attr_tmin1, log_full_edge_attr_tmin1, _ = self.p_sample(
                    batched_graph,
                    t_node,
                    t_edge,
                    return_soft=return_soft,
                    keep_zeros=keep_zeros,
                )

            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            batched_graph.log_node_attr_t = log_node_attr_tmin1

        if return_soft and self.fair_score_sp and hasattr(self, "_fair_score_q") and hasattr(self, "_fair_score_h"):
            batched_graph.full_edge_score_prob = self._fair_score_q.detach().clone()
            batched_graph.full_edge_score_logit = self._fair_score_h.detach().clone()

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

        if return_edge_deltas:
            return batched_graph, edge_deltas
        return batched_graph

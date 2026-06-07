import math
import numpy as np
import torch_geometric as pyg
import torch
import torch_scatter
from diffusion.diffusion_base import index_to_log_onehot
from torch.nn import functional as F
from torch import nn
from torch.nn.parameter import Parameter

norm_dict = {
    'Batch': lambda d: torch.nn.BatchNorm1d(d),
    'None': lambda d: torch.nn.Identity(),
    "Inst": lambda d: pyg.nn.norm.InstanceNorm(d),
    "Graph": lambda d: pyg.nn.norm.GraphNorm(d),
}

class SelEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out


class TimeEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out
        

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NodeModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, seq_lens, n_layers=6):
        super().__init__()
        self.num_bits = num_bits
        self.max_num_nodes = max_num_nodes
        self.seq_lens = seq_lens
        self.n_layers = n_layers
        self.embedding = nn.Linear(num_bits, 64)
        self.pos_embedding = SinusoidalPosEmb(64, seq_lens)
        self.g_embedding = nn.Linear(1, 256)
        self.res_embedding = nn.Linear(1, 64)
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, x, g_v, res_count):
        x = self.embedding(x)
        g = g_v / self.max_num_nodes
        r = res_count / self.max_num_nodes

        g = self.g_embedding(g)
        r = self.res_embedding(r[..., None])
        t = torch.arange(0,x.shape[1])[None,:].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0],-1, 64)
        x = torch.cat([x, t, r], dim=-1)
        g = g[None,:,:].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (g, g))
        x = self.linear(self.dropout(x))
        return x

class BitModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(3, 64)
        self.max_num_nodes = max_num_nodes
        self.n_layers = n_layers
        self.num_bits = num_bits
        self.pos_embedding = SinusoidalPosEmb(64, num_bits)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.res_embedding = nn.Linear(1, 128) 

    def forward(self, bits, hidden_nodes, res_count):
        x = self.embedding(bits)
        r = res_count/self.max_num_nodes
        r = self.res_embedding(r)
        t = torch.arange(0,x.shape[1])[None, :].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0], -1, 64)
        x = torch.cat([x, t], dim=-1)
        hidden_nodes = torch.cat([hidden_nodes, r],dim=-1)
        hidden_nodes = hidden_nodes[None,...].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (hidden_nodes, hidden_nodes))
        x = self.linear(self.dropout(x))
        return x

class MiniAttentionLayer(torch.nn.Module):
    def __init__(self, node_dim, in_edge_dim, out_edge_dim, d_model, num_heads=2):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model*num_heads, num_heads, batch_first=True)
        self.qkv_node = torch.nn.Linear(node_dim, d_model * 3 * num_heads)
        self.qkv_edge = torch.nn.Linear(in_edge_dim, d_model * 3 * num_heads)
        self.edge_linear = torch.nn.Sequential(torch.nn.Linear(d_model * num_heads, d_model), 
                                                torch.nn.SiLU(), 
                                                torch.nn.Linear(d_model, out_edge_dim))
    def forward(self, node_us, node_vs, edges):

        # node_us/vs: (B, D)
        q_node_us, k_node_us, v_node_us = self.qkv_node(node_us).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_node_vs, k_node_vs, v_node_vs = self.qkv_node(node_vs).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_edges, k_edges, v_edges = self.qkv_edge(edges).chunk(3, -1) # (B, D*num_heads) for q/k/v

        q = torch.stack([q_node_us, q_node_vs, q_edges], 1) # (B, 3, D*num_heads)
        k = torch.stack([k_node_us, k_node_vs, k_edges], 1) # (B, 3, D*num_heads)
        v = torch.stack([v_node_us, v_node_vs, v_edges], 1) # (B, 3, D*num_heads)

        h, _ = self.multihead_attn(q, k, v)
        h_edge = h[:, -1, :]
        h_edge = self.edge_linear(h_edge)

        return h_edge

class TGNN_degree_guided(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, use_node_feat=False, edge_dropout=0.0, **kwargs) -> None:
        super().__init__()
        # ... (기존 __init__ 코드는 그대로 유지) ...
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.edge_dropout = float(edge_dropout)
        self.embedding_t = torch.nn.Linear(1, dim)
        self.embedding_0 = torch.nn.Linear(1, dim)
        self.embedding_sel = torch.nn.Embedding(2, dim)
        self.node_in = torch.torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.SiLU()
        )
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))

        self.node_out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*4, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim*2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim*2, dim*2)
        )
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )

        self.use_node_feat = use_node_feat

        # NEW: x -> dim projector (입력 dim을 몰라도 되게 LazyLinear)
        self.node_feat_in = torch.nn.LazyLinear(dim)

        # [ADD] active node prediction head (node-wise logits)
        self.predict_s = kwargs.get("predict_s", False)
        if self.predict_s:
            self.s_head = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.SiLU(),
                torch.nn.Linear(dim, 1),
            )
        
        self.node_feat_dropout = float(kwargs.get("node_feat_dropout", 0.0))
        self.node_feat_mask_prob = float(kwargs.get("node_feat_mask_prob", 0.0))
        self.node_feat_dropout_layer = torch.nn.Dropout(p=self.node_feat_dropout)
        
        self.degree_t_jitter = int(kwargs.get("degree_t_jitter", 0))
        self.degree_0_jitter = int(kwargs.get("degree_0_jitter", 0))
        self.degree_t_mask_prob = float(kwargs.get("degree_t_mask_prob", 0.0))
        self.degree_0_mask_prob = float(kwargs.get("degree_0_mask_prob", 0.0))
        self.global_context_dropout = float(kwargs.get("global_context_dropout", 0.0))
        self.global_ctx_dropout = torch.nn.Dropout(p=self.global_context_dropout)
        self.edge_mlp_chunk_size = int(kwargs.get("edge_mlp_chunk_size", 0) or 0)

    def _compute_edge_logits_chunked(self, node_repr, row, col):
        chunk_size = int(getattr(self, "edge_mlp_chunk_size", 0) or 0)
        if chunk_size <= 0 or row.numel() <= chunk_size:
            edge_emb = node_repr[row] + node_repr[col]
            return self.final_out(edge_emb)

        edge_class_chunks = []
        for start in range(0, row.numel(), chunk_size):
            end = min(start + chunk_size, row.numel())
            edge_emb_chunk = node_repr[row[start:end]] + node_repr[col[start:end]]
            edge_class_chunks.append(self.final_out(edge_emb_chunk))
        return torch.cat(edge_class_chunks, dim=0)

    def _corrupt_degree(self, degree, jitter, mask_prob):
        if self.training:
            if jitter > 0:
                noise = torch.randint(-jitter, jitter + 1, degree.shape, device=degree.device)
                degree = degree + noise
            if mask_prob > 0:
                mask = torch.bernoulli(torch.full(degree.shape, 1.0 - mask_prob, device=degree.device)).long()
                degree = degree * mask
        return degree.clamp(0, self.max_degree + 1)

    # [수정 1] 인자에 mode=None과 **kwargs를 반드시 추가해야 합니다.
    def forward(self, pyg_data, t_node, t_edge, mode=None, **kwargs):
        if mode == 'log_prob':
            return self.log_prob(pyg_data, t_node, t_edge, **kwargs)

        # (1) Compute edge_index_clean from xt exactly as before
        if hasattr(pyg_data, 'edge_index_t'):
            edge_index_clean = pyg_data.edge_index_t
        else:
            edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
            is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]
            edge_index_clean = pyg_data.full_edge_index[:, is_edge_indices]
            edge_index_clean = torch.cat([edge_index_clean, edge_index_clean.flip(0)], dim=-1)

        # Separate message-passing graph with dropout
        edge_index_mp = edge_index_clean
        if self.training and self.edge_dropout > 0.0:
            edge_index_mp, _ = pyg.utils.dropout_edge(
                edge_index_mp,
                p=self.edge_dropout,
                force_undirected=False,
                training=True,
            )

        # (2) degree features: compute from edge_index_clean, not edge_index_mp
        nodes_t_raw = pyg.utils.degree(edge_index_clean[0], num_nodes=pyg_data.num_nodes).long()
        nodes_0_raw = pyg_data.degree.long()

        # Apply corruption only during training
        nodes_t = self._corrupt_degree(nodes_t_raw, self.degree_t_jitter, self.degree_t_mask_prob)
        nodes_0 = self._corrupt_degree(nodes_0_raw, self.degree_0_jitter, self.degree_0_mask_prob)

        # Clamp and normalize for model input
        nodes_t = nodes_t.clamp(max=self.max_degree + 1).float()[..., None] / self.max_degree
        nodes_0 = nodes_0.clamp(max=self.max_degree + 1).float()[..., None] / self.max_degree

        # (3) time embedding
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)

        # =========================
        # [NEW] s_only: active logits만 예측
        # =========================
        if mode == "s_only":
            if not hasattr(self, "s_head"):
                raise RuntimeError("predict_s=True로 생성된 모델에서만 mode='s_only' 사용 가능")

            # ★ label leakage 방지: node_selection은 전부 0으로 고정
            node_selection = torch.zeros_like(nodes_t.squeeze(-1), dtype=torch.long)

            nodes = torch.cat([
                self.embedding_t(nodes_t),
                self.embedding_0(nodes_0),
                self.embedding_sel(node_selection),
            ], dim=-1)
            nodes = self.node_in(nodes)

            # node feature injection
            if self.use_node_feat and hasattr(pyg_data, "x") and pyg_data.x is not None:
                x = pyg_data.x
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                projected_x = self.node_feat_in(x.float())
                if self.training:
                    if self.node_feat_dropout > 0:
                        projected_x = self.node_feat_dropout_layer(projected_x)
                    if self.node_feat_mask_prob > 0:
                        mask = torch.bernoulli(torch.full((projected_x.size(0), 1), 1.0 - self.node_feat_mask_prob, device=projected_x.device))
                        projected_x = projected_x * mask
                nodes = nodes + projected_x

            # time 조건도 같이 주기(가장 단순한 방식: add)
            nodes_s = nodes + t

            s_logits = self.s_head(nodes_s).squeeze(-1)  # [num_nodes]
            return s_logits

        # =========================
        # 기존 edge 예측 경로 (mode None)
        # =========================
        node_selection = torch.zeros_like(nodes_t.squeeze(-1))
        if hasattr(pyg_data, "active_node_indices"):
            node_selection[pyg_data.active_node_indices] = 1
        node_selection = node_selection.long()

        nodes = torch.cat([
            self.embedding_t(nodes_t),
            self.embedding_0(nodes_0),
            self.embedding_sel(node_selection),
        ], dim=-1)
        nodes = self.node_in(nodes)

        if self.use_node_feat and hasattr(pyg_data, "x") and pyg_data.x is not None:
            x = pyg_data.x
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            projected_x = self.node_feat_in(x.float())
            if self.training:
                if self.node_feat_dropout > 0:
                    projected_x = self.node_feat_dropout_layer(projected_x)
                if self.node_feat_mask_prob > 0:
                    mask = torch.bernoulli(torch.full((projected_x.size(0), 1), 1.0 - self.node_feat_mask_prob, device=projected_x.device))
                    projected_x = projected_x * mask
            nodes = nodes + projected_x
        
        h = nodes.unsqueeze(0)
        contexts = torch_scatter.scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)
        contexts = self.global_ctx_dropout(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index_mp)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = torch_scatter.scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = self.global_ctx_dropout(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts

        # mlp add
        if hasattr(pyg_data, "active_edge_indices"):
            edge_indices = pyg_data.active_edge_indices
        else:
            edge_indices = torch.arange(pyg_data.full_edge_index.size(1), device=pyg_data.full_edge_index.device)
        row = pyg_data.full_edge_index[0].index_select(0, edge_indices)
        col = pyg_data.full_edge_index[1].index_select(0, edge_indices)

        nodes = torch.cat([nodes, self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_out_mlp(nodes)

        edge_class = self._compute_edge_logits_chunked(nodes, row, col)

        return pyg_data.log_node_attr, edge_class

    # [주의] 만약 이 클래스 안에 log_prob 함수가 없다면, 
    # forward가 호출할 대상이 없어서 또 에러가 납니다.
    # 보통 이 클래스 아래나 부모 클래스에 log_prob가 정의되어 있어야 합니다.

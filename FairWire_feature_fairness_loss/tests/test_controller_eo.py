"""Exercise the actual controller methods on CPU without loading DGL/GNNs."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fairness_surrogate", ROOT / "Model/fairness_surrogate.py")
surrogate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surrogate)


def controller_class(source=None):
    source = source or (ROOT / "Model/fair_diffusion.py").read_text()
    parsed = ast.parse(source)
    nodes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "BaseModel"]
    scope = dict(torch=torch, nn=nn, F=F,
                 group_fairness_terms=surrogate.group_fairness_terms,
                 normalize_fair_score_metric=surrogate.normalize_fair_score_metric)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "controller_methods", "exec"), scope)
    return scope["BaseModel"]


def make_model(metric="eo", normalize=True, cls=None):
    cls = cls or controller_class()
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.T = 2
    model.num_nodes = 4
    model.fair_label_attr = "y"
    model.E_marginal = torch.tensor([0.8, 0.2])
    model.src, model.dst = torch.triu_indices(4, 4, offset=1)
    model.fair_score_metric = metric
    model.fair_score_eo_min_mass = 1e-6
    model.fair_score_controller_train = True
    model.fair_score_k = 0.35
    model.fair_score_eta = 0.7
    model.fair_score_eta_scale = 1.0
    model.fair_score_guidance_normalize = normalize
    model.fair_score_learn_k = model.fair_score_learn_eta = True
    model.fair_score_fair_loss_weight = 1.0
    model.fair_score_k_tracking_loss_weight = 0.0
    model.fair_score_utility_loss_weight = 0.0
    model._init_fair_controller_params()
    return model


def make_replay(mask=None):
    mask = torch.tensor([True, False, False, False, False, True]) if mask is None else mask
    q = torch.full((6,), 0.2)
    return dict(h_init=torch.logit(q), q_init=q, R1_init=q[mask].sum(), R0_init=q[~mask].sum(),
                N1=mask.float().sum(), N0=(~mask).float().sum(), full_mask=mask,
                steps=[dict(t_index=1, edge_ids=torch.arange(6), z_raw=torch.tensor([2., -.7, .2, -.1, -1., 1.])),
                       dict(t_index=0, edge_ids=torch.arange(6), z_raw=torch.tensor([1., -.1, -.8, .3, -.5, 2.]))])


@pytest.mark.parametrize("normalize", [False, True])
def test_sp_guidance_matches_original_formula(normalize):
    model = make_model("sp", normalize)
    z = torch.tensor([2., -.7, .2, -.1, -1., 1.])
    h = torch.tensor([-.3, -.5, -.2, -.9, -.8, -.1])
    mask = make_replay()["full_mask"]
    n1, n0 = mask.sum(), (~mask).sum()
    q = torch.sigmoid(h)
    k = torch.full_like(z, .35)
    grad, _ = model._compute_fair_controller_guidance(z, h, q[mask].sum(), q[~mask].sum(), n1, n0, mask, k)
    q_pre = torch.sigmoid(h + k * (z - h))
    gap = q_pre[mask].mean() - q_pre[~mask].mean()
    expected = gap * (mask.float()/n1 - (~mask).float()/n0) * k * q_pre * (1-q_pre)
    expected = expected / expected.abs().mean() if normalize else expected * (0.5*(n1+n0))
    torch.testing.assert_close(grad, expected)


@pytest.mark.parametrize("normalize", [False, True])
def test_eo_guidance_matches_autograd_with_frozen_condition(normalize):
    model = make_model("eo", normalize)
    z = torch.tensor([2., -.7, .2, -.1, -1., 1.], requires_grad=True)
    h = torch.tensor([-.3, -.5, -.2, -.9, -.8, -.1])
    condition = torch.tensor([-.5, -.2, -.4, -.9, -.3, -.1], requires_grad=True)
    mask = make_replay()["full_mask"]
    q, w = h.sigmoid(), condition.detach().sigmoid()
    k = torch.full_like(z, .35)
    w_pre = (condition.detach() + k * (z.detach()-condition.detach())).sigmoid()
    q_pre = (h + k * (z-h)).sigmoid()
    gap = (w_pre*q_pre)[mask].sum()/w_pre[mask].sum() - (w_pre*q_pre)[~mask].sum()/w_pre[~mask].sum()
    expected, = torch.autograd.grad(0.5*gap.square(), z)
    grad, diagnostic = model._compute_fair_controller_guidance(
        z, h, q[mask].sum(), q[~mask].sum(), mask.sum(), (~mask).sum(), mask, k,
        condition_h_active=condition, C1=w[mask].sum(), C0=w[~mask].sum(),
        U1=(w*q)[mask].sum(), U0=(w*q)[~mask].sum())
    expected = expected/expected.abs().mean() if normalize else expected*(0.5*w_pre.sum())
    torch.testing.assert_close(grad, expected)
    assert not grad.requires_grad
    assert not diagnostic["condition_h_pre"].requires_grad
    assert not diagnostic["w_pre"].requires_grad
    assert condition.grad is None


@pytest.mark.parametrize("normalize", [False, True])
def test_eo_replay_learns_eta_but_does_not_move_positive_condition(normalize):
    model = make_model("eo", normalize)
    loss, stats = model.compute_fair_controller_loss_from_replay(make_replay())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.fair_score_eta_raw.grad).all()
    assert model.fair_score_eta_raw.grad.abs().sum() > 0
    assert model.fair_score_k_raw.grad.abs().sum() == 0
    assert stats["fair_controller_eo_positive_mass_same"] > 0
    model.fair_score_k_tracking_loss_weight = 1.0
    model.zero_grad()
    loss, _ = model.compute_fair_controller_loss_from_replay(make_replay())
    loss.backward()
    assert model.fair_score_k_raw.grad.abs().sum() > 0


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("invalid", ["same_missing", "diff_missing", "tiny_mass"])
def test_eo_invalid_groups_have_zero_fair_loss_and_gradient(normalize, invalid):
    model = make_model("eo", normalize)
    mask = torch.zeros(6, dtype=torch.bool) if invalid == "same_missing" else torch.ones(6, dtype=torch.bool)
    replay = make_replay(mask if invalid != "tiny_mass" else None)
    if invalid == "tiny_mass":
        model.fair_score_eo_min_mass = 1e6
    loss, stats = model.compute_fair_controller_loss_from_replay(replay)
    loss.backward()
    assert loss.item() == 0
    assert stats["fair_controller_mean_abs_shift"] == 0
    assert stats["fair_controller_eo_gap_final_abs_mean"] == 0
    assert torch.equal(model.fair_score_eta_raw.grad, torch.zeros_like(model.fair_score_eta_raw))


def test_eo_sampling_incremental_buffers_match_full_reduction():
    model = make_model()
    model._init_score_sp_state(torch.tensor([0, 0, 1, 1]), torch.ones(4, 4)-torch.eye(4))
    for edges, z in [(torch.tensor([0, 2, 5]), torch.tensor([1.2, -.7, 1.8])),
                     (torch.tensor([1, 3, 4]), torch.tensor([-.4, .2, -.3])),
                     (torch.tensor([0, 1, 5]), torch.tensor([.6, -.9, 1.5]))]:
        logits = torch.stack([torch.zeros_like(z), z], dim=-1)
        _, trace = model._apply_score_sp_guidance(logits, edges, .7, .35)
        w, q, mask = model._fair_condition_h.sigmoid(), model._fair_score_q, model._fair_edge_sensitive_mask
        torch.testing.assert_close(model._fair_score_C1, w[mask].sum())
        torch.testing.assert_close(model._fair_score_C0, w[~mask].sum())
        torch.testing.assert_close(model._fair_score_U1, (w*q)[mask].sum())
        torch.testing.assert_close(model._fair_score_U0, (w*q)[~mask].sum())
        assert "fair_score_delta_eo" in trace
        assert "fair_score_delta_sp" not in trace


def test_checkpoint_restores_eo_and_legacy_defaults_to_sp(tmp_path):
    model = make_model("eo")
    model.fair_score_eo_min_mass = .003
    saved = model.get_fair_controller_state_dict()
    path = tmp_path / "controller.pt"
    torch.save({"controller": saved}, path)
    restored = make_model("sp")
    restored.load_fair_controller_state_dict(torch.load(path, weights_only=True))
    assert restored.fair_score_metric == "eo"
    assert restored.fair_score_eo_min_mass == .003
    torch.testing.assert_close(restored.fair_score_k_raw, model.fair_score_k_raw)
    torch.testing.assert_close(restored.fair_score_eta_raw, model.fair_score_eta_raw)
    del saved["fair_score_metric"], saved["fair_score_eo_min_mass"]
    restored.load_fair_controller_state_dict(saved)
    assert restored.fair_score_metric == "sp"
    assert restored.fair_score_eo_min_mass == 1e-6

import argparse
import unittest
from types import SimpleNamespace

import torch

from diffusion.diffusion_binomial_active import BinomialDiffusionActive
from diffusion.fairness_surrogate import group_fairness_terms
from datasets.evaluator import compute_edge_score_eo_stats_from_components
from model import add_model_args


def fixture(metric='sp', normalized=False, one_group=False):
    model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
    torch.nn.Module.__init__(model)
    model.fair_score_metric = metric
    model.fair_score_eo_min_mass = 1e-6
    model.fair_score_sp = True
    model.fair_score_apply_sample = True
    model.fair_score_guidance_normalize = normalized
    model.fair_score_k = 0.35
    model.fair_score_eta = 0.7
    model.fair_label_attr = 'y'
    model.predict_s = False
    model.num_node_classes = model.num_edge_classes = 2
    edges = torch.triu_indices(4, 4, offset=1)
    graph = SimpleNamespace(
        full_edge_index=edges, batch=torch.zeros(4, dtype=torch.long),
        y=torch.tensor([0, 0, 0, 0] if one_group else [0, 0, 1, 1]),
        num_graphs=1, num_nodes=4, degree=torch.ones(4),
        nodes_per_graph=torch.tensor([4]),
        active_node_indices=torch.arange(4), active_edge_indices=torch.arange(6),
        log_full_edge_attr_t=torch.tensor([[0., -70.]]).repeat(6, 1),
    )
    model._p_sample_and_set_actives = lambda *args: None
    z = torch.tensor([1.7, -0.6, -1.3, 0.3, -0.8, 1.1])
    model._p_pred = lambda *args: (
        torch.log_softmax(torch.zeros(4, 2), dim=1),
        model._shift_binary_log_probs_from_pos_logit(z[graph.active_edge_indices]),
    )
    model._init_score_sp_state_if_needed(graph)
    return model, graph, z


class EOGuidanceTest(unittest.TestCase):
    def test_metric_defaults_and_choices(self):
        parser = argparse.ArgumentParser()
        add_model_args(parser)
        self.assertEqual(parser.parse_args([]).fair_score_metric, 'sp')
        self.assertEqual(parser.parse_args(['--fair_score_metric', 'eo']).fair_score_metric, 'eo')

    def test_sp_guidance_matches_original_formula(self):
        for normalized in (False, True):
            model, graph, z = fixture(normalized=normalized)
            h = model._fair_score_h.clone()
            q = torch.sigmoid(h + model.fair_score_k * (z - h))
            same = model._fair_edge_sensitive_mask
            gap = q[same].mean() - q[~same].mean()
            derivative = same.float() / same.sum() - (~same).float() / (~same).sum()
            grad = gap * 3 * derivative * model.fair_score_k * q * (1 - q)
            if normalized:
                grad = grad / grad.abs().mean()
            expected_q = torch.sigmoid(h + model.fair_score_k * (z - model.fair_score_eta * grad - h))
            model.p_sample(graph, torch.zeros(4, dtype=torch.long), None)
            torch.testing.assert_close(model._fair_score_q, expected_q)

    def test_eo_guidance_and_cached_sums_follow_weighted_formula(self):
        model, graph, z = fixture('eo')
        for active in (torch.arange(6), torch.tensor([0, 1, 4])):
            graph.active_edge_indices = active
            q = model._fair_score_q.clone()
            h = model._fair_score_h.clone()
            w = model._fair_condition_w.clone()
            wh = model._fair_condition_h.clone()
            q[active] = torch.sigmoid(h[active] + model.fair_score_k * (z[active] - h[active]))
            w[active] = torch.sigmoid(wh[active] + model.fair_score_k * (z[active] - wh[active]))
            same = model._fair_edge_sensitive_mask
            mass1, mass0 = w[same].sum(), w[~same].sum()
            gap = (w[same] * q[same]).sum() / mass1 - (w[~same] * q[~same]).sum() / mass0
            derivative = w[active] * (same[active].float() / mass1 - (~same[active]).float() / mass0)
            grad = gap * (mass1 + mass0) * 0.5 * derivative * model.fair_score_k * q[active] * (1 - q[active])
            expected = torch.sigmoid(h[active] + model.fair_score_k * (z[active] - model.fair_score_eta * grad - h[active]))
            _, _, trace = model.p_sample(graph, torch.zeros(4, dtype=torch.long), None)
            torch.testing.assert_close(model._fair_score_q[active], expected)
            torch.testing.assert_close(trace['fair_score_delta_eo'], gap.reshape(1))
            torch.testing.assert_close(model._fair_condition_w, w)
            torch.testing.assert_close(model._fair_score_U1, (w[same] * model._fair_score_q[same]).sum().reshape(1))
            torch.testing.assert_close(model._fair_score_U0, (w[~same] * model._fair_score_q[~same]).sum().reshape(1))
            self.assertFalse(model._fair_condition_w.requires_grad)

    def test_eo_empty_group_or_insufficient_mass_has_no_shift(self):
        for one_group, threshold in ((True, 1e-6), (False, 100.0)):
            model, graph, z = fixture('eo', normalized=True, one_group=one_group)
            model.fair_score_eo_min_mass = threshold
            h = model._fair_score_h.clone()
            expected = torch.sigmoid(h + model.fair_score_k * (z - h))
            _, _, trace = model.p_sample(graph, torch.zeros(4, dtype=torch.long), None)
            torch.testing.assert_close(model._fair_score_q, expected)
            self.assertEqual(trace['fair_score_mean_abs_logit_shift'], 0.0)

    def test_eo_surrogate_derivative_matches_autograd(self):
        q = torch.tensor([.8, .3, .5, .1], requires_grad=True)
        w = torch.tensor([.9, .2, .6, .4])
        same = torch.tensor([True, True, False, False])
        moment = lambda x: x.sum().reshape(1)
        terms = group_fairness_terms(
            'eo', moment(q[same]), moment(q[~same]),
            moment(w[same]), moment(w[~same]),
            moment((w*q)[same]), moment((w*q)[~same]),
            moment(same.float()), moment((~same).float()), q_active=q,
            positive_weight_active=w, batch_active=torch.zeros(4, dtype=torch.long), same_active=same,
        )
        derivative, = torch.autograd.grad(terms['gap'].sum(), q)
        torch.testing.assert_close(derivative, terms['derivative'])

    def test_eo_evaluation_respects_kept_nodes(self):
        model, graph, _ = fixture('eo')
        result = compute_edge_score_eo_stats_from_components(
            graph.full_edge_index, torch.tensor([.8, .1, .4, .3, .2, .9]),
            torch.ones(6), graph.y, kept_nodes=[0, 1, 2],
        )
        self.assertAlmostEqual(result['fair_edge_score_eo_gap'], .8 - (.1 + .3) / 2, places=6)


if __name__ == '__main__':
    unittest.main()

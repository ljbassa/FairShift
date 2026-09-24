import unittest

import torch

from diffusion.diffusion_binomial_active import BinomialDiffusionActive
from diffusion.fairness_surrogate import group_fairness_terms


def _moments(q, same, positive_weight=None):
    same_float = same.to(q.dtype)
    diff_float = (~same).to(q.dtype)
    if positive_weight is None:
        positive_weight = q.detach()
    return (
        (q * same_float).sum().reshape(1),
        (q * diff_float).sum().reshape(1),
        (positive_weight * same_float).sum().reshape(1),
        (positive_weight * diff_float).sum().reshape(1),
        (positive_weight * q * same_float).sum().reshape(1),
        (positive_weight * q * diff_float).sum().reshape(1),
        same_float.sum().reshape(1),
        diff_float.sum().reshape(1),
    )


def _batched_moments(q, same, batch, positive_weight):
    num_graphs = int(batch.max().item()) + 1
    same_float = same.to(q.dtype)
    diff_float = (~same).to(q.dtype)

    def reduce(values):
        out = torch.zeros(num_graphs, dtype=q.dtype)
        return out.index_add(0, batch, values)

    return (
        reduce(q * same_float),
        reduce(q * diff_float),
        reduce(positive_weight * same_float),
        reduce(positive_weight * diff_float),
        reduce(positive_weight * q * same_float),
        reduce(positive_weight * q * diff_float),
        reduce(same_float),
        reduce(diff_float),
    )


class FairnessSurrogateTest(unittest.TestCase):
    def test_sp_formula_remains_group_mean_gap(self):
        q = torch.tensor([0.9, 0.3, 0.4, 0.2], dtype=torch.float64, requires_grad=True)
        same = torch.tensor([True, True, False, False])
        batch = torch.zeros(q.numel(), dtype=torch.long)
        terms = group_fairness_terms(
            "sp",
            *_moments(q, same),
            q_active=q,
            batch_active=batch,
            same_active=same,
        )
        self.assertAlmostEqual(terms["gap"].item(), 0.3)
        torch.testing.assert_close(
            terms["derivative"],
            torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float64),
        )

    def test_raw_sp_guidance_matches_legacy_formula(self):
        model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
        torch.nn.Module.__init__(model)
        model.fair_score_metric = "sp"
        model.fair_score_eo_min_mass = 1e-6
        model.fair_score_guidance_normalize = False
        q_prev = torch.full((4,), 0.2, dtype=torch.float64)
        h_prev = torch.logit(q_prev)
        z = torch.logit(torch.tensor([0.9, 0.7, 0.4, 0.2], dtype=torch.float64))
        same = torch.tensor([True, True, False, False])
        batch = torch.zeros(4, dtype=torch.long)
        k = torch.full((4,), 0.5, dtype=torch.float64)
        moments = _moments(q_prev, same, q_prev)
        actual, _ = model._compute_fair_controller_guidance(
            z_active=z,
            h_active=h_prev,
            condition_h_active=h_prev,
            R1=moments[0], R0=moments[1], C1=moments[2], C0=moments[3],
            U1=moments[4], U0=moments[5], N1=moments[6], N0=moments[7],
            batch_active=batch,
            mask_active=same,
            k_active=k,
        )
        q_cand = torch.sigmoid(h_prev + k * (z - h_prev))
        r_same = q_cand[same].sum()
        r_diff = q_cand[~same].sum()
        gap = r_same / 2.0 - r_diff / 2.0
        step_scale = 0.5 * (2.0 + 2.0)
        derivative = torch.where(same, torch.tensor(0.5), torch.tensor(-0.5)).to(torch.float64)
        expected = gap * (step_scale * derivative) * k * q_cand * (1.0 - q_cand)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_eo_is_soft_positive_conditioned_score_gap(self):
        q = torch.tensor([0.9, 0.3, 0.4, 0.2], dtype=torch.float64)
        same = torch.tensor([True, True, False, False])
        positive_weight = torch.tensor([0.8, 0.2, 0.6, 0.3], dtype=torch.float64)
        terms = group_fairness_terms("eo", *_moments(q, same, positive_weight))

        expected_same = (0.8 * 0.9 + 0.2 * 0.3) / (0.8 + 0.2)
        expected_diff = (0.6 * 0.4 + 0.3 * 0.2) / (0.6 + 0.3)
        self.assertAlmostEqual(terms["gap"].item(), expected_same - expected_diff)
        self.assertTrue(terms["valid_graph"].item())

    def test_eo_analytic_derivative_matches_autograd(self):
        q = torch.tensor([0.82, 0.37, 0.55, 0.18], dtype=torch.float64, requires_grad=True)
        same = torch.tensor([True, True, False, False])
        positive_weight = torch.tensor([0.7, 0.2, 0.8, 0.1], dtype=torch.float64)
        moments = _moments(q, same, positive_weight)
        terms = group_fairness_terms(
            "eo",
            *moments,
            q_active=q,
            positive_weight_active=positive_weight,
            batch_active=torch.zeros(q.numel(), dtype=torch.long),
            same_active=same,
        )
        analytic = terms["derivative"].detach().clone()
        terms["gap"].sum().backward()
        torch.testing.assert_close(analytic, q.grad, rtol=1e-10, atol=1e-12)

    def test_eo_conditioner_is_stop_gradient(self):
        q = torch.tensor([0.8, 0.4, 0.5, 0.2], dtype=torch.float64, requires_grad=True)
        conditioner = torch.tensor([0.7, 0.2, 0.6, 0.1], dtype=torch.float64, requires_grad=True)
        same = torch.tensor([True, True, False, False])
        w = conditioner.detach()
        terms = group_fairness_terms(
            "eo",
            *_moments(q, same, w),
            q_active=q,
            positive_weight_active=w,
            batch_active=torch.zeros(4, dtype=torch.long),
            same_active=same,
        )
        (0.5 * terms["gap"].pow(2).sum()).backward()
        self.assertIsNone(conditioner.grad)
        self.assertIsNotNone(q.grad)
        self.assertGreater(q.grad.abs().sum().item(), 0.0)

    def test_eo_guidance_step_reduces_squared_gap(self):
        q = torch.tensor([0.9, 0.7, 0.35, 0.2], dtype=torch.float64)
        same = torch.tensor([True, True, False, False])
        positive_weight = torch.tensor([0.8, 0.3, 0.7, 0.2], dtype=torch.float64)
        batch = torch.zeros(q.numel(), dtype=torch.long)
        terms = group_fairness_terms(
            "eo",
            *_moments(q, same, positive_weight),
            q_active=q,
            positive_weight_active=positive_weight,
            batch_active=batch,
            same_active=same,
        )
        old_loss = 0.5 * terms["gap"].pow(2)
        grad_logit = terms["gap"].index_select(0, batch) * terms["derivative"] * q * (1.0 - q)
        q_new = torch.sigmoid(torch.logit(q) - 0.1 * grad_logit)
        new_terms = group_fairness_terms("eo", *_moments(q_new, same, positive_weight))
        new_loss = 0.5 * new_terms["gap"].pow(2)
        self.assertLess(new_loss.item(), old_loss.item())

    def test_missing_group_is_a_finite_noop(self):
        q = torch.tensor([0.7, 0.4], dtype=torch.float64)
        same = torch.tensor([True, True])
        positive_weight = torch.tensor([0.8, 0.3], dtype=torch.float64)
        batch = torch.zeros(q.numel(), dtype=torch.long)
        terms = group_fairness_terms(
            "eo",
            *_moments(q, same, positive_weight),
            q_active=q,
            positive_weight_active=positive_weight,
            batch_active=batch,
            same_active=same,
        )
        self.assertFalse(terms["valid_graph"].item())
        torch.testing.assert_close(terms["gap"], torch.zeros_like(terms["gap"]))
        torch.testing.assert_close(terms["derivative"], torch.zeros_like(q))
        self.assertTrue(torch.isfinite(terms["derivative"]).all())

    def test_negligible_positive_mass_is_a_finite_noop(self):
        score_same = torch.tensor([0.5], dtype=torch.float64)
        score_diff = torch.tensor([1.0], dtype=torch.float64)
        positive_mass_same = torch.tensor([2e-12], dtype=torch.float64)
        positive_mass_diff = torch.tensor([1.0], dtype=torch.float64)
        positive_score_same = torch.tensor([1e-12], dtype=torch.float64)
        positive_score_diff = torch.tensor([0.5], dtype=torch.float64)
        counts = torch.tensor([2.0], dtype=torch.float64)
        terms = group_fairness_terms(
            "eo",
            score_same,
            score_diff,
            positive_mass_same,
            positive_mass_diff,
            positive_score_same,
            positive_score_diff,
            counts,
            counts,
            min_positive_mass=1e-6,
        )
        self.assertFalse(terms["valid_graph"].item())
        self.assertEqual(terms["gap"].item(), 0.0)

    def test_mixed_valid_invalid_batch_is_isolated_and_finite(self):
        q = torch.tensor([0.8, 0.5, 0.4, 0.2, 0.7, 0.3], dtype=torch.float64)
        w = torch.tensor([0.6, 0.2, 0.5, 0.1, 0.8, 0.4], dtype=torch.float64)
        same = torch.tensor([True, True, False, False, True, True])
        batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
        terms = group_fairness_terms(
            "eo",
            *_batched_moments(q, same, batch, w),
            q_active=q,
            positive_weight_active=w,
            batch_active=batch,
            same_active=same,
        )
        torch.testing.assert_close(terms["valid_graph"], torch.tensor([True, False]))
        self.assertNotEqual(terms["gap"][0].item(), 0.0)
        self.assertEqual(terms["gap"][1].item(), 0.0)
        torch.testing.assert_close(terms["derivative"][batch == 1], torch.zeros(2, dtype=torch.float64))
        self.assertTrue(torch.isfinite(terms["derivative"]).all())

    def _guidance_fixture(self, normalize, z_tail_shift=0.0):
        model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
        torch.nn.Module.__init__(model)
        model.fair_score_metric = "eo"
        model.fair_score_eo_min_mass = 1e-6
        model.fair_score_guidance_normalize = normalize
        q = torch.full((8,), 0.2)
        h = torch.logit(q)
        same = torch.tensor([True, True, False, False, True, True, False, False])
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        w = q.clone()
        moments = _batched_moments(q, same, batch, w)
        z = torch.logit(torch.tensor([0.9, 0.7, 0.4, 0.2, 0.6, 0.3, 0.8, 0.1]))
        z[4:] += z_tail_shift
        grad, diagnostics = model._compute_fair_controller_guidance(
            z_active=z,
            h_active=h,
            condition_h_active=h,
            R1=moments[0], R0=moments[1], C1=moments[2], C0=moments[3],
            U1=moments[4], U0=moments[5], N1=moments[6], N0=moments[7],
            batch_active=batch,
            mask_active=same,
            k_active=torch.full((8,), 0.5),
        )
        return grad, diagnostics, batch

    def test_normalized_and_raw_guidance_are_finite_and_batch_isolated(self):
        normalized, norm_diag, batch = self._guidance_fixture(True)
        raw, raw_diag, _ = self._guidance_fixture(False)
        shifted, _, _ = self._guidance_fixture(True, z_tail_shift=4.0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.isfinite(raw).all())
        self.assertFalse(torch.allclose(normalized, raw))
        torch.testing.assert_close(normalized[batch == 0], shifted[batch == 0])
        self.assertIn("support_same", norm_diag)
        self.assertIn("invalid_graph_fraction", raw_diag)

    def test_eo_controller_replay_loss_backpropagates(self):
        model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
        torch.nn.Module.__init__(model)
        model.num_timesteps = 2
        model.fair_score_controller_train = True
        model.fair_score_k = 0.5
        model.fair_score_eta = 0.05
        model.fair_score_eta_scale = 1.0
        model.fair_score_k_raw = torch.nn.Parameter(torch.zeros(2))
        model.fair_score_eta_raw = torch.nn.Parameter(torch.zeros(2))
        model.fair_score_metric = "eo"
        model.fair_score_eo_min_mass = 1e-6
        model.fair_score_guidance_normalize = True
        model.fair_score_fair_loss_weight = 1.0
        model.fair_score_k_tracking_loss_weight = 0.01
        model.fair_score_utility_loss_weight = 0.1

        q0 = torch.full((4,), 0.2)
        same = torch.tensor([True, True, False, False])
        replay = {
            "h_init": torch.logit(q0),
            "q_init": q0,
            "R1_init": torch.tensor([0.4]),
            "R0_init": torch.tensor([0.4]),
            "U1_init": torch.tensor([0.08]),
            "U0_init": torch.tensor([0.08]),
            "N1": torch.tensor([2.0]),
            "N0": torch.tensor([2.0]),
            "full_mask": same,
            "full_batch": torch.zeros(4, dtype=torch.long),
            "num_graphs": 1,
            "steps": [
                {
                    "active_edge_indices": torch.arange(4),
                    "z_raw": torch.logit(torch.tensor([0.9, 0.7, 0.35, 0.2])),
                    "batch_active": torch.zeros(4, dtype=torch.long),
                    "mask_active": same,
                    "t_graph": torch.tensor([1]),
                }
            ],
        }

        loss, stats = model.compute_fair_controller_loss_from_replay(replay)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["fair_score_metric"], "eo")
        self.assertEqual(stats["fair_controller_valid_graphs"], 1)
        self.assertGreater(stats["fair_controller_gap_final_abs_mean"], 0.0)
        self.assertTrue(torch.isfinite(model.fair_score_k_raw.grad).all())
        self.assertTrue(torch.isfinite(model.fair_score_eta_raw.grad).all())
        self.assertGreater(int((model.fair_score_eta_raw.grad != 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()

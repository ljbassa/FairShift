import contextlib
import io
import itertools
import unittest

import torch

from diffusion.diffusion_binomial_active import BinomialDiffusionActive


MODES = tuple(itertools.product(("per_step", "shared"), ("per_step", "fixed_one")))


def make_controller(eta_mode="per_step", k_mode="per_step", metric="sp", timesteps=3):
    return BinomialDiffusionActive(
        num_node_classes=2,
        num_edge_classes=2,
        initial_graph_sampler=None,
        denoise_fn=torch.nn.Linear(2, 2),
        timesteps=timesteps,
        final_prob_node=[0.5, 0.5],
        final_prob_edge=[0.5, 0.5],
        device="cpu",
        fair_score_controller_train=True,
        fair_score_eta=0.05,
        fair_score_k=0.5,
        fair_score_eta_mode=eta_mode,
        fair_score_k_mode=k_mode,
        fair_score_metric=metric,
        fair_score_fair_loss_weight=1.0,
        fair_score_k_tracking_loss_weight=0.01,
        fair_score_utility_loss_weight=0.1,
    )


def freeze_controller(model):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.freeze_for_fair_controller_training()


def make_replay():
    q_init = torch.full((4,), 0.2)
    same = torch.tensor([True, True, False, False])
    batch = torch.zeros(4, dtype=torch.long)
    return {
        "h_init": torch.logit(q_init),
        "q_init": q_init,
        "R1_init": torch.tensor([0.4]),
        "R0_init": torch.tensor([0.4]),
        "U1_init": torch.tensor([0.08]),
        "U0_init": torch.tensor([0.08]),
        "N1": torch.tensor([2.0]),
        "N0": torch.tensor([2.0]),
        "full_mask": same,
        "full_batch": batch,
        "num_graphs": 1,
        "steps": [
            {
                "active_edge_indices": torch.arange(4),
                "z_raw": torch.logit(torch.tensor(probabilities)),
                "batch_active": batch,
                "mask_active": same,
                "t_graph": torch.tensor([timestep]),
            }
            for timestep, probabilities in (
                (2, [0.9, 0.7, 0.35, 0.2]),
                (1, [0.85, 0.65, 0.3, 0.15]),
                (0, [0.8, 0.6, 0.3, 0.2]),
            )
        ],
    }


class ControllerAblationTest(unittest.TestCase):
    def test_parameter_counts_and_frozen_backbone_for_all_modes(self):
        for eta_mode, k_mode in MODES:
            with self.subTest(eta_mode=eta_mode, k_mode=k_mode):
                model = make_controller(eta_mode, k_mode)
                parameters = freeze_controller(model)
                expected_eta = 1 if eta_mode == "shared" else model.num_timesteps
                expected_k = 0 if k_mode == "fixed_one" else model.num_timesteps
                self.assertEqual(model.fair_score_eta_raw.shape, (expected_eta,))
                self.assertEqual(sum(parameter.numel() for parameter in parameters), expected_eta + expected_k)
                self.assertEqual(
                    {id(parameter) for parameter in parameters},
                    {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
                )
                self.assertFalse(model._denoise_fn.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in model._denoise_fn.parameters()))
                if k_mode == "fixed_one":
                    self.assertIsNone(model.fair_score_k_raw)
                    self.assertNotIn("fair_score_k_raw", dict(model.named_parameters()))
                else:
                    self.assertEqual(model.fair_score_k_raw.shape, (model.num_timesteps,))

    def test_effective_schedules_preserve_timestep_query_shape(self):
        queries = (torch.tensor(1), torch.tensor([2, 0, 1]), torch.tensor([[2, 1], [0, 2]]))
        for eta_mode, k_mode in MODES:
            with self.subTest(eta_mode=eta_mode, k_mode=k_mode):
                model = make_controller(eta_mode, k_mode).double()
                for getter, expected in (
                    (model._get_effective_fair_score_eta, 0.05),
                    (model._get_effective_fair_score_k, 1.0 if k_mode == "fixed_one" else 0.5),
                ):
                    schedule = getter()
                    self.assertEqual(schedule.shape, (model.num_timesteps,))
                    self.assertEqual(schedule.dtype, torch.float64)
                    torch.testing.assert_close(schedule, torch.full_like(schedule, expected))
                    for query in queries:
                        result = getter(t_graph=query)
                        self.assertEqual(result.shape, query.shape)
                        torch.testing.assert_close(result, schedule[query])

    def test_shared_eta_accumulates_timestep_gradients_and_stays_shared(self):
        shared = make_controller("shared").double()
        per_step = make_controller("per_step").double()
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        (shared._get_effective_fair_score_eta() * weights).sum().backward()
        (per_step._get_effective_fair_score_eta() * weights).sum().backward()
        torch.testing.assert_close(shared.fair_score_eta_raw.grad, per_step.fair_score_eta_raw.grad.sum().reshape(1))
        self.assertGreater(shared.fair_score_eta_raw.grad.abs().item(), 0.0)

        before = shared._get_effective_fair_score_eta().detach().clone()
        torch.optim.SGD([shared.fair_score_eta_raw], lr=0.1).step()
        after = shared._get_effective_fair_score_eta().detach()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.equal(after, after[0].expand_as(after)))

    def test_fixed_k_stays_exactly_one_after_optimizer_updates(self):
        for eta_mode in ("per_step", "shared"):
            with self.subTest(eta_mode=eta_mode):
                model = make_controller(eta_mode, "fixed_one")
                optimizer = torch.optim.Adam(freeze_controller(model), lr=0.02)
                for _ in range(3):
                    k = model._get_effective_fair_score_k()
                    self.assertTrue(torch.equal(k, torch.ones_like(k)))
                    self.assertFalse(k.requires_grad)
                    optimizer.zero_grad()
                    loss, _ = model.compute_fair_controller_loss_from_replay(make_replay())
                    loss.backward()
                    optimizer.step()
                k = model._get_effective_fair_score_k()
                self.assertTrue(torch.equal(k, torch.ones_like(k)))
                self.assertIsNone(model.fair_score_k_raw)

    def test_sp_and_eo_replay_backpropagate_for_all_modes(self):
        for metric, (eta_mode, k_mode) in itertools.product(("sp", "eo"), MODES):
            with self.subTest(metric=metric, eta_mode=eta_mode, k_mode=k_mode):
                model = make_controller(eta_mode, k_mode, metric)
                freeze_controller(model)
                loss, stats = model.compute_fair_controller_loss_from_replay(make_replay())
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(stats["fair_score_metric"], metric)
                self.assertEqual(stats["fair_controller_valid_graphs"], 1)
                eta_grad = model.fair_score_eta_raw.grad
                self.assertIsNotNone(eta_grad)
                self.assertTrue(torch.isfinite(eta_grad).all())
                self.assertGreater(eta_grad.abs().sum().item(), 0.0)
                self.assertTrue(all(parameter.grad is None for parameter in model._denoise_fn.parameters()))
                if k_mode == "per_step":
                    self.assertTrue(torch.isfinite(model.fair_score_k_raw.grad).all())
                    self.assertGreater(model.fair_score_k_raw.grad.abs().sum().item(), 0.0)

    def test_checkpoint_roundtrip_preserves_modes_and_schedules(self):
        metadata_pairs = set()
        for eta_mode, k_mode in MODES:
            with self.subTest(eta_mode=eta_mode, k_mode=k_mode):
                model = make_controller(eta_mode, k_mode, "eo")
                with torch.no_grad():
                    model.fair_score_eta_raw.add_(0.4)
                    if model.fair_score_k_raw is not None:
                        model.fair_score_k_raw.add_(0.2)
                state = model.get_fair_controller_state_dict()
                metadata_pairs.add((state["fair_score_eta_mode"], state["fair_score_k_mode"]))
                self.assertEqual(state["num_timesteps"], 3)
                self.assertEqual(state["fair_score_eta_raw"].numel(), 1 if eta_mode == "shared" else 3)
                if k_mode == "fixed_one":
                    self.assertIsNone(state["fair_score_k_raw"])

                checkpoint = io.BytesIO()
                torch.save({"controller": state}, checkpoint)
                checkpoint.seek(0)
                restored = make_controller(eta_mode, k_mode)
                restored.load_fair_controller_state_dict(torch.load(checkpoint, weights_only=True))
                self.assertEqual(restored.fair_score_metric, "eo")
                self.assertEqual(restored.fair_score_eta_mode, eta_mode)
                self.assertEqual(restored.fair_score_k_mode, k_mode)
                torch.testing.assert_close(restored._get_effective_fair_score_eta(), model._get_effective_fair_score_eta())
                torch.testing.assert_close(restored._get_effective_fair_score_k(), model._get_effective_fair_score_k())
        self.assertEqual(len(metadata_pairs), 4)

    def test_strict_checkpoint_rejects_each_mode_mismatch(self):
        for source_modes, destination_modes in itertools.product(MODES, MODES):
            if source_modes == destination_modes:
                continue
            with self.subTest(source=source_modes, destination=destination_modes):
                # T=1 gives shared and per-step eta the same tensor shape; mode
                # validation must still keep the two experiment types distinct.
                source = make_controller(*source_modes, timesteps=1)
                destination = make_controller(*destination_modes, timesteps=1)
                with self.assertRaises(ValueError):
                    destination.load_fair_controller_state_dict(source.get_fair_controller_state_dict())

    def test_legacy_per_step_checkpoint_metadata_and_missing_modes_load(self):
        model = make_controller()
        state = model.get_fair_controller_state_dict()
        state["fair_score_eta_raw"] = torch.tensor([0.1, 0.2, 0.3])
        state["fair_score_k_raw"] = torch.tensor([-0.2, 0.1, 0.4])
        for with_metadata in (True, False):
            with self.subTest(with_metadata=with_metadata):
                legacy = dict(state)
                if with_metadata:
                    legacy["fair_score_eta_mode"] = "per_step_multiplier_softplus"
                    legacy["fair_score_k_mode"] = "per_step_sigmoid"
                else:
                    legacy.pop("fair_score_eta_mode")
                    legacy.pop("fair_score_k_mode")
                restored = make_controller()
                restored.load_fair_controller_state_dict(legacy)
                torch.testing.assert_close(restored.fair_score_eta_raw, state["fair_score_eta_raw"])
                torch.testing.assert_close(restored.fair_score_k_raw, state["fair_score_k_raw"])


if __name__ == "__main__":
    unittest.main()

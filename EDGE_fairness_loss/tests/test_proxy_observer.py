"""Small CPU tests of the production guided loop with a toy denoiser."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from diffusion.diffusion_binomial_active import BinomialDiffusionActive


def _model(metric="eo", empty_t=None, same_only=False):
    model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.num_timesteps = 20
    model.num_node_classes = 2
    model.num_edge_classes = 2
    model.predict_s = False
    model.sampling_stage = "stage1_base"
    model.fair_score_controller_train = True
    model.fair_score_k = 0.4
    model.fair_score_eta = 0.8
    model.fair_score_metric = metric
    model.fair_score_eo_min_mass = 1e-6
    model.fair_score_guidance_normalize = True
    model.fair_label_attr = "y"
    model._init_controller_guidance_params()

    def initial_sample(num_samples):
        assert num_samples == 1
        pairs = torch.triu_indices(4, 4, offset=1)
        graph = Data(
            num_nodes=4,
            full_edge_index=pairs,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            nodes_per_graph=torch.tensor([4]),
            edges_per_graph=torch.tensor([6]),
            degree=torch.ones(4),
            y=torch.zeros(4, dtype=torch.long) if same_only else torch.tensor([0, 0, 1, 1]),
            log_node_attr_t=F.one_hot(torch.zeros(4, dtype=torch.long), 2).float().clamp_min(1e-30).log(),
            log_full_edge_attr_t=F.one_hot(torch.zeros(6, dtype=torch.long), 2).float().clamp_min(1e-30).log(),
        )
        return Batch.from_data_list([graph])

    def set_actives(graph, t_node):
        # Exercise the real sampler's RNG ordering and partial-cache updates.
        indices = torch.randperm(6)[:4]
        if int(t_node[0]) == empty_t:
            indices = indices[:0]
        graph.active_edge_indices = indices
        graph.active_node_indices = torch.unique(graph.full_edge_index[:, indices])

    def predict(graph, t_node, t_edge):
        indices = graph.active_edge_indices
        previous = graph.log_full_edge_attr_t.argmax(-1).float()
        same = graph.y[graph.full_edge_index[0]] == graph.y[graph.full_edge_index[1]]
        # Depend on sampled trajectory as well as sensitive relation and time.
        z = (same.float() * 3.0 - 1.5 + previous * 0.6 + float(t_node[0]) * 0.01)[indices]
        model._test_last_z = z.clone()
        return torch.zeros((4, 2)).log_softmax(-1), torch.stack([F.logsigmoid(-z), F.logsigmoid(z)], -1)

    model.initial_graph_sampler = SimpleNamespace(sample=initial_sample)
    model._prepare_data_for_sampling = lambda graph: graph
    model._p_sample_and_set_actives = set_actives
    model._p_pred = predict
    return model


def _sample(model, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.sample(1, **kwargs)


class ProxyObserverTest(unittest.TestCase):
    def test_on_off_preserves_generated_graph_cache_and_rng_for_sp_and_eo(self):
        for metric in ("sp", "eo"):
            for return_deltas in (False, True):
                with self.subTest(metric=metric, return_deltas=return_deltas):
                    off = _model(metric)
                    torch.manual_seed(817)
                    graph_off = _sample(off, return_edge_deltas=return_deltas)
                    rng_off = torch.get_rng_state().clone()
                    on = _model(metric)
                    snapshots = []
                    torch.manual_seed(817)
                    graph_on = _sample(on, proxy_observer=snapshots.append, return_edge_deltas=return_deltas)
                    if return_deltas:
                        graph_off, traces_off = graph_off
                        graph_on, traces_on = graph_on
                        for before, after in zip(traces_off, traces_on):
                            torch.testing.assert_close(before["delta_hard"], after["delta_hard"], rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng_off, rtol=0, atol=0)
                    for attr in ("edge_index", "log_full_edge_attr_t", "log_node_attr_t"):
                        torch.testing.assert_close(getattr(graph_on, attr), getattr(graph_off, attr), rtol=0, atol=0)
                    for attr in ("_fair_score_h", "_fair_score_q", "_fair_condition_h", "_fair_condition_w",
                                 "_fair_score_R1", "_fair_score_R0", "_fair_condition_C1", "_fair_condition_C0",
                                 "_fair_score_U1", "_fair_score_U0"):
                        torch.testing.assert_close(getattr(on, attr), getattr(off, attr), rtol=0, atol=0)
                    self.assertEqual([s["loop_index"] for s in snapshots], [4, 9, 17])
                    self.assertEqual([s["diffusion_t"] for s in snapshots], [15, 10, 2])
                    self.assertEqual([s["progress"] for s in snapshots], [.25, .5, .9])
                    for snapshot in snapshots:
                        self.assertEqual(snapshot["pair_ids"].shape, (2, 6))
                        self.assertEqual("w" in snapshot, metric == "eo")
                        for value in snapshot.values():
                            if isinstance(value, torch.Tensor):
                                self.assertEqual(value.device.type, "cpu")
                                self.assertFalse(value.requires_grad)

    def test_eo_observer_uses_guided_history_and_separate_unshifted_condition(self):
        model = _model("eo")
        observed = []

        def observer(snapshot):
            active = snapshot["active_pair_indices"]
            k = model._get_effective_fair_score_k(t_graph=torch.tensor([snapshot["diffusion_t"]]))
            expected_q = model._fair_score_q.clone()
            expected_q[active] = torch.sigmoid(
                model._fair_score_h[active] + k * (model._test_last_z - model._fair_score_h[active])
            )
            expected_w = model._fair_condition_w.clone()
            expected_w[active] = torch.sigmoid(
                model._fair_condition_h[active] + k.detach() * (model._test_last_z - model._fair_condition_h[active])
            )
            torch.testing.assert_close(snapshot["q"], expected_q, rtol=0, atol=0)
            torch.testing.assert_close(snapshot["w"], expected_w, rtol=0, atol=0)
            self.assertFalse(torch.equal(snapshot["q"], snapshot["w"]))
            same = snapshot["same_mask"]
            q, w = snapshot["q"].double(), snapshot["w"].double()
            gap = (w[same] * q[same]).sum() / w[same].sum() - (w[~same] * q[~same]).sum() / w[~same].sum()
            torch.testing.assert_close(snapshot["production_proxy"].double()[0], gap, rtol=2e-5, atol=2e-6)
            observed.append(snapshot)

        torch.manual_seed(817)
        _sample(model, proxy_observer=observer)
        self.assertEqual(len(observed), 3)

    def test_observer_can_mutate_cpu_record_without_mutating_live_state(self):
        off = _model("eo")
        torch.manual_seed(48)
        baseline = _sample(off)
        on = _model("eo")

        def mutate(snapshot):
            for value in snapshot.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()

        torch.manual_seed(48)
        observed = _sample(on, proxy_observer=mutate)
        torch.testing.assert_close(observed.log_full_edge_attr_t, baseline.log_full_edge_attr_t, rtol=0, atol=0)
        torch.testing.assert_close(on._fair_score_q, off._fair_score_q, rtol=0, atol=0)
        torch.testing.assert_close(on._fair_condition_w, off._fair_condition_w, rtol=0, atol=0)

    def test_empty_active_selected_stage_is_observed_without_state_update(self):
        model = _model("eo", empty_t=10)
        snapshots = []

        def observer(snapshot):
            if snapshot["diffusion_t"] == 10:
                self.assertFalse(snapshot["guidance_applied"])
                self.assertEqual(snapshot["active_pair_indices"].numel(), 0)
                torch.testing.assert_close(snapshot["q"], model._fair_score_q, rtol=0, atol=0)
                torch.testing.assert_close(snapshot["w"], model._fair_condition_w, rtol=0, atol=0)
            snapshots.append(snapshot)

        torch.manual_seed(817)
        _sample(model, proxy_observer=observer)
        self.assertEqual(len(snapshots), 3)

    def test_invalid_proxy_is_nan_but_production_value_is_preserved(self):
        for metric in ("sp", "eo"):
            model = _model(metric, same_only=True)
            snapshots = []
            torch.manual_seed(817)
            _sample(model, proxy_observer=snapshots.append)
            self.assertEqual(len(snapshots), 3)
            for snapshot in snapshots:
                self.assertFalse(snapshot["valid_graph"].item())
                self.assertTrue(torch.isnan(snapshot["proxy_gap"]).all())
                self.assertTrue(torch.isfinite(snapshot["production_proxy"]).all())

    def test_observer_rejects_replay_unguided_and_ambiguous_stages(self):
        model = _model()
        for kwargs in ({"return_controller_replay": True}, {"controller_replay": {}},
                       {"proxy_observer_progress": [0.25, 0.25]}, {"proxy_observer_progress": [float("nan")]}):
            with self.assertRaises(ValueError):
                _sample(model, proxy_observer=lambda _: None, **kwargs)
        model.fair_score_controller_train = False
        with self.assertRaises(ValueError):
            _sample(model, proxy_observer=lambda _: None)


if __name__ == "__main__":
    unittest.main()

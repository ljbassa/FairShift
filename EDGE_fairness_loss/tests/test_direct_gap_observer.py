"""Read-only direct-gap events through the real guided sampling loop on CPU."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from diffusion.diffusion_binomial_active import BinomialDiffusionActive


torch.set_num_threads(2)


def make_model(metric="sp", empty_final=False):
    model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.num_timesteps = 4
    model.num_node_classes = model.num_edge_classes = 2
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
    model.fair_score_k_raw.requires_grad_(False)
    model.forward_calls = 0

    def initial_sample(count):
        graphs = []
        for _ in range(count):
            graphs.append(Data(
                num_nodes=4,
                full_edge_index=torch.triu_indices(4, 4, offset=1),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                nodes_per_graph=torch.tensor([4]),
                edges_per_graph=torch.tensor([6]),
                degree=torch.ones(4),
                y=torch.tensor([0, 0, 1, 1]),
                log_node_attr_t=F.one_hot(torch.zeros(4, dtype=torch.long), 2).float().clamp_min(1e-30).log(),
                log_full_edge_attr_t=F.one_hot(torch.zeros(6, dtype=torch.long), 2).float().clamp_min(1e-30).log(),
            ))
        return Batch.from_data_list(graphs)

    def set_actives(graph, t_node):
        time = int(t_node[0])
        # Entry 5 is never visited; other entries become inactive again.
        indices = torch.tensor({3: [0, 1], 2: [2, 3], 1: [0, 4], 0: [0, 1]}[time])
        if empty_final and time == 0:
            indices = indices[:0]
        graph.active_edge_indices = torch.cat([indices + 6 * b for b in range(graph.num_graphs)])
        graph.active_node_indices = torch.unique(graph.full_edge_index[:, graph.active_edge_indices])

    def predict(graph, t_node, t_edge):
        model.forward_calls += 1
        same = graph.y[graph.full_edge_index[0]] == graph.y[graph.full_edge_index[1]]
        previous = graph.log_full_edge_attr_t.argmax(-1).float()
        z = (3.0 * same.float() - 1.5 + 0.6 * previous + 0.1 * float(t_node[0]))[graph.active_edge_indices]
        model.last_raw_z = z.clone()
        return torch.zeros((graph.num_nodes, 2)).log_softmax(-1), torch.stack([F.logsigmoid(-z), F.logsigmoid(z)], -1)

    model.initial_graph_sampler = SimpleNamespace(sample=initial_sample)
    model._prepare_data_for_sampling = lambda graph: graph
    model._p_sample_and_set_actives = set_actives
    model._p_pred = predict
    return model


def sample(model, count=1, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.sample(count, **kwargs)


class DirectGapObserverTest(unittest.TestCase):
    def test_on_off_identical_graph_cache_rng_and_forward_count(self):
        for metric in ("sp", "eo"):
            for deltas in (False, True):
                with self.subTest(metric=metric, deltas=deltas):
                    off, on = make_model(metric), make_model(metric)
                    torch.manual_seed(829)
                    baseline = sample(off, return_edge_deltas=deltas)
                    baseline_rng = torch.get_rng_state().clone()
                    records = []
                    torch.manual_seed(829)
                    observed = sample(on, return_edge_deltas=deltas, direct_gap_observer=records.append)
                    if deltas:
                        baseline, baseline_deltas = baseline
                        observed, observed_deltas = observed
                        for left, right in zip(baseline_deltas, observed_deltas):
                            torch.testing.assert_close(left["delta_hard"], right["delta_hard"], rtol=0, atol=0)
                    self.assertEqual(on.forward_calls, off.forward_calls)
                    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
                    for name in ("edge_index", "log_full_edge_attr_t", "log_node_attr_t"):
                        torch.testing.assert_close(getattr(observed, name), getattr(baseline, name), rtol=0, atol=0)
                    for name in ("_fair_score_h", "_fair_score_q", "_fair_condition_h", "_fair_condition_w",
                                 "_fair_score_R1", "_fair_score_R0", "_fair_condition_C1", "_fair_condition_C0",
                                 "_fair_score_U1", "_fair_score_U0"):
                        torch.testing.assert_close(getattr(on, name), getattr(off, name), rtol=0, atol=0)
                    self.assertEqual([r["phase"] for r in records], ["pre", "pre", "pre", "post"])
                    self.assertEqual([r["diffusion_t"] for r in records], [3, 2, 0, 0])
                    self.assertEqual([r["progress"] for r in records], [.25, .5, 1., 1.])
                    self.assertEqual([r["event_id"] for r in records], [0, 2, 6, 7])
                    self.assertEqual(records[-2]["requested_progress"], .9)
                    self.assertEqual(records[-1]["score_name"], "q_final")
                    torch.testing.assert_close(records[-1]["q"], on._fair_score_q, rtol=0, atol=0)
                    self.assertFalse(torch.equal(records[-2]["q"], records[-1]["q"]))
                    for record in records:
                        self.assertEqual("w" in record, metric == "eo")
                        for value in record.values():
                            if isinstance(value, torch.Tensor):
                                self.assertEqual(value.device.type, "cpu")
                                self.assertFalse(value.requires_grad)

    def test_full_cache_pre_post_inactive_unvisited_and_immutable(self):
        model = make_model("eo")
        records, frozen = [], []

        def observe(record):
            active = record["active_pair_indices"]
            expected_q = model._fair_score_q.clone()
            expected_w = model._fair_condition_w.clone()
            if record["phase"] == "pre":
                k = model._get_effective_fair_score_k(t_graph=torch.tensor([record["diffusion_t"]]))
                expected_q[active] = torch.sigmoid(model._fair_score_h[active] + k * (model.last_raw_z - model._fair_score_h[active]))
                expected_w[active] = torch.sigmoid(model._fair_condition_h[active] + k * (model.last_raw_z - model._fair_condition_h[active]))
            torch.testing.assert_close(record["q"], expected_q, rtol=0, atol=0)
            torch.testing.assert_close(record["w"], expected_w, rtol=0, atol=0)
            self.assertEqual(record["pair_ids"].shape, (2, 6))
            self.assertEqual(record["q"].numel(), 6)
            self.assertAlmostEqual(float(record["q"][5]), 1. / 3., places=6)
            self.assertEqual(int(record["visit_count"][5]), 0)
            self.assertFalse(record["visited_mask"][5])
            records.append(record)
            frozen.append({key: value.clone() for key, value in record.items() if isinstance(value, torch.Tensor)})

        torch.manual_seed(43)
        sample(model, direct_gap_observer=observe, direct_gap_progress=(.25, .5, .75, 1.))
        for record, original in zip(records, frozen):
            for key, value in original.items():
                torch.testing.assert_close(record[key], value, rtol=0, atol=0)
        torch.testing.assert_close(records[1]["visit_count"], torch.tensor([1, 1, 0, 0, 0, 0]))
        torch.testing.assert_close(records[-1]["visit_count"], torch.tensor([3, 2, 1, 1, 1, 0]))
        self.assertFalse(torch.equal(records[-1]["q"], records[-1]["w"]))

    def test_mutating_every_snapshot_tensor_does_not_change_live_state(self):
        off, on = make_model("eo"), make_model("eo")
        torch.manual_seed(95)
        baseline = sample(off)

        def mutate(record):
            for value in record.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()

        torch.manual_seed(95)
        observed = sample(on, direct_gap_observer=mutate)
        torch.testing.assert_close(observed.log_full_edge_attr_t, baseline.log_full_edge_attr_t, rtol=0, atol=0)
        torch.testing.assert_close(on._fair_score_q, off._fair_score_q, rtol=0, atol=0)
        torch.testing.assert_close(on._fair_condition_w, off._fair_condition_w, rtol=0, atol=0)

    def test_final_empty_active_still_emits_final_and_optional_last_pre(self):
        model, records = make_model("eo", empty_final=True), []
        sample(model, direct_gap_observer=records.append, direct_gap_progress=(1.,))
        self.assertEqual(len(records), 2)
        self.assertEqual([r["score_name"] for r in records], ["q_bar", "q_final"])
        for record in records:
            self.assertFalse(record["guidance_applied"])
            self.assertEqual(record["active_pair_indices"].numel(), 0)
            torch.testing.assert_close(record["q"], model._fair_score_q, rtol=0, atol=0)
        self.assertNotEqual(records[0]["event_id"], records[1]["event_id"])

    def test_final_only_and_batch_ids(self):
        model, records = make_model(), []
        sample(model, count=2, direct_gap_observer=records.append, direct_gap_progress=())
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["phase"], "post")
        torch.testing.assert_close(record["pair_batch"], torch.tensor([0] * 6 + [1] * 6))
        torch.testing.assert_close(record["node_batch"], torch.tensor([0] * 4 + [1] * 4))
        torch.testing.assert_close(record["pair_ids"][:, 6:], record["pair_ids"][:, :6] + 4)
        self.assertTrue(torch.all(record["pair_ids"][0] < record["pair_ids"][1]))

    def test_existing_proxy_observer_can_run_simultaneously(self):
        model, old_records, direct_records = make_model("eo"), [], []
        sample(model, proxy_observer=old_records.append, direct_gap_observer=direct_records.append)
        self.assertEqual(len(old_records), 3)
        for old, direct in zip(old_records, direct_records):
            self.assertNotIn("phase", old)
            for key in ("q", "w", "pair_ids", "same_mask", "pair_batch", "active_pair_indices"):
                torch.testing.assert_close(old[key], direct[key], rtol=0, atol=0)

    def test_rejects_replay_unguided_and_ambiguous_progress(self):
        model = make_model()
        for kwargs in ({"return_controller_replay": True}, {"controller_replay": {}},
                       {"direct_gap_progress": (.25, .25)}, {"direct_gap_progress": (.9, 1.)},
                       {"direct_gap_progress": (float("nan"),)}, {"direct_gap_progress": (0.,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample(model, direct_gap_observer=lambda _: None, **kwargs)
        with self.assertRaises(TypeError):
            sample(model, direct_gap_observer=True)
        model.fair_score_controller_train = False
        with self.assertRaises(ValueError):
            sample(model, direct_gap_observer=lambda _: None)


if __name__ == "__main__":
    unittest.main()

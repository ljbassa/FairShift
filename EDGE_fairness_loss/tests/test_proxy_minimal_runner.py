"""CPU-only orchestration tests; every CUDA operation and model fit is mocked."""

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch_geometric.data import Batch, Data

import proxy_minimal_metrics as metrics
import run_proxy_minimal as runner
from proxy_minimal_gcn import EmbeddingDecoder


REPO = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = REPO / "results/proxy_minimal"


def _spec():
    return {
        "dataset": "cora", "device": "cuda:4", "decode_chunk_size": 5,
        "root_seeds": list(range(31000, 31030)),
        "backbone_args": "synthetic_backbone_args",
        "backbone_checkpoint": "synthetic_backbone_checkpoint",
        "graph": "synthetic_reference_graph",
        "controllers": {
            target: {"args": f"synthetic_{target}_args", "checkpoint": f"synthetic_{target}_checkpoint"}
            for target in ("sp", "eo")
        },
        # Synthetic configuration exercises passing a single object, not model selection.
        "gcn": {"config": {"fixture_only": True}},
    }


def _manifest():
    return {
        "status": "ready_awaiting_gpu_approval", "blockers": [],
        "actual_counts": {"graphs": 0, "gcn_fits": 0, "gcn_fit_attempts": 0,
                          "observations": 0, "gpu_smoke": 0, "retries": 0},
        "actual_stages": ["cpu_preflight"], "environment": {}, "cost": {},
    }


def _graph():
    pairs = torch.triu_indices(8, 8, offset=1)
    return Data(
        num_nodes=8, x=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), orig_id=torch.arange(8),
        edge_index=pairs[:, :20].clone(),
    ), pairs


class ProxyMinimalRunnerTest(unittest.TestCase):
    def test_exact_budget_observer_routing_and_shared_terminal_embedding(self):
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="runner_cpu_", dir=TEST_TEMP_ROOT) as tmp:
            output = Path(tmp)
            spec, manifest = _spec(), _manifest()
            graph, pairs = _graph()
            sampler = object()
            samples, constructions, fits, decoders = [], [], [], []
            args_by_path = {
                spec["backbone_args"]: SimpleNamespace(target="uncontrolled", fair_score_eo_min_mass=1e-6),
                **{entry["args"]: SimpleNamespace(target=target, fair_score_eo_min_mass=1e-6)
                   for target, entry in spec["controllers"].items()},
            }
            checkpoints = {
                spec["backbone_checkpoint"]: {"model": {"frozen_tensor": torch.tensor([1.0])}},
                **{entry["checkpoint"]: {"synthetic_target": target}
                   for target, entry in spec["controllers"].items()},
            }

            def build(args, actual_sampler, backbone, controller=None, *, device):
                self.assertIs(actual_sampler, sampler)
                self.assertEqual(device, "cuda:4")
                constructions.append((args.target, backbone, controller))
                target = args.target

                def sample(count, *, proxy_observer, proxy_observer_progress):
                    self.assertEqual(count, 1)
                    self.assertEqual(tuple(proxy_observer_progress), runner.PROGRESS)
                    self.assertEqual(proxy_observer is None, target == "uncontrolled")
                    samples.append(target)
                    if proxy_observer is not None:
                        same = graph.y[pairs[0]] == graph.y[pairs[1]]
                        for progress in proxy_observer_progress:
                            step = math.ceil(progress * 20)
                            q = torch.linspace(.1, .9, pairs.shape[1]) + progress * .01
                            w = torch.linspace(.2, .7, pairs.shape[1])
                            weight = w if target == "eo" else torch.ones_like(q)
                            gap = (q[same] * weight[same]).sum() / weight[same].sum()
                            gap -= (q[~same] * weight[~same]).sum() / weight[~same].sum()
                            snapshot = {
                                "target": target, "metric": target, "q": q,
                                "pair_ids": pairs.clone(), "same_mask": same.clone(),
                                "pair_batch": torch.zeros(pairs.shape[1], dtype=torch.long),
                                "requested_progress": progress, "progress": step / 20,
                                "loop_index": step - 1, "step_number": step,
                                "diffusion_t": 20 - step, "total_steps": 20,
                                "production_proxy": gap.reshape(1),
                                "valid_graph": torch.tensor([True]),
                            }
                            if target == "eo":
                                snapshot["w"] = w
                            proxy_observer(snapshot)
                    return Batch.from_data_list([graph.clone()])

                return SimpleNamespace(sample=sample)

            def fit(generated, config, *, split_seed, fit_seed, device, chunk_size):
                self.assertIs(config, spec["gcn"]["config"])
                self.assertEqual(device, "cuda:4")
                self.assertEqual(chunk_size, 5)
                torch.testing.assert_close(generated.edge_index, graph.edge_index)
                fits.append((split_seed, fit_seed))
                embedding = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 20 + len(fits) * .001
                decoder = EmbeddingDecoder(embedding, chunk_size)
                decoders.append(decoder)
                test_pairs = pairs[:, [0, 3, 20, 22]]
                return {
                    "decoder": decoder, "embedding": embedding,
                    "test_pairs": test_pairs, "test_labels": torch.tensor([1, 1, 0, 0]),
                    "test_scores": decoder(test_pairs),
                    "train_pairs": pairs[:, 4:20], "val_pairs": pairs[:, [1, 2, 21, 23]],
                    "meta": {"gcn_fits": 1, "gcn_best_epoch": 1},
                }

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner, "OUTPUT", output))
                stack.enter_context(mock.patch.object(runner, "load_args", side_effect=args_by_path.__getitem__))
                stack.enter_context(mock.patch.object(runner, "load_checkpoint", side_effect=checkpoints.__getitem__))
                stack.enter_context(mock.patch.object(runner, "read_reference", return_value=(graph, sampler, {})))
                stack.enter_context(mock.patch.object(runner, "build_model", side_effect=build))
                fit_mock = stack.enter_context(mock.patch.object(runner, "fit_terminal_gcn", side_effect=fit))
                seed_mock = stack.enter_context(mock.patch.object(runner, "seed_all"))
                stack.enter_context(mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}))
                stack.enter_context(mock.patch("torch.cuda.is_available", return_value=True))
                stack.enter_context(mock.patch("torch.cuda.device_count", return_value=5))
                selected_device = stack.enter_context(mock.patch("torch.cuda.set_device"))
                stack.enter_context(mock.patch("torch.cuda.reset_peak_memory_stats"))
                stack.enter_context(mock.patch("torch.cuda.max_memory_allocated", return_value=0))
                stack.enter_context(mock.patch("torch.cuda.max_memory_reserved", return_value=0))
                gpu_name = stack.enter_context(mock.patch("torch.cuda.get_device_name", return_value="mock CPU fixture"))
                synchronize = stack.enter_context(mock.patch("torch.cuda.synchronize"))
                analyze = stack.enter_context(mock.patch.object(metrics, "analyze_snapshot", wraps=metrics.analyze_snapshot))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runner.execute(spec, manifest)

            self.assertEqual(Counter(samples), {"uncontrolled": 10, "sp": 10, "eo": 10})
            self.assertEqual(fit_mock.call_count, 30)
            self.assertEqual(analyze.call_count, 60)
            self.assertEqual(seed_mock.call_args_list, [mock.call(seed, "cuda:4") for seed in spec["root_seeds"]])
            self.assertEqual(fits, [(seed + 1000000, seed + 2000000) for seed in spec["root_seeds"]])
            self.assertIsNone(constructions[0][2])
            self.assertEqual([item[0] for item in constructions], ["uncontrolled", "sp", "eo"])
            self.assertTrue(all(item[1] is constructions[0][1] for item in constructions))
            self.assertEqual(constructions[1][2]["synthetic_target"], "sp")
            self.assertEqual(constructions[2][2]["synthetic_target"], "eo")
            gpu_name.assert_called_once_with("cuda:4")
            selected_device.assert_called_once_with(4)
            self.assertEqual(synchronize.call_args_list, [mock.call("cuda:4")] * 30)
            for index, call in enumerate(analyze.call_args_list):
                self.assertIs(call.kwargs["decode_pairs"], decoders[10 + index // 3])
                snapshot = call.args[0]
                self.assertEqual(snapshot["requested_progress"], runner.PROGRESS[index % 3])
                self.assertEqual(snapshot["loop_index"], (4, 9, 17)[index % 3])
                self.assertEqual(snapshot["diffusion_t"], (15, 10, 2)[index % 3])
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["actual_counts"], {
                "graphs": 30, "gcn_fits": 30, "gcn_fit_attempts": 30,
                "observations": 60, "gpu_smoke": 0, "retries": 0,
            })
            with (output / "terminal_metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 30)
            with (output / "proxy_alignment.csv").open() as stream:
                alignment = list(csv.DictReader(stream))
            self.assertEqual(len(alignment), 60)
            self.assertEqual(Counter((row["target"], float(row["requested_progress"])) for row in alignment),
                             {(target, progress): 10 for target in ("sp", "eo") for progress in runner.PROGRESS})
            aggregates = json.loads((output / "aggregate.json").read_text())
            self.assertEqual(len(aggregates["alignment"]), 6)
            self.assertTrue(all(row["graph_count"] == 10 for row in aggregates["alignment"]))
            self.assertEqual(len(list((output / "artifacts").glob("*.pt"))), 30)
            sample_artifact = torch.load(output / "artifacts/sp_31010.pt", map_location="cpu", weights_only=False)
            self.assertNotIn("x", sample_artifact)
            self.assertNotIn("backbone", sample_artifact)
            self.assertEqual(len(sample_artifact["observations"]), 3)
            self.assertIn("pair_ids", sample_artifact["observer_support"])

    def test_missing_selection_dry_run_blocks_without_execute_or_cuda(self):
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="runner_preflight_", dir=TEST_TEMP_ROOT) as tmp:
            output = Path(tmp)
            spec = _spec()
            spec["gcn"] = None
            spec_path = output / "synthetic_spec.json"
            spec_path.write_text(json.dumps(spec))
            graph, _pairs = _graph()
            args = SimpleNamespace(dataset="cora", diffusion_steps=20)
            baseline = SimpleNamespace(num_timesteps=20, parameters=lambda: [])
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner, "OUTPUT", output))
                stack.enter_context(mock.patch.object(runner, "file_record", side_effect=lambda path: {
                    "path": str(path), "sha256": "synthetic", "bytes": 0}))
                stack.enter_context(mock.patch.object(runner, "load_args", return_value=args))
                stack.enter_context(mock.patch.object(runner, "load_checkpoint", return_value={"model": {}}))
                stack.enter_context(mock.patch.object(runner, "read_reference", return_value=(
                    graph, object(), {"full_pair_count": 28})))
                stack.enter_context(mock.patch.object(runner, "build_model", return_value=baseline))
                execute = stack.enter_context(mock.patch.object(runner, "execute", side_effect=AssertionError("GPU execute forbidden")))
                fit = stack.enter_context(mock.patch.object(runner, "fit_terminal_gcn", side_effect=AssertionError("GCN fit forbidden")))
                for name in ("is_available", "device_count", "get_device_name", "synchronize", "manual_seed", "manual_seed_all", "set_device"):
                    stack.enter_context(mock.patch(f"torch.cuda.{name}", side_effect=AssertionError("CUDA forbidden")))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                exit_code = runner.main(["--spec", str(spec_path), "--dry-run"])
            self.assertEqual(exit_code, 2)
            execute.assert_not_called()
            fit.assert_not_called()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "blocked")
            self.assertEqual(manifest["actual_counts"]["graphs"], 0)
            self.assertEqual(manifest["actual_counts"]["gcn_fits"], 0)
            self.assertEqual(manifest["actual_stages"], ["cpu_preflight"])
            self.assertTrue(any("T-SP:" in item and "selection" in item for item in manifest["blockers"]))
            self.assertTrue(any("T-EO:" in item and "selection" in item for item in manifest["blockers"]))
            self.assertTrue(any("GCN:" in item for item in manifest["blockers"]))
            self.assertFalse((output / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()

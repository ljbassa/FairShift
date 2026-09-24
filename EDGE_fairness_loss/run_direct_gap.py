#!/usr/bin/env python3
"""Manifest-based read-only proxy/downstream gap diagnostics. Default: preflight.

All outputs are exclusive-create. `observe --smoke` runs exactly one graph and one
evaluator seed; `batch` is an explicit separate command. No training in observe.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

from direct_gap_runtime import (build_observed_model, cpu_copy, file_hash, fit_existing_evaluator,
                                graph_hash, load, object_hash, tensor_hash)

ROOT = Path(__file__).resolve().parent


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def code_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    sources = ["run_direct_gap.py", "direct_gap_runtime.py", "direct_gap_analysis.py", "import_legacy_direct_gap.py",
               "direct_gap_ablation.py", "diffusion/diffusion_binomial_active.py",
               "diffusion/fairness_surrogate.py", "evaluate_generated_graphs.py", "model.py"]
    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": {p: file_hash(ROOT / p) for p in sources if (ROOT / p).exists()}}


def fresh_output(path):
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def resolve_manifest(path):
    path = Path(path).expanduser().resolve()
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1")
    spec["manifest_source"] = str(path)
    base = Path(spec.get("asset_root", str(path.parent)))
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    ids = set()
    for config in spec["configurations"]:
        if not config.get("id") or config["id"] in ids:
            raise ValueError("Configuration IDs must be nonempty and unique")
        ids.add(config["id"])
        if Path(config["id"]).name != config["id"] or config["id"] in (".", ".."):
            raise ValueError("Configuration IDs must be safe single path components")
        for key in ("backbone_checkpoint", "backbone_args", "graph"):
            if config.get(key):
                config[key] = str((base / config[key]).resolve())
        for key in ("checkpoint",):
            if config.get("controller", {}).get(key):
                config["controller"][key] = str((base / config["controller"][key]).resolve())
        seen = set()
        for run in config["runs"]:
            if run["graph_seed"] in seen or len(set(run["evaluator_seeds"])) != len(run["evaluator_seeds"]):
                raise ValueError("Duplicate graph/evaluator seeds within a configuration")
            if not run["evaluator_seeds"]:
                raise ValueError("At least one evaluator seed is required")
            seen.add(run["graph_seed"])
    threshold = spec.get("sign_threshold", 1e-6)
    if not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("sign_threshold must be prespecified, finite and nonnegative")
    if spec.get("evaluator_protocol", "existing_samplepy_grid_v1") != "existing_samplepy_grid_v1":
        raise ValueError("Online execution retains existing_samplepy_grid_v1 evaluator")
    return spec


def preflight_config(config):
    result = {"id": config["id"], "assets": {}, "blockers": []}
    for key in ("backbone_checkpoint", "backbone_args", "graph"):
        path = config.get(key)
        if not path or not Path(path).is_file():
            result["blockers"].append(f"Missing {key}: {path}")
        else:
            result["assets"][key] = {"path": path, "sha256": file_hash(path)}
    controller = config.get("controller", {})
    if controller.get("kind") in ("T", "tied"):
        path = controller.get("checkpoint")
        if not path or not Path(path).is_file():
            result["blockers"].append("Compatible fixed-k eta-only controller checkpoint mapping is unresolved")
        else:
            result["assets"]["controller"] = {"path": path, "sha256": file_hash(path)}
    elif controller.get("kind") != "F":
        result["blockers"].append("Controller kind must explicitly be F, T or tied")
    if not result["blockers"]:
        try:
            model, data, actual = build_observed_model(config, device="cpu")
            result["actual"] = actual
            result["metadata"] = {"nodes": data.num_nodes, "feature_hash": tensor_hash(data.x)}
            del model, data
        except Exception as exc:
            result["blockers"].append(f"{type(exc).__name__}: {exc}")
    result["status"] = "blocked" if result["blockers"] else "ready"
    return result


def bind_provenance(data, snapshots, config, run, actual, *, protocol):
    """Bind only after generation; future graph/evaluation data never enters observer."""
    digest = graph_hash(data)
    order = getattr(data, "orig_id", None)
    if order is None:
        order = torch.arange(data.num_nodes)
    identity = {"dataset": config["dataset"], "configuration_id": config.get("configuration_id", config["id"]),
                "graph_id": f"{config['id']}:graph_seed={run['graph_seed']}",
                "graph_hash": digest, "backbone_hash": actual["backbone_hash"],
                "controller_hash": actual["controller_hash"], "node_order_hash": tensor_hash(order),
                "pair_mapping": "unordered_i_lt_j", "batch_id": 0,
                "controller_seed": actual["controller"].get("calibration_seed"),
                "graph_seed": run["graph_seed"], "split_seed": run["split_seed"],
                "protocol": protocol, "generation_mode": actual["generation_mode"],
                "T": actual["T"], "k": actual.get("k"), "variant": actual["variant"],
                "normalization": actual["normalization"],
                "eo_min_mass": actual.get("eo_min_mass", 1e-6)}
    for index, snapshot in enumerate(snapshots):
        snapshot["snapshot_id"] = f"{identity['graph_id']}:{snapshot['phase']}:{index}"
        snapshot["provenance"] = {**identity, "snapshot_phase": snapshot["phase"]}
    return identity


def analyze_bound(snapshots, evaluation, identity, evaluator_seed, *, sign_threshold):
    from direct_gap_analysis import analyze_snapshot
    evaluation["provenance"] = {**identity, "evaluator_seed": evaluator_seed}
    records = []
    for snapshot in snapshots:
        provenance = {**identity, "evaluator_seed": evaluator_seed, "snapshot_phase": snapshot["phase"],
                      "feature_source": evaluation["feature_source"], "group_source": evaluation["group_source"]}
        rows = analyze_snapshot(snapshot, evaluation, provenance,
                                min_mass=identity.get("eo_min_mass", 1e-6),
                                sign_threshold=sign_threshold)
        for row in rows:
            row["evaluator_fit_completed"] = True
        records.extend(rows)
    return records


def execute_run(config, run, actual_preflight, output, spec, *, smoke=False):
    from proxy_minimal_gcn import seed_all
    run_dir = output / config["id"] / f"graph_{run['graph_seed']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    device = spec.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable; no silent fallback")
    model, reference, actual = build_observed_model(config, device=device)
    comparable = cpu_copy(actual)
    comparable["args"]["device"] = actual_preflight["args"]["device"]
    if comparable != actual_preflight:
        raise ValueError("Model assets/configuration changed after preflight")
    snapshots = []
    seed_all(run["graph_seed"], device)
    with (run_dir / "generation.log").open("w") as log, redirect_stdout(log), torch.no_grad():
        generated = model.sample(1, direct_gap_observer=snapshots.append,
                                 direct_gap_progress=tuple(spec.get("progress", [.25, .5, .9, 1.0]))).cpu().to_data_list()[0]
    for key in ("x", "y", "sens", "orig_id"):
        before = getattr(reference, key, None)
        after = getattr(generated, key, None)
        if before is not None and (after is None or not torch.equal(before.cpu(), after.cpu())):
            raise ValueError(f"Generated graph changed metadata/node ordering: {key}")
    if not any(s["phase"] == "post" and s["score_name"] == "q_final" for s in snapshots):
        raise ValueError("Required final committed q_final snapshot is missing")
    identity = bind_provenance(generated, snapshots, config, run, actual, protocol="existing_samplepy_grid_v1")
    trajectory = {"data": generated, "snapshots": snapshots, "provenance": identity, "actual": actual}
    torch.save(trajectory, run_dir / "trajectory.pt")
    write_json(run_dir / "trajectory_identity.json", {**identity, "artifact_sha256": file_hash(run_dir / "trajectory.pt")})
    del model, reference
    records, evaluator_files, evaluator_failures = [], [], []
    seeds = run["evaluator_seeds"][:1] if smoke else run["evaluator_seeds"]
    for evaluator_seed in seeds:
        try:
            evaluation = fit_existing_evaluator(generated, split_seed=run["split_seed"],
                                                 evaluator_seed=evaluator_seed, device=device,
                                                 group_attr=config.get("group_attr", "y"))
            records.extend(analyze_bound(snapshots, evaluation, identity, evaluator_seed,
                                         sign_threshold=spec.get("sign_threshold", 1e-6)))
            path = run_dir / f"evaluation_{evaluator_seed}.pt"
            torch.save(evaluation, path)
            evaluator_files.append({"path": str(path), "sha256": file_hash(path)})
        except Exception as exc:
            failure = {"evaluator_seed": evaluator_seed, "reason": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            evaluator_failures.append(failure)
            write_json(run_dir / f"evaluation_{evaluator_seed}_failed.json", failure)
            # Keep each prespecified evaluator repeat in the hierarchy. Otherwise
            # a graph mean could silently select only its successful GNN seeds.
            for snapshot in snapshots:
                for metric in ("sp", "eo"):
                    records.append({**identity, "evaluator_seed": evaluator_seed, "metric": metric,
                                    "phase": snapshot["phase"], "score_name": snapshot["score_name"],
                                    "snapshot_id": snapshot["snapshot_id"], "progress": snapshot["progress"],
                                    "chunk_index": snapshot["chunk_index"],
                                    "diagnostic_status": "failed", "unavailable_reason": failure["reason"],
                                    "evaluator_fit_completed": False,
                                    "sign_threshold": spec.get("sign_threshold", 1e-6),
                                    **{name: float("nan") for name in ("a", "b", "c", "d", "downstream_gap")}})
    return records, {"status": "complete" if not evaluator_failures else "incomplete",
                     "reason": "" if not evaluator_failures else "One or more planned evaluator seeds failed; no retry/replacement",
                     "provenance": identity, "actual": actual, "evaluator_failures": evaluator_failures,
                     "trajectory": {"path": str(run_dir / "trajectory.pt"),
                                    "sha256": file_hash(run_dir / "trajectory.pt")},
                     "evaluations": evaluator_files, "snapshots": len(snapshots)}


def write_results(output, records, manifest):
    from direct_gap_analysis import write_analysis
    write_analysis(records, output, sign_threshold=manifest.get("sign_threshold", 1e-6))
    write_json(output / "resolved_manifest.json", manifest)
    completed = sum(r.get("status") == "complete" for r in manifest.get("runs", []))
    failures = [r for r in manifest.get("runs", []) if r.get("status") != "complete"]
    text = ["# Direct proxy/downstream gap diagnostic", "",
            f"Mode: {manifest['mode']}. Status: {manifest.get('status')}. Completed graph runs: {completed}.", "",
            "Scores use probability units; *_pp columns separately multiply gaps by 100.",
            "SP uses all actual generated test pairs; EO uses generated held-out positives.",
            "Correlations keep dataset, protocol, phase and progress separate. Multiple evaluator seeds are nested within graphs.",
            "One graph verifies the pipeline; it cannot establish correlation. Undefined statistics retain reasons.",
            "Positive correlation is empirical alignment under observed conditions, not a causal downstream-descent guarantee for the sampler.", "",
            "## Executed and unexecuted", "",
            f"Planned configurations: {len(manifest.get('configurations', []))}; failed/blocked run entries: {len(failures)}.",
            "No checkpoint selection by test score or proxy correlation; no automatic sweep/retry; source assets are read-only.", ""]
    for note in manifest.get("notes", []):
        text.append(f"- {note}")
    for item in manifest.get("preflight", []):
        for blocker in item.get("blockers", []):
            text.append(f"- NOT RUN {item['id']}: {blocker}")
    for item in failures:
        text.append(f"- {item.get('configuration_id')}: {item.get('status')}: {item.get('reason')}")
    (output / "report.md").write_text("\n".join(text) + "\n")


def run_manifest(args):
    spec = resolve_manifest(args.manifest)
    output = fresh_output(args.output)
    manifest = {**spec, "mode": args.command, "code": code_identity(), "status": "preflight",
                "started_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "preflight": []}
    records = []
    for config in spec["configurations"]:
        audit = preflight_config(config)
        manifest["preflight"].append(audit)
    if args.command == "preflight":
        manifest["status"] = "blocked" if any(x["blockers"] for x in manifest["preflight"]) else "ready"
        write_results(output, records, manifest)
        return 2 if manifest["status"] == "blocked" else 0
    selected = spec["configurations"][:1] if args.command == "observe" else spec["configurations"]
    manifest["notes"] = ["observe is a one-graph/one-evaluator smoke; batch requires a separate explicit command."]
    if args.command == "observe":
        manifest["notes"].append("Other manifest configurations/seeds are planned but unexecuted by the smoke command.")
    for config in selected:
        audit = next(x for x in manifest["preflight"] if x["id"] == config["id"])
        if audit["blockers"]:
            manifest["runs"].append({"configuration_id": config["id"], "status": "blocked",
                                     "reason": "; ".join(audit["blockers"])})
            continue
        runs = config["runs"][:1] if args.command == "observe" else config["runs"]
        for run in runs:
            try:
                rows, detail = execute_run(config, run, audit["actual"], output, spec,
                                            smoke=args.command == "observe")
                records.extend(rows)
                manifest["runs"].append({"configuration_id": config["id"], **detail})
            except Exception as exc:
                manifest["runs"].append({"configuration_id": config["id"], "graph_seed": run["graph_seed"],
                                         "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()})
                write_json(output / "resolved_manifest.json", manifest)
    manifest["status"] = "complete" if manifest["runs"] and all(x["status"] == "complete" for x in manifest["runs"]) else "incomplete"
    # Verify original on-disk checkpoints/data were not modified.
    for audit in manifest["preflight"]:
        for asset in audit["assets"].values():
            if file_hash(asset["path"]) != asset["sha256"]:
                raise RuntimeError(f"Source asset changed: {asset['path']}")
    manifest["source_assets_unchanged"] = True
    write_results(output, records, manifest)
    return 0 if manifest["status"] == "complete" else 2


def analyze_saved(args):
    spec_path = Path(args.manifest).resolve()
    spec = json.loads(spec_path.read_text())
    output = fresh_output(args.output)
    records = []
    manifest = {**spec, "mode": "analyze", "code": code_identity(), "runs": [], "status": "complete"}
    for entry in spec["artifacts"]:
        paths = {}
        for key in ("trajectory", "evaluation"):
            record = entry[key]
            path = (spec_path.parent / record["path"]).resolve()
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"{key} artifact hash mismatch")
            paths[key] = path
        trajectory, evaluation = load(paths["trajectory"]), load(paths["evaluation"])
        identity = trajectory["provenance"]
        # Never overwrite stored evaluation identity to make an incompatible join pass.
        for key, value in identity.items():
            if evaluation["provenance"].get(key) != value:
                raise ValueError(f"Trajectory/evaluation provenance mismatch: {key}")
        if graph_hash(trajectory["data"]) != identity["graph_hash"]:
            raise ValueError("Trajectory generated graph hash mismatch")
        data = trajectory["data"]
        order = getattr(data, "orig_id", None)
        order = torch.arange(data.num_nodes) if order is None else order
        if tensor_hash(order) != identity["node_order_hash"]:
            raise ValueError("Trajectory node-order content hash mismatch")
        from evaluate_generated_graphs import unique_undirected_edge_index
        if not torch.equal(unique_undirected_edge_index(data.edge_index),
                           unique_undirected_edge_index(torch.as_tensor(evaluation["generated_positive_pairs"]))):
            raise ValueError("Evaluator was fitted/evaluated on a different generated graph")
        if tensor_hash(data.x) != evaluation["feature_hash"]:
            raise ValueError("Evaluator feature content hash mismatch")
        if tensor_hash(evaluation["node_groups"]) != evaluation["group_hash"]:
            raise ValueError("Evaluator group content hash mismatch")
        prefix = "generated_graph."
        if not evaluation["group_source"].startswith(prefix):
            raise ValueError("Unsupported evaluator group source")
        attr = evaluation["group_source"][len(prefix):]
        if getattr(data, attr, None) is None or not torch.equal(
                torch.as_tensor(evaluation["node_groups"]), getattr(data, attr).reshape(-1).long()):
            raise ValueError("Evaluator group source differs from generated metadata")
        rows = analyze_bound(trajectory["snapshots"], evaluation, identity,
                             evaluation["provenance"]["evaluator_seed"],
                             sign_threshold=spec.get("sign_threshold", 1e-6))
        records.extend(rows)
        manifest["runs"].append({"status": "complete", "provenance": identity, "artifacts": entry})
    write_results(output, records, manifest)
    return 0


def toy_smoke(args):
    """Synthetic full pipeline, not a Cora result or controller calibration."""
    from types import SimpleNamespace
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from diffusion.diffusion_binomial_active import BinomialDiffusionActive
    from direct_gap_runtime import attach_fixed_schedule
    from proxy_minimal_gcn import seed_all

    output = fresh_output(args.output)
    model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
    torch.nn.Module.__init__(model)
    model.device, model.num_timesteps = "cpu", 20
    model.num_node_classes = model.num_edge_classes = 2
    model.predict_s, model.sampling_stage = False, "stage1_base"
    model.fair_score_controller_train, model.fair_score_k, model.fair_score_eta = True, .5, .02
    model.fair_score_metric, model.fair_score_eo_min_mass = "sp", 1e-6
    model.fair_score_guidance_normalize, model.fair_label_attr = True, "y"
    model._init_controller_guidance_params()
    attach_fixed_schedule(model, .5, [.02] * 20)
    pairs = torch.triu_indices(16, 16, 1)

    def initial(count):
        if count != 1:
            raise ValueError("Toy smoke has one graph")
        data = Data(num_nodes=16, full_edge_index=pairs.clone(), edge_index=torch.empty((2, 0), dtype=torch.long),
                    nodes_per_graph=torch.tensor([16]), edges_per_graph=torch.tensor([pairs.shape[1]]),
                    degree=torch.ones(16), y=torch.arange(16) % 2, orig_id=torch.arange(16),
                    x=torch.stack((torch.arange(16).float() / 16, (torch.arange(16) % 3).float()), -1),
                    log_node_attr_t=F.one_hot(torch.zeros(16, dtype=torch.long), 2).float().clamp_min(1e-30).log(),
                    log_full_edge_attr_t=F.one_hot(torch.zeros(pairs.shape[1], dtype=torch.long), 2).float().clamp_min(1e-30).log())
        return Batch.from_data_list([data])

    def actives(data, t):
        # Some pairs stay unvisited, including at final commit.
        data.active_edge_indices = torch.arange(pairs.shape[1] - 7)
        data.active_node_indices = torch.unique(pairs[:, data.active_edge_indices])

    def predict(data, t, unused):
        index = data.active_edge_indices
        same = data.y[pairs[0, index]] == data.y[pairs[1, index]]
        z = same.float() * .5 - .7 + (index % 5).float() * .08
        return torch.zeros((16, 2)).log_softmax(-1), torch.stack((F.logsigmoid(-z), F.logsigmoid(z)), -1)

    model.initial_graph_sampler = SimpleNamespace(sample=initial)
    model._prepare_data_for_sampling = lambda data: data
    model._p_sample_and_set_actives, model._p_pred = actives, predict
    snapshots = []
    seed_all(731, "cpu")
    with torch.no_grad(), redirect_stdout(io.StringIO()):
        off = model.sample(1).cpu().to_data_list()[0]
    off_rng = torch.get_rng_state().clone()
    seed_all(731, "cpu")
    with torch.no_grad(), redirect_stdout(io.StringIO()):
        data = model.sample(1, direct_gap_observer=snapshots.append,
                            direct_gap_progress=(.25, .5, .9, 1.)).cpu().to_data_list()[0]
    if not torch.equal(off.edge_index, data.edge_index) or not torch.equal(off_rng, torch.get_rng_state()):
        raise AssertionError("Observer changed toy sampling or RNG state")
    actual = {"T": 20, "k": .5, "eta_schedule": [.02] * 20, "variant": "F",
              "normalization": True, "controller": {"calibration_seed": None},
              "backbone_hash": "synthetic_deterministic_denoiser_v1", "controller_hash": object_hash({"k": .5, "eta": .02}),
              "generation_mode": "synthetic_online_guided_no_replay"}
    config, run = {"id": "toy_fixed_k_sp", "dataset": "synthetic_toy"}, {"graph_seed": 731, "split_seed": 1731}
    identity = bind_provenance(data, snapshots, config, run, actual, protocol="existing_samplepy_grid_v1")
    evaluation = fit_existing_evaluator(data, split_seed=1731, evaluator_seed=2731, device="cpu")
    records = analyze_bound(snapshots, evaluation, identity, 2731, sign_threshold=1e-6)
    torch.save({"data": data, "snapshots": snapshots, "provenance": identity, "actual": actual}, output / "trajectory.pt")
    torch.save(evaluation, output / "evaluation.pt")
    manifest = {"schema_version": 1, "mode": "toy-smoke", "status": "complete", "code": code_identity(),
                "sign_threshold": 1e-6, "configurations": [config], "observer_on_off_graph_and_rng_equal": True,
                "runs": [{"status": "complete", "provenance": identity, "actual": actual,
                          "snapshots": len(snapshots), "evaluator_trials": len(evaluation["trials"])}],
                "notes": ["Executed synthetic CPU sampler → existing evaluator tuning → pair join → signed gap CSV/plots.",
                          "Cora fixed-k T generation NOT executed: compatible controller mapping unresolved.",
                          "Synthetic data are not evidence about Cora fairness or correlation."]}
    write_json(output / "offline_manifest.json", {"schema_version": 1, "sign_threshold": 1e-6,
                "artifacts": [{"trajectory": {"path": "trajectory.pt", "sha256": file_hash(output / "trajectory.pt")},
                               "evaluation": {"path": "evaluation.pt", "sha256": file_hash(output / "evaluation.pt")}}]})
    write_results(output, records, manifest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("command", choices=("preflight", "observe", "batch", "analyze", "toy-smoke", "prepare-k-ablation", "calibrate-k-ablation"), nargs="?", default="preflight")
    parser.add_argument("--manifest", default=str(ROOT / "configs/direct_gap_cora_sp.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-calibration", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:args.cpu_threads])
    if args.command == "toy-smoke":
        return toy_smoke(args)
    if args.command == "analyze":
        return analyze_saved(args)
    if args.command in ("prepare-k-ablation", "calibrate-k-ablation"):
        from direct_gap_ablation import run_ablation
        return run_ablation(args)
    return run_manifest(args)


if __name__ == "__main__":
    sys.exit(main())

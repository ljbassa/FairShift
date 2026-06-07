import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

from data import load_dataset, load_datasets_nc, preprocess
from Model import ModelSync
from sample import build_pyg_data_from_sample
from setup_utils import set_seed


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tag_float(value):
    return str(value).replace("-", "m").replace(".", "p")


def prepare_args(args):
    if args.controller_pretrained_ckpt is None:
        args.controller_pretrained_ckpt = args.model_path
    if args.controller_pretrained_ckpt is None:
        raise ValueError("--controller_pretrained_ckpt is required (or use legacy --model_path).")
    args.model_path = args.controller_pretrained_ckpt

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if args.name is None:
        args.name = (
            f"controller_eta{tag_float(args.fair_score_eta)}"
            f"_k{tag_float(args.fair_score_k)}"
            f"_lr{tag_float(args.controller_lr)}"
            f"_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    if args.eval_every is None:
        args.eval_every = args.controller_epochs
    if args.check_every is None:
        args.check_every = args.controller_epochs

    args.controller_epochs = max(1, int(args.controller_epochs))
    args.controller_replay_refresh = max(1, int(args.controller_replay_refresh))
    args.controller_replay_num_samples = max(1, int(args.controller_replay_num_samples))
    args.controller_replay_fixed = bool(args.controller_replay_fixed)
    args.num_generation = max(1, int(args.num_generation))
    args.eval_every = max(1, int(args.eval_every))
    args.check_every = max(1, int(args.check_every))
    args.sample_batch_size = max(1, int(args.sample_batch_size))
    if not args.fair_score_learn_k and args.fair_score_k_tracking_loss_weight != 0.0:
        print(
            "[controller] fair_score_learn_k=False; "
            "consider --fair_score_k_tracking_loss_weight 0.0 so checkpoint selection ignores the fixed-k tracking term."
        )
    return args


def prepare_data(dataset, device):
    if dataset in ["cora", "citeseer", "amazon_photo", "amazon_computer"]:
        graph = load_dataset(dataset)
    else:
        graph = load_datasets_nc(dataset)

    X_one_hot_3d, s, y, E_one_hot, \
        X_marginal, s_marginal, y_marginal, E_marginal, \
        X_cond_s_marginals, X_cond_y_marginals, y_cond_s_marginals, p_values = preprocess(graph)

    if y_marginal is not None:
        y_marginal = y_marginal.to(device)
    if y_cond_s_marginals is not None:
        y_cond_s_marginals = y_cond_s_marginals.to(device)

    return {
        "num_nodes": s.size(0),
        "X_one_hot_3d": X_one_hot_3d.to(device),
        "s": s.to(device),
        "y": y.to(device) if y is not None else None,
        "X_marginal": X_marginal.to(device),
        "s_marginal": s_marginal.to(device),
        "y_marginal": y_marginal,
        "E_marginal": E_marginal.to(device),
        "y_cond_s_marginal": y_cond_s_marginals,
        "p_values": p_values,
    }


def build_model(args, checkpoint, device):
    dataset = checkpoint["dataset"]
    train_yaml_data = checkpoint["train_yaml_data"]
    data_kwargs = prepare_data(dataset, device)

    model = ModelSync(
        X_marginal=data_kwargs["X_marginal"],
        s_marginal=data_kwargs["s_marginal"],
        y_marginal=data_kwargs["y_marginal"],
        E_marginal=data_kwargs["E_marginal"],
        num_nodes=data_kwargs["num_nodes"],
        p_values=data_kwargs["p_values"],
        y_cond_s_marginal=data_kwargs["y_cond_s_marginal"],
        gnn_X_config=train_yaml_data["gnn_X"],
        gnn_E_config=train_yaml_data["gnn_E"],
        fair_label_attr=args.fair_label_attr,
        fair_score_eta=args.fair_score_eta,
        fair_score_k=args.fair_score_k,
        fair_score_eta_scale=args.fair_score_eta_scale,
        fair_score_controller_train=True,
        fair_score_learn_k=args.fair_score_learn_k,
        fair_score_learn_eta=args.fair_score_learn_eta,
        fair_score_guidance_normalize=args.fair_score_guidance_normalize,
        fair_score_fair_loss_weight=args.fair_score_fair_loss_weight,
        fair_score_k_tracking_loss_weight=args.fair_score_k_tracking_loss_weight,
        fair_score_utility_loss_weight=args.fair_score_utility_loss_weight,
        **train_yaml_data["diffusion"],
    ).to(device)

    if "pred_X_state_dict" in checkpoint and "pred_E_state_dict" in checkpoint:
        model.graph_encoder.pred_X.load_state_dict(checkpoint["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(checkpoint["pred_E_state_dict"])
    elif "model" in checkpoint:
        state = checkpoint["model"]
        state = {
            k: v for k, v in state.items()
            if not k.endswith("fair_score_k_raw") and not k.endswith("fair_score_eta_raw")
        }
        model.load_state_dict(state, strict=False)
    else:
        raise KeyError("Stage-1 checkpoint must contain pred_X_state_dict/pred_E_state_dict or model.")

    model._fixed_X_one_hot_3d = data_kwargs["X_one_hot_3d"]
    model._fixed_s = data_kwargs["s"]
    model._fixed_y = data_kwargs["y"]
    return model, dataset, train_yaml_data


def fixed_sample_kwargs(model):
    return {
        "fixed_X_one_hot_3d": getattr(model, "_fixed_X_one_hot_3d", None),
        "fixed_s": getattr(model, "_fixed_s", None),
        "fixed_y": getattr(model, "_fixed_y", None),
    }


def make_log_dir(args, dataset):
    if args.out_dir is not None and args.log_home is None:
        log_dir = args.out_dir
    else:
        log_base = args.log_home if args.log_home is not None else "./wandb"
        log_dir = os.path.join(log_base, dataset, "Sync", "controller", args.name)
    check_dir = os.path.join(log_dir, "check")
    os.makedirs(check_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    write_latest_run_manifest(args, dataset, log_dir, check_dir)
    return log_dir, check_dir


def write_latest_run_manifest(args, dataset, log_dir, check_dir):
    controller_root = Path(log_dir).parent
    manifest = {
        "dataset": dataset,
        "name": args.name,
        "log_dir": log_dir,
        "check_dir": check_dir,
        "base_model_path": args.controller_pretrained_ckpt,
        "full_model_best": os.path.join(check_dir, "full_model_best.pt"),
        "controller_best": os.path.join(check_dir, "controller_best.pt"),
        "sample_command": (
            "python sample.py "
            f"--model_path {os.path.join(check_dir, 'full_model_best.pt')} "
            f"--num_samples {args.num_generation} "
            "--fair_score_sp "
            f"--save_samples --save_dir {os.path.join(log_dir, 'generated_samples')} "
            f"--device {args.device} "
            "--skip_internal_eval"
        ),
    }
    try:
        controller_root.mkdir(parents=True, exist_ok=True)
        with open(controller_root / "latest_run.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(controller_root / "latest_run.txt", "w", encoding="utf-8") as f:
            f.write(log_dir + "\n")
    except OSError as exc:
        print(f"[WARN] Could not write latest controller manifest: {exc}")


def write_compat_pth_alias(path):
    path = Path(path)
    if path.suffix != ".pt":
        return
    alias = path.with_suffix(".pth")
    try:
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.link(path, alias)
    except OSError as exc:
        print(f"[WARN] Could not create compatibility checkpoint alias {alias}: {exc}")


def grad_diagnostics(model):
    def summarize(grad):
        if grad is None:
            return 0, 0.0, 0.0
        grad_abs = grad.detach().abs()
        if grad_abs.numel() == 0:
            return 0, 0.0, 0.0
        return (
            int((grad_abs > 0).sum().item()),
            float(grad_abs.mean().cpu()),
            float(grad_abs.max().cpu()),
        )

    k_nonzero, k_mean_abs, k_max_abs = summarize(model.fair_score_k_raw.grad)
    eta_nonzero, eta_mean_abs, eta_max_abs = summarize(model.fair_score_eta_raw.grad)
    return {
        "grad/k_nonzero": k_nonzero,
        "grad/eta_nonzero": eta_nonzero,
        "grad/k_mean_abs": k_mean_abs,
        "grad/eta_mean_abs": eta_mean_abs,
        "grad/k_max_abs": k_max_abs,
        "grad/eta_max_abs": eta_max_abs,
    }


def controller_stats(model, stats):
    stats = dict(stats or {})
    stats["fair_controller_k_schedule"] = model._get_effective_fair_score_k().detach().cpu().tolist()
    stats["fair_controller_eta_schedule"] = model._get_effective_fair_score_eta().detach().cpu().tolist()
    return stats


def checkpoint_payload(model, optimizer, epoch, loss, stats, args, dataset, train_yaml_data):
    stats = controller_stats(model, stats)
    return {
        "dataset": dataset,
        "train_yaml_data": train_yaml_data,
        "diffusion_stage": "stage2_controller",
        "base_model_path": args.controller_pretrained_ckpt,
        "epoch": int(epoch),
        "current_epoch": int(epoch),
        "loss": float(loss.detach().cpu()),
        "stats": stats,
        "args": vars(args),
        "controller": model.get_fair_controller_state_dict(),
        "pred_X_state_dict": deepcopy(model.graph_encoder.pred_X.state_dict()),
        "pred_E_state_dict": deepcopy(model.graph_encoder.pred_E.state_dict()),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def save_controller_checkpoint(check_dir, filename, epoch, model, optimizer, loss, stats, args, dataset, train_yaml_data):
    controller_path = os.path.join(check_dir, filename)
    full_name = filename.replace("controller_", "full_model_", 1)
    full_path = os.path.join(check_dir, full_name)

    payload = checkpoint_payload(model, optimizer, epoch, loss, stats, args, dataset, train_yaml_data)
    controller_payload = {
        "controller": payload["controller"],
        "epoch": payload["epoch"],
        "stats": payload["stats"],
        "args": payload["args"],
        "dataset": dataset,
        "train_yaml_data": train_yaml_data,
        "base_model_path": args.controller_pretrained_ckpt,
    }
    torch.save(controller_payload, controller_path)
    torch.save(payload, full_path)
    write_compat_pth_alias(controller_path)
    write_compat_pth_alias(full_path)
    return controller_path, full_path


@torch.no_grad()
def record_controller_replays(args, model):
    replays = []
    was_training = model.training
    model.eval()
    for _ in range(args.controller_replay_num_samples):
        sample_out = model.sample(
            is_diff_X=True,
            batch_size=args.sample_batch_size,
            num_workers=args.num_workers,
            **fixed_sample_kwargs(model),
            return_controller_replay=True,
        )
        replay = sample_out[-1]
        validate_controller_replay(replay, model)
        replays.append(replay)
        del sample_out
        if args.empty_cache_after_sampling and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if was_training:
        model.train()
    return replays


def validate_controller_replay(replay, model):
    required = ["s_0", "edge_group_labels", "E_init", "X_init", "full_mask", "h_init", "steps"]
    missing = [key for key in required if key not in replay]
    if missing:
        raise KeyError(f"controller replay is missing required field(s): {missing}")
    if len(replay["steps"]) != model.T:
        raise ValueError(f"controller replay has {len(replay['steps'])} steps, expected {model.T}.")
    num_full_edges = int(replay.get("num_full_edges", replay["full_mask"].numel()))
    if int(replay["full_mask"].numel()) != num_full_edges:
        raise ValueError("controller replay full_mask length does not match num_full_edges.")
    for step in replay["steps"]:
        if "z_raw" not in step:
            raise KeyError("controller replay step is missing z_raw.")
        if "edge_sample_gumbel" not in step:
            raise KeyError("controller replay step is missing edge_sample_gumbel.")
        if "X_next" not in step:
            raise KeyError("controller replay step is missing X_next.")
        if int(step["z_raw"].numel()) != num_full_edges:
            raise ValueError(
                f"controller replay z_raw length {step['z_raw'].numel()} does not match num_full_edges={num_full_edges}."
            )
        if tuple(step["edge_sample_gumbel"].shape) != (num_full_edges, 2):
            raise ValueError(
                "controller replay edge_sample_gumbel shape "
                f"{tuple(step['edge_sample_gumbel'].shape)} does not match ({num_full_edges}, 2)."
            )


def replay_summary(replays):
    num_replays = len(replays)
    num_steps = sum(len(replay.get("steps", [])) for replay in replays)
    num_z = sum(
        int(step["z_raw"].numel())
        for replay in replays
        for step in replay.get("steps", [])
        if "z_raw" in step
    )
    has_y = sum(1 for replay in replays if replay.get("y_0", None) is not None)
    has_step_noise = sum(
        1
        for replay in replays
        for step in replay.get("steps", [])
        if "edge_sample_gumbel" in step and "X_next" in step
    )
    return {
        "num_replays": num_replays,
        "num_steps": num_steps,
        "num_z_raw_values": num_z,
        "num_replays_with_y": has_y,
        "num_steps_with_fixed_noise": has_step_noise,
    }


def load_controller_replays(path, model):
    payload = torch_load(path, map_location="cpu")
    replays = payload.get("replays", payload) if isinstance(payload, dict) else payload
    if not isinstance(replays, list):
        replays = [replays]
    for replay in replays:
        validate_controller_replay(replay, model)
    print(f"[controller replay] loaded fixed replay from {path}: {replay_summary(replays)}")
    return replays


def save_controller_replays(path, replays):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({"replays": replays, "summary": replay_summary(replays)}, path)
    print(f"[controller replay] saved z_raw/y replay cache to {path}: {replay_summary(replays)}")


def compute_controller_loss_from_replays(model, replays):
    losses = []
    stats_bucket = []
    for replay in replays:
        loss, stats = model.compute_fair_controller_loss_from_replay(replay)
        losses.append(loss)
        stats_bucket.append(stats)
    loss = torch.stack(losses).mean()

    merged = {}
    keys = set().union(*(stats.keys() for stats in stats_bucket))
    for key in sorted(keys):
        vals = [stats[key] for stats in stats_bucket if key in stats]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            merged[key] = float(sum(float(v) for v in vals) / len(vals))
    return loss, merged


@torch.no_grad()
def sample_controller_pyg_graphs(args, model, total=None):
    total = int(total if total is not None else args.num_generation)
    generated_graphs = []
    was_training = model.training
    model.eval()
    for _ in range(total):
        replay_out = model.sample(
            is_diff_X=True,
            batch_size=args.sample_batch_size,
            num_workers=args.num_workers,
            **fixed_sample_kwargs(model),
            return_controller_replay=True,
        )
        replay = replay_out[-1]
        del replay_out
        sample_out = model.sample(
            is_diff_X=True,
            batch_size=args.sample_batch_size,
            num_workers=args.num_workers,
            sp_shift=True,
            controller_replay=replay,
        )
        X_0_one_hot, s_0_one_hot, y_0_one_hot, E_0, node_orig_id = sample_out[:5]
        pyg_data = build_pyg_data_from_sample(
            X_0_one_hot=X_0_one_hot.cpu(),
            s_0_one_hot=s_0_one_hot.cpu(),
            y_0_one_hot=y_0_one_hot.cpu() if y_0_one_hot is not None else None,
            E_0=E_0.cpu(),
            node_orig_id=node_orig_id.cpu(),
        )
        generated_graphs.append(pyg_data)
        del sample_out
        if args.empty_cache_after_sampling and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if was_training:
        model.train()
    return generated_graphs


def save_generated_graphs_for_lp(args, model, log_dir, tag, epoch, checkpoint_path, dataset):
    generated_dir = os.path.join(log_dir, "generated_samples")
    os.makedirs(generated_dir, exist_ok=True)

    graph_path = os.path.join(generated_dir, f"{tag}.pyg_full.pt")
    meta_path = os.path.join(generated_dir, f"{tag}.meta.json")
    generated_graphs = sample_controller_pyg_graphs(args, model, total=args.num_generation)
    torch.save(generated_graphs, graph_path)

    label_attr = getattr(args, "fair_label_attr", "y")
    eval_command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_generated_graphs.py")),
        "--graph_path", graph_path,
        "--dataset", dataset,
        "--label_attr", label_attr,
        "--sensitive_attr", label_attr,
        "--device", str(args.device),
    ]
    if args.generated_eval_max_graphs is not None:
        eval_command += ["--max_graphs", str(args.generated_eval_max_graphs)]
    if args.generated_eval_lp_epochs is not None:
        eval_command += ["--lp_epochs", str(args.generated_eval_lp_epochs)]
    if args.generated_eval_lp_patience is not None:
        eval_command += ["--lp_patience", str(args.generated_eval_lp_patience)]
    if args.generated_eval_lp_batch_size is not None:
        eval_command += ["--lp_batch_size", str(args.generated_eval_lp_batch_size)]
    eval_command_text = " ".join(eval_command)
    meta = {
        "tag": tag,
        "epoch": int(epoch) if epoch is not None else None,
        "num_graphs": len(generated_graphs),
        "graph_path": graph_path,
        "checkpoint_path": checkpoint_path,
        "evaluate_generated_graphs_command": eval_command_text,
        "run_generated_eval": bool(args.run_generated_eval),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[controller] Saved best generated PyG graphs for LP evaluation: {graph_path}")
    print(f"[controller] evaluate_generated_graphs.py command:\n{eval_command_text}")
    if args.run_generated_eval:
        print("[controller] Running evaluate_generated_graphs.py now...")
        subprocess.run(eval_command, check=True)
    return graph_path


def debug_controller_shapes(model):
    print(f"[controller debug] fair_score_k_raw.shape: {tuple(model.fair_score_k_raw.shape)}")
    print(f"[controller debug] fair_score_eta_raw.shape: {tuple(model.fair_score_eta_raw.shape)}")
    print(f"[controller debug] effective k shape: {tuple(model._get_effective_fair_score_k().shape)}")
    print(f"[controller debug] effective eta shape: {tuple(model._get_effective_fair_score_eta().shape)}")
    print(f"[controller debug] expected raw/effective full shapes: [{model.T}]")


def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 training for FairWire eta/k fairness controller."
    )
    parser.add_argument("--controller_pretrained_ckpt", type=str, default=None, help="Stage-1 FairWire checkpoint.")
    parser.add_argument("--model_path", type=str, default=None, help="Legacy alias for --controller_pretrained_ckpt.")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log_home", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cuda:1 or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fair_label_attr", type=str, default="y")
    parser.add_argument("--controller_epochs", type=int, default=1000)
    parser.add_argument("--controller_lr", type=float, default=1e-3)
    parser.add_argument("--controller_replay_num_samples", type=int, default=1)
    parser.add_argument("--controller_replay_refresh", type=int, default=100,
                        help="Only used when --controller_replay_fixed False.")
    parser.add_argument("--controller_replay_fixed", type=eval, default=False,
                        help="Reuse one frozen stage-1 z_raw/y replay for all epochs. Default False matches EDGE_fairness_loss_after refresh behavior.")
    parser.add_argument("--controller_replay_path", type=str, default=None,
                        help="Optional saved replay cache containing stage-1 z_raw/y values.")
    parser.add_argument("--controller_replay_save_path", type=str, default="auto",
                        help="Where to save the first z_raw/y replay cache. Use 'auto' for log_dir/controller_replay.pt.")
    parser.add_argument("--sample_batch_size", type=int, default=32768)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_generation", type=int, default=64)
    parser.add_argument("--eval_num_generation", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--check_every", type=int, default=None)
    parser.add_argument("--cpu_offload_generated", type=eval, default=True)
    parser.add_argument("--empty_cache_after_sampling", type=eval, default=True)
    parser.add_argument("--fair_score_eta", type=float, default=0.0)
    parser.add_argument("--fair_score_k", type=float, default=0.15)
    parser.add_argument("--fair_score_eta_scale", type=float, default=1.0)
    parser.add_argument("--fair_score_learn_k", type=eval, default=True)
    parser.add_argument("--fair_score_learn_eta", type=eval, default=True)
    parser.add_argument("--fair_score_train_loss_weight", type=float, default=1.0, help="Legacy EDGE-compatible no-op.")
    parser.add_argument("--fair_score_guidance_normalize", type=eval, default=True)
    parser.add_argument("--fair_score_fair_loss_weight", type=float, default=1.0)
    parser.add_argument("--fair_score_k_tracking_loss_weight", type=float, default=1.0)
    parser.add_argument("--fair_score_utility_loss_weight", type=float, default=1.0)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--clip_value", type=float, default=None)
    parser.add_argument("--controller_debug_shapes", action="store_true")
    parser.add_argument("--run_generated_eval", action="store_true",
                        help="After exporting controller_best generated graphs, run evaluate_generated_graphs.py automatically.")
    parser.add_argument("--generated_eval_max_graphs", type=int, default=None,
                        help="Optional --max_graphs forwarded to automatic evaluate_generated_graphs.py run.")
    parser.add_argument("--generated_eval_lp_epochs", type=int, default=None,
                        help="Optional --lp_epochs forwarded to automatic evaluate_generated_graphs.py run.")
    parser.add_argument("--generated_eval_lp_patience", type=int, default=None,
                        help="Optional --lp_patience forwarded to automatic evaluate_generated_graphs.py run.")
    parser.add_argument("--generated_eval_lp_batch_size", type=int, default=None,
                        help="Optional --lp_batch_size forwarded to automatic evaluate_generated_graphs.py run.")
    args = prepare_args(parser.parse_args())

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[controller] device={device}")

    checkpoint = torch_load(args.controller_pretrained_ckpt, map_location="cpu")
    model, dataset, train_yaml_data = build_model(args, checkpoint, device)
    if args.controller_debug_shapes:
        debug_controller_shapes(model)
    controller_params = model.freeze_for_fair_controller_training()
    optimizer = torch.optim.Adam(controller_params, lr=args.controller_lr)

    log_dir, check_dir = make_log_dir(args, dataset)
    metrics_path = os.path.join(log_dir, "controller_metrics.jsonl")
    print(f"[controller] dataset={dataset}")
    print(f"Storing controller logs in: {log_dir}")
    print(f"Storing controller checkpoints in: {check_dir}")
    if args.controller_replay_fixed:
        print(
            "[controller replay] fixed replay enabled: "
            "record/load one frozen stage-1 z_raw/y trajectory and train eta/k only through logit shift."
        )
    else:
        print(
            "[controller replay] EDGE-after refresh mode: "
            f"record a fresh stage-1 z_raw/y replay every {args.controller_replay_refresh} epoch(s)."
        )

    replays = None
    replay_save_path = args.controller_replay_save_path
    if replay_save_path == "auto":
        replay_save_path = os.path.join(log_dir, "controller_replay.pt")
    if args.controller_replay_path is not None:
        replays = load_controller_replays(args.controller_replay_path, model)
        args.controller_replay_fixed = True
    last_loss = None
    last_stats = None
    best_loss = None
    eta_zero_grad_epochs = 0
    eta_zero_grad_warn_after = 10

    for epoch in range(args.controller_epochs):
        should_refresh = (
            replays is None
            or (not args.controller_replay_fixed and epoch % args.controller_replay_refresh == 0)
        )
        if should_refresh:
            replays = record_controller_replays(args, model)
            print(f"[controller replay] using frozen stage-1 z_raw/y replay: {replay_summary(replays)}")
            if epoch == 0 and replay_save_path:
                save_controller_replays(replay_save_path, replays)

        optimizer.zero_grad()
        loss, stats = compute_controller_loss_from_replays(model, replays)
        loss.backward()

        grad_stats = grad_diagnostics(model)
        stats.update(grad_stats)
        if grad_stats["grad/eta_nonzero"] == 0:
            eta_zero_grad_epochs += 1
            if eta_zero_grad_epochs >= eta_zero_grad_warn_after and eta_zero_grad_epochs % eta_zero_grad_warn_after == 0:
                print(
                    f"[WARN] eta gradients have been zero for {eta_zero_grad_epochs} consecutive epochs. "
                    "This can happen if replay steps have no active candidates for eta to affect."
                )
        else:
            eta_zero_grad_epochs = 0

        if args.clip_value is not None:
            torch.nn.utils.clip_grad_value_(controller_params, args.clip_value)
        if args.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(controller_params, args.clip_norm)

        optimizer.step()
        last_loss = loss.detach()

        row = {"epoch": epoch + 1, "loss": float(last_loss.cpu()), **stats}
        print("Controller epoch {}/{} | loss {:.6f} | fair {:.6f} | k_track {:.6f} | util {:.6f}".format(
            epoch + 1,
            args.controller_epochs,
            row["loss"],
            stats["fair_controller_fair_loss"],
            stats["fair_controller_k_tracking_loss"],
            stats["fair_controller_utility_loss"],
        ))

        if (epoch + 1) % args.eval_every == 0:
            eval_total = args.eval_num_generation if args.eval_num_generation is not None else args.num_generation
            print(f"[eval epoch {epoch + 1}] FairWire controller eval export is deferred to final best export; eval_num_generation={eval_total}")

        stats_for_save = {k: v for k, v in row.items() if k != "epoch"}
        last_stats = stats_for_save
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        save_controller_checkpoint(
            check_dir,
            "controller_last.pt",
            epoch,
            model,
            optimizer,
            loss,
            stats_for_save,
            args,
            dataset,
            train_yaml_data,
        )

        loss_value = row["loss"]
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            save_controller_checkpoint(
                check_dir,
                "controller_best.pt",
                epoch,
                model,
                optimizer,
                loss,
                stats_for_save,
                args,
                dataset,
                train_yaml_data,
            )

        if (epoch + 1) % args.check_every == 0:
            save_controller_checkpoint(
                check_dir,
                f"controller_checkpoint_{epoch}.pt",
                epoch,
                model,
                optimizer,
                loss,
                stats_for_save,
                args,
                dataset,
                train_yaml_data,
            )

    if last_loss is not None:
        save_controller_checkpoint(
            check_dir,
            "controller_final.pt",
            args.controller_epochs - 1,
            model,
            optimizer,
            last_loss,
            last_stats or {},
            args,
            dataset,
            train_yaml_data,
        )

    best_full_path = os.path.join(check_dir, "full_model_best.pt")
    if os.path.exists(best_full_path):
        print(f"[controller] Loading best full model checkpoint for generated-graph export: {best_full_path}")
        best_checkpoint = torch_load(best_full_path, map_location=device)
        model.load_state_dict(best_checkpoint["model"])
        best_epoch = best_checkpoint.get("epoch", best_checkpoint.get("current_epoch", None))
        save_generated_graphs_for_lp(
            args=args,
            model=model,
            log_dir=log_dir,
            tag="controller_best",
            epoch=best_epoch,
            checkpoint_path=best_full_path,
            dataset=dataset,
        )
        print(
            "[controller] Sample with:\n"
            "python sample.py "
            f"--model_path {best_full_path} "
            f"--num_samples {args.num_generation} "
            "--fair_score_sp "
            f"--save_samples --save_dir {os.path.join(log_dir, 'generated_samples')} "
            f"--device {args.device} "
            "--skip_internal_eval"
        )
    else:
        print(f"[WARN] Best full model checkpoint not found; skipping generated-graph export: {best_full_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch_geometric as pyg

from diffusion.utils import add_parent_path, get_args_table, set_seeds

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
torch.set_num_threads(4)


# Data
add_parent_path(level=1)
from datasets.data import add_data_args, get_data, get_data_id

# Exp args only; do not use GraphExperiment/elbo_bpd here.
from experiment import add_exp_args

# Model
from model import add_model_args, get_model, get_model_id

# Optim args for CLI compatibility with train.py.
from diffusion.optim.multistep import add_optim_args


def prepare_args(args):
    args.fair_score_sp = True
    args.fair_score_learn_k = True
    args.fair_score_learn_eta = True
    args.fair_score_controller_train = True

    if args.controller_pretrained_ckpt is None:
        raise ValueError("--controller_pretrained_ckpt is required for controller training")
    if args.eval_every is None:
        args.eval_every = args.controller_epochs
    if args.check_every is None:
        args.check_every = args.controller_epochs
    args.eval_every = max(1, int(args.eval_every))
    args.check_every = max(1, int(args.check_every))
    if args.name is None:
        args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.controller_replay_refresh = max(1, int(args.controller_replay_refresh))
    args.controller_replay_num_samples = max(1, int(args.controller_replay_num_samples))
    return args


def prepare_data_args(args, data_tuple):
    (
        train_loader,
        eval_loader,
        test_loader,
        num_node_feat,
        num_node_classes,
        num_edge_classes,
        max_degree,
        augmented_feature_dict,
        initial_graph_sampler,
        eval_evaluator,
        test_evaluator,
        monitoring_statistics,
    ) = data_tuple

    args.num_edge_classes = num_edge_classes
    args.num_node_classes = num_node_classes

    if args.final_prob_node is None:
        args.final_prob_node = [1 - 1e-12, 1e-12]
        args.num_node_classes = 2
        args.has_node_feature = False

    if 0 in args.final_prob_edge:
        args.final_prob_edge[np.argmax(args.final_prob_edge)] -= 1e-12
        args.final_prob_edge[np.argmin(args.final_prob_edge)] = 1e-12

    args.max_degree = max_degree
    args.num_node_feat = num_node_feat
    args.augmented_feature_dict = augmented_feature_dict

    return (
        train_loader,
        eval_loader,
        test_loader,
        initial_graph_sampler,
        eval_evaluator,
        test_evaluator,
        monitoring_statistics,
    )


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pretrained_model(model, ckpt_path, device):
    checkpoint = torch_load_checkpoint(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict.pop("fair_score_k_raw", None)
    state_dict.pop("fair_score_eta_raw", None)
    state_dict.pop("module.fair_score_k_raw", None)
    state_dict.pop("module.fair_score_eta_raw", None)
    print(
        "[controller] Ignoring fair_score_k_raw/fair_score_eta_raw from pretrained denoiser checkpoint; "
        "initializing per-step controller parameters from CLI values."
    )
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"fair_score_k_raw", "fair_score_eta_raw"}
    notable_missing = [k for k in missing if k not in allowed_missing]
    if notable_missing:
        print(f"[WARN] Missing keys while loading pretrained model: {notable_missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading pretrained model: {unexpected}")
    return checkpoint


def make_log_dir(args, data_id, model_id):
    log_base = args.log_home if args.log_home is not None else "./wandb"
    log_dir = os.path.join(log_base, data_id, model_id, "controller", args.name)
    check_dir = os.path.join(log_dir, "check")
    os.makedirs(check_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.pickle"), "wb") as f:
        pickle.dump(args, f)
    with open(os.path.join(log_dir, "args_table.txt"), "w") as f:
        f.write(str(get_args_table(vars(args))))
    return log_dir, check_dir


def to_networkx_graph(pyg_data, label_attr="y"):
    node_attrs = []
    for attr in ("y", "orig_id", label_attr):
        if hasattr(pyg_data, attr) and getattr(pyg_data, attr) is not None and attr not in node_attrs:
            node_attrs.append(attr)
    if node_attrs:
        return pyg.utils.to_networkx(pyg_data, to_undirected=True, node_attrs=node_attrs)
    return pyg.utils.to_networkx(pyg_data, to_undirected=True)


def _controller_sample_batch_size(args, total):
    sample_batch_size = getattr(args, "sample_batch_size", None)
    if sample_batch_size is None:
        sample_batch_size = 1
    return max(1, min(int(sample_batch_size), int(total)))


@torch.no_grad()
def sample_controller_pyg_graphs(args, model, total=None):
    total = int(total if total is not None else args.num_generation)
    sample_batch_size = _controller_sample_batch_size(args, total)

    generated_pyg_datas = []
    done = 0
    was_training = model.training
    model.eval()
    while done < total:
        cur_batch = min(sample_batch_size, total - done)
        replay_graph, replay = model.sample(cur_batch, return_controller_replay=True)
        del replay_graph
        generated = model.sample(cur_batch, controller_replay=replay).cpu()
        generated_pyg_datas.extend(data.cpu() for data in generated.to_data_list())
        done += cur_batch
        del generated, replay
        if getattr(args, "empty_cache_after_sampling", True) and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if was_training:
        model.train()
        model._denoise_fn.eval()
    return generated_pyg_datas


@torch.no_grad()
def evaluate_controller(args, model, evaluator):
    total = int(getattr(args, "eval_num_generation", None) or args.num_generation)
    sample_batch_size = _controller_sample_batch_size(args, total)

    generated_graphs = []
    fair_shift_values = []
    done = 0
    was_training = model.training
    model.eval()
    while done < total:
        cur_batch = min(sample_batch_size, total - done)
        replay_graph, replay = model.sample(cur_batch, return_controller_replay=True)
        del replay_graph
        generated = model.sample(cur_batch, controller_replay=replay)
        if hasattr(model, "_last_fair_guidance_mean_abs_shift"):
            fair_shift_values.append(float(model._last_fair_guidance_mean_abs_shift))
        if getattr(args, "cpu_offload_generated", True):
            generated = generated.cpu()
        for pyg_data in generated.to_data_list():
            generated_graphs.append(to_networkx_graph(pyg_data, getattr(args, "fair_label_attr", "y")))
        done += cur_batch
        del generated, replay
        if getattr(args, "empty_cache_after_sampling", True) and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if was_training:
        model.train()
        model._denoise_fn.eval()
    metrics = evaluator.evaluate(generated_graphs)
    if fair_shift_values:
        metrics["fair_guidance_mean_abs_shift"] = float(np.mean(fair_shift_values))
    return metrics


def controller_grad_diagnostics(model):
    def _summarize_grad(grad):
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

    k_nonzero, k_mean_abs, k_max_abs = _summarize_grad(model.fair_score_k_raw.grad)
    eta_nonzero, eta_mean_abs, eta_max_abs = _summarize_grad(model.fair_score_eta_raw.grad)
    return {
        "grad/k_nonzero": k_nonzero,
        "grad/eta_nonzero": eta_nonzero,
        "grad/k_mean_abs": k_mean_abs,
        "grad/eta_mean_abs": eta_mean_abs,
        "grad/k_max_abs": k_max_abs,
        "grad/eta_max_abs": eta_max_abs,
    }


@torch.no_grad()
def debug_controller_shapes(args, model):
    print(f"[controller debug] fair_score_k_raw.shape: {model.fair_score_k_raw.shape}")
    print(f"[controller debug] fair_score_eta_raw.shape: {model.fair_score_eta_raw.shape}")
    print(f"[controller debug] effective k shape: {model._get_effective_fair_score_k().shape}")
    print(f"[controller debug] effective eta shape: {model._get_effective_fair_score_eta().shape}")

    t_graph = torch.tensor(
        [0, args.diffusion_steps // 2, args.diffusion_steps - 1],
        device=args.device,
    )
    print(f"[controller debug] t_graph: {t_graph}")
    print(f"[controller debug] effective k[t_graph]: {model._get_effective_fair_score_k(t_graph=t_graph)}")
    print(f"[controller debug] effective eta[t_graph]: {model._get_effective_fair_score_eta(t_graph=t_graph)}")
    print(
        "[controller debug] expected raw/effective full shapes: "
        f"[{args.diffusion_steps}], indexed shape: [3]"
    )


def save_controller_checkpoint(check_dir, filename, epoch, model, optimizer, loss, stats, args):
    controller_path = os.path.join(check_dir, filename)
    full_name = filename.replace("controller_", "full_model_", 1)
    full_path = os.path.join(check_dir, full_name)
    stats = dict(stats or {})
    with torch.no_grad():
        stats["fair_controller_k_schedule"] = (
            model._get_effective_fair_score_k().detach().cpu().tolist()
        )
        stats["fair_controller_eta_schedule"] = (
            model._get_effective_fair_score_eta().detach().cpu().tolist()
        )
    controller_checkpoint = {
        "controller": model.get_fair_controller_state_dict(),
        "epoch": int(epoch),
        "stats": stats,
        "args": vars(args),
    }
    torch.save(controller_checkpoint, controller_path)
    torch.save(
        {
            "epoch": int(epoch),
            "current_epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": float(loss.detach().cpu()),
            "stats": stats,
            "args": vars(args),
        },
        full_path,
    )
    return controller_path, full_path


def save_generated_graphs_for_lp(args, model, log_dir, tag, epoch, checkpoint_path):
    generated_dir = os.path.join(log_dir, "generated_samples")
    os.makedirs(generated_dir, exist_ok=True)

    num_samples = int(args.num_generation)
    graph_path = os.path.join(generated_dir, f"{tag}.pyg_full.pt")
    meta_path = os.path.join(generated_dir, f"{tag}.meta.json")
    generated_graphs = sample_controller_pyg_graphs(args, model, total=num_samples)
    torch.save(generated_graphs, graph_path)

    label_attr = getattr(args, "fair_label_attr", "y")
    eval_command = (
        "python evaluate_generated_graphs.py "
        f"--graph_path {graph_path} "
        f"--dataset {args.dataset} "
        f"--label_attr {label_attr} "
        f"--sensitive_attr {label_attr} "
        f"--device {args.device}"
    )
    meta = {
        "tag": tag,
        "epoch": int(epoch) if epoch is not None else None,
        "num_graphs": len(generated_graphs),
        "graph_path": graph_path,
        "checkpoint_path": checkpoint_path,
        "evaluate_generated_graphs_command": eval_command,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[controller] Saved best generated PyG graphs for LP evaluation: {graph_path}")
    print(f"[controller] evaluate_generated_graphs.py command:\n{eval_command}")
    return graph_path


def main():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_exp_args(parser)
    add_model_args(parser)
    add_optim_args(parser)
    parser.add_argument(
        "--controller_debug_shapes",
        action="store_true",
        help="print per-step controller parameter/getter shape checks before training",
    )
    args = prepare_args(parser.parse_args())
    set_seeds(args.seed)

    if args.parallel == "dp":
        print("[WARN] train_controller.py trains controller parameters on a single model instance; ignoring --parallel dp.")

    data_tuple = get_data(args)
    (
        _train_loader,
        _eval_loader,
        _test_loader,
        initial_graph_sampler,
        eval_evaluator,
        _test_evaluator,
        _monitoring_statistics,
    ) = prepare_data_args(args, data_tuple)

    data_id = get_data_id(args)
    model = get_model(args, initial_graph_sampler=initial_graph_sampler)
    model_id = get_model_id(args)
    model = model.to(args.device)

    load_pretrained_model(model, args.controller_pretrained_ckpt, args.device)
    if args.controller_debug_shapes:
        debug_controller_shapes(args, model)
    controller_params = model.freeze_for_fair_controller_training()
    optimizer = torch.optim.Adam(controller_params, lr=args.controller_lr)

    log_dir, check_dir = make_log_dir(args, data_id, model_id)
    metrics_path = os.path.join(log_dir, "controller_metrics.jsonl")
    print(f"Storing controller logs in: {log_dir}")
    print(f"Storing controller checkpoints in: {check_dir}")

    replay = None
    last_loss = None
    last_stats = None
    best_loss = None
    eta_zero_grad_epochs = 0
    eta_zero_grad_warn_after = 10

    for epoch in range(args.controller_epochs):
        if epoch == 0 or epoch % args.controller_replay_refresh == 0:
            with torch.no_grad():
                replay_graph, replay = model.sample(
                    args.controller_replay_num_samples,
                    return_controller_replay=True,
                )
                replay_graph = None

        optimizer.zero_grad()
        loss, stats = model.compute_fair_controller_loss_from_replay(replay)
        loss.backward()

        if __debug__:
            assert model.fair_score_k_raw.numel() == args.diffusion_steps
            assert model.fair_score_eta_raw.numel() == args.diffusion_steps

        grad_stats = controller_grad_diagnostics(model)
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
            eval_metrics = evaluate_controller(args, model, eval_evaluator)
            row.update({f"eval/{k}": float(v) for k, v in eval_metrics.items() if np.isscalar(v)})
            print(f"[eval epoch {epoch + 1}] {eval_metrics}")

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
        )

    best_full_path = os.path.join(check_dir, "full_model_best.pt")
    if os.path.exists(best_full_path):
        print(f"[controller] Loading best full model checkpoint for generated-graph export: {best_full_path}")
        best_checkpoint = torch_load_checkpoint(best_full_path, map_location=args.device)
        model.load_state_dict(best_checkpoint["model"])
        best_epoch = best_checkpoint.get("epoch", best_checkpoint.get("current_epoch", None))
        save_generated_graphs_for_lp(
            args=args,
            model=model,
            log_dir=log_dir,
            tag="controller_best",
            epoch=best_epoch,
            checkpoint_path=best_full_path,
        )
    else:
        print(f"[WARN] Best full model checkpoint not found; skipping generated-graph export: {best_full_path}")


if __name__ == "__main__":
    main()

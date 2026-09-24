"""One fixed GCN fit per generated graph; no hyperparameter search entrypoint."""

from copy import deepcopy
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_generated_graphs import SamplePyGAE, safe_auc, samplepy_prepare_for_gae
from proxy_minimal_pilot import PILOT_GCN_CONFIG


CONFIG_KEYS = {
    "lr", "num_layers", "hidden_size", "dropout", "max_epochs", "patience",
    "batch_size", "min_delta", "weight_decay", "selection_metric",
}

class InvalidTerminalFit(ValueError):
    def __init__(self, message, fits):
        super().__init__(message)
        self.fits = fits


def _require_finite(value, description):
    """Observe numerics without changing fitting state or consuming randomness."""
    if not bool(torch.isfinite(value.detach()).all()):
        raise RuntimeError(f"Unexpected nonfinite GCN {description}; stop without retry")


def validate_config(config):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"A fixed GCN config must specify exactly {sorted(CONFIG_KEYS)}")
    if config["selection_metric"] != "validation_auc":
        raise ValueError("GCN epoch selection must use validation_auc")
    for key in ("num_layers", "hidden_size", "max_epochs", "patience", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid fixed GCN {key}")
    for key in ("lr", "dropout", "min_delta", "weight_decay"):
        if not isinstance(config[key], (float, int)) or not math.isfinite(config[key]):
            raise ValueError(f"Invalid fixed GCN {key}")
    if config["lr"] <= 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid fixed GCN learning rate/dropout")
    if config["min_delta"] < 0 or config["weight_decay"] < 0:
        raise ValueError("min_delta and weight_decay must be nonnegative")
    return dict(config)


def seed_all(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == "cuda":
        # Seed only the explicitly selected physical GPU.
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


class EmbeddingDecoder:
    """Decode bounded pair chunks using a single frozen terminal embedding."""

    def __init__(self, embedding, chunk_size=65536):
        self.embedding = embedding.detach()
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    @torch.no_grad()
    def __call__(self, pairs):
        pairs = torch.as_tensor(pairs, dtype=torch.long)
        output = []
        for start in range(0, pairs.shape[1], self.chunk_size):
            u, v = pairs[:, start:start + self.chunk_size].to(self.embedding.device)
            output.append(torch.sigmoid((self.embedding[u] * self.embedding[v]).sum(-1)).cpu())
        return torch.cat(output) if output else torch.empty(0)


class PrespecifiedPilotGAE(SamplePyGAE):
    """The pilot's one-layer GCN applies its fixed dropout to input features.

    The legacy encoder applies dropout only between hidden layers, which makes
    its dropout argument inert for one layer. This local pilot encoder keeps the
    same linear layer and train adjacency, while making the prescribed p=0.1
    effective during fitting. Evaluation and the frozen terminal embedding use
    the complete reference features. The strict evaluator is unchanged.
    """

    def forward(self, adjacency, features):
        return self.gcn(adjacency, F.dropout(
            features, p=self.gcn.dropout, training=self.training))


def fit_terminal_gcn(data, config, *, split_seed, fit_seed, device, chunk_size=65536,
                     policy="validation_selected"):
    """Fit once. Validation selects an epoch; held-out test never enters training.

    Retains the repository's GAE architecture, split rounding, normalized train
    adjacency and uniform ordered training-negative sampler (including possible
    self/positive collisions). No real-reference test pairs are used.
    """
    config = validate_config(config)
    if policy not in ("validation_selected", "prespecified_pilot"):
        raise ValueError(f"Unknown GCN preparation policy: {policy}")
    pilot = policy == "prespecified_pilot"
    if pilot and config != PILOT_GCN_CONFIG:
        raise ValueError("prespecified_pilot requires the exact prespecified GCN configuration")
    if getattr(data, "x", None) is None:
        raise ValueError("E1 requires the reference features; no feature fallback")
    seed_all(split_seed, "cpu")
    try:
        split = samplepy_prepare_for_gae(data)
    except ValueError as exc:
        raise InvalidTerminalFit(str(exc), fits=0) from exc
    train_pairs = split["train_mask"].nonzero().t().contiguous()
    val_pairs = split["val_mask"].nonzero().t().contiguous()
    test_pairs = split["test_mask"].nonzero().t().contiguous()
    val_labels = (split["A_full"][val_pairs[0], val_pairs[1]] != 0).long()
    test_labels = (split["A_full"][test_pairs[0], test_pairs[1]] != 0).long()
    # Dense normalization is shared with existing code; message passing is sparse.
    adjacency = split["A_train"].to_sparse().coalesce().to(device)
    features = data.x.detach().float().to(device)
    seed_all(fit_seed, device)
    model_class = PrespecifiedPilotGAE if pilot else SamplePyGAE
    model = model_class(features.shape[1], config["num_layers"],
                        config["hidden_size"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    train_pairs_device = train_pairs.to(device)
    best_auc, best_epoch, best_state, bad_epochs = -math.inf, 0, None, 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        embedding = model(adjacency, features)
        batch = train_pairs_device
        if batch.shape[1] > config["batch_size"]:
            indices = torch.randint(batch.shape[1], (config["batch_size"],), device=device)
            batch = batch[:, indices]
        negative = torch.randint(features.shape[0], batch.shape, device=device)
        pos_logits = (embedding[batch[0]] * embedding[batch[1]]).sum(-1)
        neg_logits = (embedding[negative[0]] * embedding[negative[1]]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat((pos_logits, neg_logits)),
            torch.cat((torch.ones_like(pos_logits), torch.zeros_like(neg_logits))))
        _require_finite(loss, f"training loss at epoch {epoch}")
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                _require_finite(parameter.grad, f"gradient {name} at epoch {epoch}")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_embedding = model(adjacency, features)
            _require_finite(val_embedding, f"validation embedding at epoch {epoch}")
            val_decoder = EmbeddingDecoder(val_embedding, chunk_size)
            val_scores = val_decoder(val_pairs)
            _require_finite(val_scores, f"validation scores at epoch {epoch}")
            val_auc = safe_auc(val_labels.numpy(), val_scores.numpy())
            if val_labels.unique().numel() >= 2 and not np.isfinite(val_auc):
                raise RuntimeError(
                    f"Unexpected nonfinite GCN validation AUC at epoch {epoch}; stop without retry")
        if np.isfinite(val_auc) and val_auc > best_auc + config["min_delta"]:
            best_auc, best_epoch = float(val_auc), epoch
            best_state, bad_epochs = deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if bad_epochs >= config["patience"]:
            break
    if best_state is None:
        raise InvalidTerminalFit("GCN had no finite validation AUC; keep this root invalid, do not retry", fits=1)
    model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        embedding = model(adjacency, features).detach()
        _require_finite(embedding, "terminal embedding")
    decoder = EmbeddingDecoder(embedding, chunk_size)
    return {
        "decoder": decoder, "embedding": embedding,
        "test_pairs": test_pairs, "test_labels": test_labels,
        "test_scores": decoder(test_pairs),
        "train_pairs": train_pairs, "val_pairs": val_pairs,
        "meta": {"gcn_fits": 1, "gcn_best_epoch": best_epoch,
                 "gcn_epochs_run": epoch, "gcn_best_val_auc": best_auc,
                 "gcn_policy": policy, "gcn_config": config,
                 "gcn_optimizer": "Adam",
                 "gcn_dropout_application": "input_features_training_only" if pilot else "legacy_hidden_layers_only",
                 "split_seed": split_seed, "fit_seed": fit_seed,
                 "train_num_pos": int(split["num_train_pos"]),
                 "val_num_pos": int(split["num_val_pos"]),
                 "test_num_pos": int(split["num_test_pos"]),
                 "test_num_neg": int(split["num_test_neg"])},
    }

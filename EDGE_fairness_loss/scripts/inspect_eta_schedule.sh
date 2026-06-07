#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 <controller_checkpoint.pt> [out_csv] [top_k=12] [softmax_tau=1.0]" >&2
  echo "Example: $0 wandb/cora/multinomial_diffusion/controller/RUN/check/controller_last.pt eta_schedule.csv 20" >&2
  exit 2
fi

CKPT_PATH="$1"
OUT_CSV="${2:-}"
TOP_K="${3:-12}"
SOFTMAX_TAU="${4:-1.0}"

python - "$CKPT_PATH" "$OUT_CSV" "$TOP_K" "$SOFTMAX_TAU" <<'PY'
import csv
import sys

import torch
import torch.nn.functional as F

ckpt_path, out_csv, top_k_raw, tau_raw = sys.argv[1:5]
top_k = int(top_k_raw)
tau = float(tau_raw)

ckpt = torch.load(ckpt_path, map_location="cpu")
controller = ckpt.get("controller", ckpt)

eta_raw = controller["fair_score_eta_raw"].detach().float().reshape(-1)
k_raw = controller.get("fair_score_k_raw")
k_raw = k_raw.detach().float().reshape(-1) if k_raw is not None else None

T = int(controller.get("num_timesteps", eta_raw.numel()))
base_eta = float(controller.get("fair_score_eta_base", controller.get("fair_score_eta", 1.0)))
eta_mode = str(controller.get("fair_score_eta_mode", "per_step_multiplier_softplus"))

if eta_raw.numel() != T:
    raise ValueError(f"eta_raw has {eta_raw.numel()} entries, but num_timesteps={T}")

if eta_mode == "per_step_multiplier_softplus":
    denom = F.softplus(torch.zeros((), dtype=eta_raw.dtype))
    eta = base_eta * F.softplus(eta_raw) / denom
elif "softmax" in eta_mode:
    weights = torch.softmax(eta_raw / tau, dim=0)
    eta = base_eta * T * weights
else:
    raise ValueError(f"Unsupported eta mode: {eta_mode}")

if k_raw is not None:
    k = torch.sigmoid(k_raw)
else:
    k = torch.full_like(eta, float("nan"))

mult = eta / base_eta if base_eta != 0 else torch.full_like(eta, float("nan"))
order_hi = torch.argsort(eta, descending=True)
order_lo = torch.argsort(eta, descending=False)
top_k = min(top_k, T)

def f(x):
    return float(x.detach().cpu())

epoch = ckpt.get("epoch", ckpt.get("current_epoch", None))
print(f"checkpoint: {ckpt_path}")
print(f"epoch: {epoch}")
print(f"eta_mode: {eta_mode}")
print(f"base_eta: {base_eta:g}")
print(f"num_timesteps: {T}")
print("t convention: t=0 is final reverse step A^1 -> A^0; t=T-1 is first reverse step A^T -> A^{T-1}")
print()
print(f"eta mean/min/max: {f(eta.mean()):.6g} / {f(eta.min()):.6g} / {f(eta.max()):.6g}")
print(f"eta multiplier mean/min/max: {f(mult.mean()):.6g} / {f(mult.min()):.6g} / {f(mult.max()):.6g}")
print(f"eta std: {f(eta.std(unbiased=False)):.6g}")
print(f"eta max/min ratio: {f(eta.max() / eta.clamp_min(1e-30).min()):.6g}")
print(f"k mean/min/max: {f(k.mean()):.6g} / {f(k.min()):.6g} / {f(k.max()):.6g}")
print()

for label, idxs in (("largest eta_t", order_hi[:top_k]), ("smallest eta_t", order_lo[:top_k])):
    print(label)
    print("rank\tt\teta_t\tmultiplier\teta_raw\tk_t")
    for rank, idx in enumerate(idxs.tolist(), 1):
        print(
            f"{rank}\t{idx}\t{f(eta[idx]):.6g}\t{f(mult[idx]):.6g}\t"
            f"{f(eta_raw[idx]):.6g}\t{f(k[idx]):.6g}"
        )
    print()

if out_csv:
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["t", "eta_t", "eta_multiplier", "eta_raw", "k_t"])
        for t in range(T):
            writer.writerow([t, f(eta[t]), f(mult[t]), f(eta_raw[t]), f(k[t])])
    print(f"wrote csv: {out_csv}")
PY

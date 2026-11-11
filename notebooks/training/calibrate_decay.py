"""Calibrate and compare one-step vs horizon-aware PoT decay on real held-out text.

Example:
  python calibrate_decay.py --pretrained state-spaces/mamba2-130m \
    --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
    --tokens 200000 --out results/decay_calibration.json

The script streams the deterministic validation-document shard; it does not
create synthetic language-model data. Results characterize the model's decay
coefficients on this calibration corpus and are not a benchmark score.
"""
import argparse, json, os, itertools
import torch
from ssm_core import load_pretrained, QuantConfig, get_tokenizer
from retention import exponent_grid, horizon_aware_exponent, retention_error_report, dyadic_levels, retention_costs


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="state-spaces/mamba2-130m")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--val-modulo", type=int, default=100)
    p.add_argument("--tokens", type=int, default=200000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=14)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--weight", choices=["uniform", "log", "early", "late"], default="uniform")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--dyadic-terms", type=int, default=2,
                   help="also score a multi-term dyadic leak grid (1 - sum 2^-k_i); 0 disables")
    p.add_argument("--max-coefficients", type=int, default=20000,
                   help="cap calibration coefficients to bound the HARPQ candidate-search memory")
    p.add_argument("--out", required=True)
    return p.parse_args()


def dyadic_select(a, levels, horizon, mode, chunk=262144):
    """Best dyadic level per coefficient under the same retention objective (chunked to bound GPU memory)."""
    hats, costs = [], []
    for lo in range(0, a.numel(), chunk):
        c = retention_costs(a[lo:lo + chunk].float(), levels, horizon, mode)
        m, i = c.min(-1)
        hats.append(levels[i]); costs.append(m)
    return torch.cat(hats), torch.cat(costs)


def region_report(a_ref, hats, horizon, floor):
    """Retention error split by whether the true decay lies below the 1-term grid floor (1 - 2^-k_min)."""
    out = {}
    for name, mask in (("below_floor", a_ref < floor), ("at_or_above_floor", a_ref >= floor)):
        n = int(mask.sum().item())
        out[name] = {"count": n, "fraction": n / a_ref.numel()}
        if n:
            for m, ah in hats.items():
                r = retention_error_report(a_ref[mask], ah[mask], horizon)
                out[name][m] = {"retention_rmse": r["retention_rmse"], "retention_max_abs": r["retention_max_abs"]}
    return out


@torch.no_grad()
def main():
    a = get_args()
    from datasets import load_dataset
    if not torch.cuda.is_available():
        raise RuntimeError("Decay calibration requires CUDA to run the pretrained Mamba model.")
    tok = get_tokenizer()
    model, _ = load_pretrained(a.pretrained, QuantConfig(decay="exp"))
    model = model.cuda().eval()
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if a.dtype == "auto" else (torch.bfloat16 if a.dtype == "bf16" else torch.float16)
    if a.max_coefficients < 1:
        raise ValueError("--max-coefficients must be positive")
    ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True)
    ds = ds.filter(lambda ex, idx: idx % a.val_modulo == 0, with_indices=True)
    grid = exponent_grid(a.k_min, a.k_max, 1.0, device="cuda")
    chunks, buf, seen = [], [], 0
    for ex in ds:
        txt = ex.get("text", "")
        if not isinstance(txt, str) or not txt.strip():
            continue
        buf.extend(tok(txt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= a.seq_len + 1:
            ids = torch.tensor(buf[:a.seq_len], device="cuda", dtype=torch.long)[None]
            buf = buf[a.seq_len:]
            # Capture continuous decay coefficients directly from each layer.
            model.set_track_decay(True)
            with torch.autocast("cuda", dtype=amp_dtype):
                model(ids)
            batch_values = [layer.mixer.last_decay_ref.detach().float().reshape(-1).cpu()
                            for layer in model.backbone.layers]
            batch_values = torch.cat(batch_values)
            remaining = max(0, a.max_coefficients - sum(v.numel() for v in chunks))
            if remaining:
                if batch_values.numel() > remaining:
                    stride = max(1, batch_values.numel() // remaining)
                    batch_values = batch_values[::stride][:remaining]
                chunks.append(batch_values)
            model.set_track_decay(False)
            seen += a.seq_len
            if seen >= a.tokens or sum(v.numel() for v in chunks) >= a.max_coefficients:
                break
        if seen >= a.tokens or sum(v.numel() for v in chunks) >= a.max_coefficients:
            break
    if not chunks:
        raise RuntimeError("No validation tokens were produced; check dataset/config and split.")
    a_ref = torch.cat(chunks).clamp(1e-8, 1 - 1e-8)
    # Move only the coefficient vector to CUDA for vectorized candidate scoring.
    a_ref = a_ref.cuda()
    k_h, cost_h, k_near, cost_near = horizon_aware_exponent(a_ref, grid, a.horizon, a.weight)
    a_near = 1 - torch.exp2(-k_near)
    a_h = 1 - torch.exp2(-k_h)
    hats = {"nearest": a_near, "harpq": a_h}
    dy = None
    if a.dyadic_terms > 0:
        levels = dyadic_levels(a.k_min, a.k_max, a.dyadic_terms, device="cuda")
        a_d, cost_d = dyadic_select(a_ref, levels, a.horizon, a.weight)
        hats["dyadic"] = a_d
        dy = {"terms": a.dyadic_terms, "n_levels": int(levels.numel()),
              "retention": retention_error_report(a_ref, a_d, a.horizon),
              "objective_cost_mean": cost_d.mean().item(),
              "level_usage_top10": [[round(float(levels[i]), 6), int(c)] for c, i in zip(*torch.sort(torch.bincount(
                  torch.bucketize(a_d, levels).clamp_max(levels.numel() - 1), minlength=levels.numel()), descending=True))][:10]}
    out = {
        "source": {"model": a.pretrained, "dataset": a.dataset, "subset": a.subset,
                   "validation_rule": f"document_index % {a.val_modulo} == 0", "tokens_seen": seen},
        "method": {"proposed_name": "HARPQ", "objective": "weighted finite-horizon stationary retention-curve MSE",
                   "horizon": a.horizon, "lag_weight": a.weight, "k_grid": [a.k_min, a.k_max],
                   "calibration_dtype": str(amp_dtype)},
        "coefficients": int(a_ref.numel()),
        "fraction_exponents_changed_vs_nearest": (k_h != k_near).float().mean().item(),
        "nearest_one_step": retention_error_report(a_ref, a_near, a.horizon),
        "horizon_aware": retention_error_report(a_ref, a_h, a.horizon),
        "objective_cost": {"nearest_mean": cost_near.mean().item(), "horizon_mean": cost_h.mean().item(),
                           "relative_reduction": (1 - cost_h.mean() / cost_near.mean().clamp_min(1e-12)).item()},
        "k_histogram_nearest": torch.bincount(k_near.long() - a.k_min, minlength=a.k_max-a.k_min+1).cpu().tolist(),
        "k_histogram_horizon": torch.bincount(k_h.long() - a.k_min, minlength=a.k_max-a.k_min+1).cpu().tolist(),
        "dyadic": dy,
        "by_region": region_report(a_ref, hats, a.horizon, 1 - 2.0 ** -a.k_min),
        "warning": "Calibration diagnostics only. This does not establish perplexity, speed, energy, or global novelty.",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

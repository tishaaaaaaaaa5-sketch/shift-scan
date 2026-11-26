"""Rank Mamba-2 blocks for mixed-precision protection using held-out validation CE."""
import argparse
import itertools
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_core import QuantConfig, TernaryLinear, get_tokenizer, load_pretrained
from train_qat import batches, sequence_stream


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="state-spaces/mamba2-130m")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--val-seqs", type=int, default=64)
    p.add_argument("--val-docs", type=int, default=20000)
    p.add_argument("--val-modulo", type=int, default=100)
    p.add_argument("--num-protect", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.inference_mode()
def validation_ce(model, val_batches, dtype):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in val_batches:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_tokens += y.numel()
    return total_loss / max(1, total_tokens)


def as_full_precision(module):
    if not isinstance(module, TernaryLinear):
        raise TypeError("Expected TernaryLinear projection")
    layer = nn.Linear(module.in_features, module.out_features, bias=False,
                      device=module.weight.device, dtype=module.weight.dtype)
    layer.weight.data.copy_(module.weight.data)
    return layer


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Layer sensitivity calibration requires a CUDA GPU.")
    if a.num_protect < 1:
        raise ValueError("--num-protect must be at least 1")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dtype = torch.float16 if a.dtype == "fp16" else torch.bfloat16

    tok = get_tokenizer()
    val = list(itertools.islice(
        batches(sequence_stream(tok, a.dataset, a.subset, a.seq_len,
                                role="validation", take_docs=a.val_docs,
                                seed=a.seed, val_modulo=a.val_modulo), a.micro_bs),
        max(1, a.val_seqs // a.micro_bs)))
    if not val:
        raise RuntimeError("Validation stream is empty; verify dataset access and split settings.")

    qcfg = QuantConfig(decay="exp", ternary=True, group_size=128,
                       activation_aware=False, fp_layers="")
    model, mcfg = load_pretrained(a.pretrained, qcfg)
    model = model.cuda().eval()
    if a.num_protect > mcfg.n_layer:
        raise ValueError(f"num_protect={a.num_protect} exceeds n_layer={mcfg.n_layer}")

    all_ternary_ce = validation_ce(model, val, dtype)
    scores = []
    for idx, block in enumerate(model.backbone.layers):
        mixer = block.mixer
        old_in, old_out = mixer.in_proj, mixer.out_proj
        if not isinstance(old_in, TernaryLinear) or not isinstance(old_out, TernaryLinear):
            raise RuntimeError(f"Block {idx} does not have two ternary projections; check quantization config.")
        mixer.in_proj = as_full_precision(old_in)
        mixer.out_proj = as_full_precision(old_out)
        protected_ce = validation_ce(model, val, dtype)
        mixer.in_proj, mixer.out_proj = old_in, old_out
        scores.append({
            "layer": idx,
            "all_ternary_validation_ce": all_ternary_ce,
            "single_layer_fp_validation_ce": protected_ce,
            "protection_gain_nats_per_token": all_ternary_ce - protected_ce,
        })
        print(f"layer={idx:02d} protection_gain={scores[-1]['protection_gain_nats_per_token']:+.6f}")

    ranked = sorted(scores, key=lambda row: row["protection_gain_nats_per_token"], reverse=True)
    sensitivity_layers = [row["layer"] for row in ranked[:a.num_protect]]
    random_layers = sorted(random.Random(a.seed).sample(range(mcfg.n_layer), a.num_protect))
    result = {
        "method": "single-block full-precision restoration over all-ternary model",
        "metric": "held-out validation cross-entropy (nats/token)",
        "pretrained_model": a.pretrained,
        "dataset": a.dataset,
        "dataset_config": a.subset,
        "validation_rule": f"document_index % {a.val_modulo} == 0",
        "validation_sequences": sum(x.shape[0] for x, _ in val),
        "seq_len": a.seq_len,
        "seed": a.seed,
        "dtype": a.dtype,
        "num_protected_blocks": a.num_protect,
        "all_ternary_validation_ce": all_ternary_ce,
        "sensitivity_layers": sensitivity_layers,
        "random_control_layers": random_layers,
        "scores": scores,
        "warning": "Selection uses the validation shard for calibration. Do not report this shard as an independent test set.",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Sensitivity-selected blocks:", sensitivity_layers)
    print("Parameter-matched random blocks:", random_layers)
    print("Wrote", a.out)


if __name__ == "__main__":
    main()

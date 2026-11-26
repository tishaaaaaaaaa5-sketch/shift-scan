"""
train_qat.py -- convert a pretrained Mamba-2 into a shift-decay / ternary student by distillation (QAT from pretrained).

Example (A100/H100, ~100M tokens):
  python train_qat.py --name D_shift_tern --decay shift --ternary 1 --tokens 100e6
Ablation grid (sensitivity study): change --decay / --ternary / --fp-rows / --gain / --k-step, keep everything else fixed.
"""
import argparse, itertools, json, math, os, time, random, platform
from importlib.metadata import PackageNotFoundError, version
import numpy as np
import torch
import torch.nn.functional as F

from ssm_core import (QuantConfig, load_pretrained, Mamba2LM, ModelConfig, get_tokenizer,
                      save_checkpoint, size_report)


def boolean(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--pretrained", default="state-spaces/mamba2-130m")
    p.add_argument("--decay", default="shift", choices=["exp", "shift"])
    p.add_argument("--gain", default="dt", choices=["dt", "leak"])
    p.add_argument("--k-min", type=float, default=1.0)
    p.add_argument("--k-max", type=float, default=14.0)
    p.add_argument("--k-step", type=float, default=1.0)
    p.add_argument("--decay-quantizer", default="nearest", choices=["nearest", "horizon", "dyadic"],
                   help="nearest one-step leak quantizer or horizon-aware retention-curve objective")
    p.add_argument("--decay-horizon", type=int, default=64,
                   help="lag horizon used by the horizon-aware decay quantizer")
    p.add_argument("--dyadic-terms", type=int, default=2, help="shift terms per decay when --decay-quantizer dyadic")
    p.add_argument("--decay-lag-weight", default="uniform", choices=["uniform", "log", "early", "late"])
    p.add_argument("--decay-group-size", type=int, default=1,
                   help="share decay exponent across this many adjacent heads (must divide nheads)")
    p.add_argument("--activation-aware", type=boolean, default=False,
                   help="use EMA input-channel second moments for ternary group scales")
    p.add_argument("--ternary", type=boolean, default=True)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--fp-rows", default="none", choices=["none", "dt", "bc", "dt+bc"])
    p.add_argument("--quantize-out-proj", type=boolean, default=True)
    p.add_argument("--act-bits", type=int, default=0)
    p.add_argument("--tokens", type=float, default=100e6)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--micro-bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--anneal-steps", type=int, default=0, help="ramp quantization strength 0->1 over N steps (0 = off)")
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0)
    p.add_argument("--state-weight", type=float, default=0.05,
                   help="normalized final recurrent-state distillation loss; 0 disables it")
    p.add_argument("--state-loss-type", default="hybrid", choices=["mse", "cosine", "hybrid"],
                   help="state KD: normalized MSE, cosine geometry, or their hybrid with log-RMS matching")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--val-seqs", type=int, default=64)
    p.add_argument("--val-docs", type=int, default=20000)
    p.add_argument("--val-modulo", type=int, default=100,
                   help="deterministic document split: index %% modulo == 0 is validation; other docs train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", default=None, help="resume model/optimizer/scaler/RNG state from a previous last.pt")
    p.add_argument("--wandb", type=boolean, default=False)
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def sequence_stream(tok, dataset, subset, seq_len, role="train", take_docs=None, seed=0, val_modulo=100):
    """Stream REAL FineWeb-Edu documents with a deterministic, disjoint train/validation split.

    Document indices divisible by val_modulo are held out for validation. Training
    documents are filtered before buffer shuffling, so no validation document enters
    the student/teacher QAT stream.
    """
    from datasets import load_dataset
    if val_modulo < 2:
        raise ValueError("val_modulo must be >= 2 to reserve a held-out shard")
    ds = load_dataset(dataset, name=subset, split="train", streaming=True)
    if role == "validation":
        ds = ds.filter(lambda ex, idx: idx % val_modulo == 0, with_indices=True)
    elif role == "train":
        ds = ds.filter(lambda ex, idx: idx % val_modulo != 0, with_indices=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    else:
        raise ValueError("role must be 'train' or 'validation'")
    if take_docs is not None:
        ds = ds.take(take_docs)
    eos = tok.eos_token_id
    buf = []
    for ex in ds:
        text = ex.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if ids:
            buf.extend(ids + [eos])
        while len(buf) >= seq_len + 1:
            yield torch.tensor(buf[: seq_len + 1], dtype=torch.long)
            buf = buf[seq_len:]


def batches(stream, bs):
    chunk = []
    for s in stream:
        chunk.append(s)
        if len(chunk) == bs:
            t = torch.stack(chunk)
            chunk = []
            yield t[:, :-1], t[:, 1:]


# ----------------------------------------------------------------------------- losses
def kd_loss(s_logits, t_logits, T=1.0):
    s = F.log_softmax(s_logits.float() / T, -1)
    t = F.log_softmax(t_logits.float() / T, -1)
    return (t.exp() * (t - s)).sum(-1).mean() * T * T


def recurrent_state_loss(student, teacher, mode="hybrid"):
    """Layer-averaged final-state distillation with scale and geometry controls."""
    s_mixers = [b.mixer for b in student.backbone.layers]
    t_mixers = [b.mixer for b in teacher.backbone.layers]
    terms = []
    for sm, tm in zip(s_mixers, t_mixers):
        if sm.last_state is None or tm.last_state is None:
            continue
        target = tm.last_state.detach().float()
        pred = sm.last_state.float()
        denom = target.square().mean().detach().clamp_min(1e-6)
        mse = (pred - target).square().mean() / denom
        if mode == "mse":
            term = mse
        else:
            pvec, tvec = pred.flatten(-2), target.flatten(-2)
            cosine = 1.0 - F.cosine_similarity(pvec, tvec, dim=-1, eps=1e-8).mean()
            if mode == "cosine":
                term = cosine
            elif mode == "hybrid":
                p_rms = pred.square().mean().sqrt().clamp_min(1e-8)
                t_rms = target.square().mean().sqrt().clamp_min(1e-8)
                magnitude = (p_rms.log() - t_rms.log()).square()
                term = mse + 0.25 * cosine + 0.1 * magnitude
            else:
                raise ValueError("state loss mode must be mse|cosine|hybrid")
        terms.append(term)
    if not terms:
        return next(student.parameters()).new_zeros(())
    return torch.stack(terms).mean()


@torch.no_grad()
def evaluate(student, teacher, val, amp, dev, state_loss_type="hybrid"):
    student.eval()
    out = {"s_ce": 0.0, "t_ce": 0.0, "kd": 0.0, "state_mse": 0.0}
    student.set_track_state(True)
    teacher.set_track_state(True)
    for x, y in val:
        x, y = x.to(dev), y.to(dev)
        with amp():
            sl, tl_ = student(x), teacher(x)
        out["s_ce"] += F.cross_entropy(sl.float().flatten(0, 1), y.flatten()).item()
        out["t_ce"] += F.cross_entropy(tl_.float().flatten(0, 1), y.flatten()).item()
        out["kd"] += kd_loss(sl, tl_).item()
        out["state_mse"] += recurrent_state_loss(student, teacher, state_loss_type).item()
    n = len(val)
    student.train()
    student.set_track_state(False)
    teacher.set_track_state(False)
    res = {k: v / n for k, v in out.items()}
    res["student_ppl"], res["teacher_ppl"] = math.exp(res["s_ce"]), math.exp(res["t_ce"])
    return res


@torch.no_grad()
def k_stats(student, x, amp):
    """Distribution of learned integer shifts per layer (only meaningful for decay='shift')."""
    if student.qcfg.decay != "shift":
        return {}
    student.set_track_k(True)
    student.eval()
    with amp():
        student(x)
    student.train()
    student.set_track_k(False)
    stats = {}
    mixers = [blk.mixer for blk in student.backbone.layers]
    allk = torch.cat([m.last_k.flatten() for m in mixers]).cpu()
    stats["k_mean"] = allk.mean().item()
    stats["frac_at_kmin"] = (allk <= student.qcfg.k_min + 1e-6).float().mean().item()
    stats["frac_at_kmax"] = (allk >= student.qcfg.k_max - 1e-6).float().mean().item()
    bins = int(round((student.qcfg.k_max - student.qcfg.k_min) / student.qcfg.k_step)) + 1
    stats["k_std"] = allk.std(unbiased=False).item()
    stats["k_hist"] = torch.histc(
        allk, bins=bins,
        min=student.qcfg.k_min - 0.5 * student.qcfg.k_step,
        max=student.qcfg.k_max + 0.5 * student.qcfg.k_step,
    ).tolist()
    return stats


# ----------------------------------------------------------------------------- main
def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("QAT training requires a CUDA GPU; run python tests.py for CPU-only checks.")
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out_dir = os.path.join(a.out, a.name)
    os.makedirs(out_dir, exist_ok=True)
    run_config = vars(a).copy()
    run_config["data_protocol"] = {
        "dataset": a.dataset, "subset": a.subset, "split": "train streaming",
        "validation_rule": f"document_index % {a.val_modulo} == 0",
        "training_rule": f"document_index % {a.val_modulo} != 0",
        "synthetic_training_data": False,
    }
    json.dump(run_config, open(os.path.join(out_dir, "args.json"), "w"), indent=1)

    qcfg = QuantConfig(decay=a.decay, gain=a.gain, k_min=a.k_min, k_max=a.k_max, k_step=a.k_step,
                       ternary=a.ternary, group_size=a.group_size, fp_rows=a.fp_rows,
                       quantize_out_proj=a.quantize_out_proj, act_bits=a.act_bits,
                       decay_group_size=a.decay_group_size, activation_aware=a.activation_aware,
                       decay_quantizer=a.decay_quantizer, decay_horizon=a.decay_horizon,
                       decay_lag_weight=a.decay_lag_weight, dyadic_terms=a.dyadic_terms)
    student, mcfg = load_pretrained(a.pretrained, qcfg)
    teacher, _ = load_pretrained(a.pretrained, QuantConfig())                # exp decay, fp -> the teacher
    student, teacher = student.to(dev), teacher.to(dev).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    size_info = size_report(student)
    print("student size report:", json.dumps(size_info, indent=1))
    json.dump(size_info, open(os.path.join(out_dir, "size_report.json"), "w"), indent=1)
    def pkg_version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
        "transformers": pkg_version("transformers"),
        "datasets": pkg_version("datasets"),
        "triton": pkg_version("triton"),
        "lm_eval": pkg_version("lm-eval"),
        "pretrained_model_id": a.pretrained,
        "tokenizer_id": "EleutherAI/gpt-neox-20b",
        "dataset_id": a.dataset,
        "dataset_config": a.subset,
        "decay_quantizer": a.decay_quantizer,
        "dyadic_terms": a.dyadic_terms,
        "decay_horizon": a.decay_horizon,
        "decay_lag_weight": a.decay_lag_weight,
        "state_loss_type": a.state_loss_type,
        "seed": a.seed,
    }
    json.dump(manifest, open(os.path.join(out_dir, "run_manifest.json"), "w"), indent=1)

    amp_dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    amp = lambda: torch.autocast("cuda", dtype=amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(a.dtype == "fp16"))
    start_step, seen = 0, 0
    opt_state_pending = None
    if a.resume:
        resume_ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        if resume_ck.get("qcfg") != vars(qcfg) or resume_ck.get("mcfg") != vars(mcfg):
            raise ValueError("Resume checkpoint architecture/quantization config does not match this run's CLI settings.")
        student.load_state_dict(resume_ck["state_dict"], strict=True)
        extra = resume_ck.get("extra", {})
        if "optimizer_state" in extra:
            opt_state_pending = extra["optimizer_state"]
        else:
            opt_state_pending = None
        if "scaler_state" in extra:
            scaler.load_state_dict(extra["scaler_state"])
        start_step = int(extra.get("step", 0))
        seen = int(extra.get("tokens", 0))
        if "torch_rng_state" in extra:
            torch.set_rng_state(extra["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in extra:
            torch.cuda.set_rng_state_all(extra["cuda_rng_state_all"])
        if "numpy_rng_state" in extra:
            np.random.set_state(extra["numpy_rng_state"])
        if "python_rng_state" in extra:
            random.setstate(extra["python_rng_state"])

    decay, no_decay = [], []
    for n_, p_ in student.named_parameters():
        (decay if p_.ndim >= 2 and "embedding" not in n_ and "conv1d" not in n_ else no_decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    if a.resume and opt_state_pending is not None:
        opt.load_state_dict(opt_state_pending)

    tok = get_tokenizer()
    val = list(itertools.islice(
        batches(sequence_stream(tok, a.dataset, a.subset, a.seq_len, role="validation",
                                take_docs=a.val_docs, seed=a.seed, val_modulo=a.val_modulo), a.micro_bs),
        max(1, a.val_seqs // a.micro_bs)))
    if not val:
        raise RuntimeError("Validation stream produced no batches. Check dataset/config/tokenizer access.")
    train_it = batches(sequence_stream(tok, a.dataset, a.subset, a.seq_len, role="train",
                                       seed=a.seed, val_modulo=a.val_modulo), a.micro_bs)
    if start_step:
        train_it = itertools.islice(train_it, start_step * a.accum, None)

    tok_per_step = a.micro_bs * a.accum * a.seq_len
    total_steps = int(a.tokens // tok_per_step)
    print(f"{total_steps} steps x {tok_per_step} tokens/step = {total_steps*tok_per_step/1e6:.1f}M tokens")

    base = evaluate(student, teacher, val, amp, dev, a.state_loss_type)
    print(f"step {start_step} (initial/resume evaluation):", base)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    log.write(json.dumps({"step": start_step, "tokens": seen, **base,
                          **k_stats(student, val[0][0].to(dev), amp)}) + "\n"); log.flush()
    if a.wandb:
        import wandb; wandb.init(project="shift-scan", name=a.name, config=vars(a))

    t0 = time.time()
    student.train()
    if start_step >= total_steps:
        raise RuntimeError(f"Resume checkpoint step {start_step} already meets/exceeds requested total steps {total_steps}.")
    for step in range(start_step + 1, total_steps + 1):
        lr_f = step / a.warmup if step <= a.warmup else a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (
            1 + math.cos(math.pi * (step - a.warmup) / max(1, total_steps - a.warmup)))
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, lr_f)
        if a.anneal_steps:
            student.set_quant_strength(min(1.0, step / a.anneal_steps))
        tot_kd = 0.0
        tot_state = 0.0
        for _ in range(a.accum):
            try:
                x, y = next(train_it)
            except StopIteration as exc:
                raise RuntimeError(
                    "The streamed training split ended before the requested token budget. "
                    "Reduce --tokens, increase the dataset sample/config, or use a larger stream."
                ) from exc
            x, y = x.to(dev), y.to(dev)
            if a.state_weight:
                student.set_track_state(True)
                teacher.set_track_state(True)
            with torch.no_grad(), amp():
                tl_ = teacher(x)
            with amp():
                sl = student(x)
                loss = a.kd_weight * kd_loss(sl, tl_, a.temp)
                state_loss = sl.new_zeros(())
                if a.state_weight:
                    state_loss = recurrent_state_loss(student, teacher, a.state_loss_type)
                    loss = loss + a.state_weight * state_loss
                if a.ce_weight:
                    loss = loss + a.ce_weight * F.cross_entropy(sl.float().flatten(0, 1), y.flatten())
            if a.state_weight:
                student.set_track_state(False)
                teacher.set_track_state(False)
            scaler.scale(loss / a.accum).backward()
            tot_kd += loss.item() / a.accum
            tot_state += state_loss.detach().item() / a.accum
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        seen += tok_per_step

        if step % a.log_every == 0:
            rec = {"step": step, "tokens": seen, "loss": tot_kd, "state_loss": tot_state,
                   "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "tok_s": seen / (time.time() - t0)}
            print(rec); log.write(json.dumps(rec) + "\n"); log.flush()
            if a.wandb: wandb.log(rec)
        if step % a.eval_every == 0 or step == total_steps:
            ev = {"step": step, **evaluate(student, teacher, val, amp, dev, a.state_loss_type), **k_stats(student, val[0][0].to(dev), amp)}
            print("EVAL", {k: v for k, v in ev.items() if k != "k_hist"})
            log.write(json.dumps(ev) + "\n"); log.flush()
            if a.wandb: wandb.log({k: v for k, v in ev.items() if k != "k_hist"})
        if step % a.save_every == 0 or step == total_steps:
            save_checkpoint(student, os.path.join(out_dir, "last.pt"), {
                "step": step, "tokens": seen, "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            })
    print("done ->", out_dir)


if __name__ == "__main__":
    main()

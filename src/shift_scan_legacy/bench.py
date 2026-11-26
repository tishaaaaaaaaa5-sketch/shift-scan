"""
bench.py -- decode latency / memory / energy.

  python bench.py --ckpt runs/D_shift_tern/last.pt --out results/bench_D.json            # converted model vs teacher
  python bench.py --pretrained-only --out results/bench_teacher.json

Reports:
  * measured: per-token decode latency (CUDA events), peak GPU memory, state bytes, optional GPU energy via NVML
  * analytic: weight/state/KV-cache memory vs context length, and per-token energy from published 45nm op energies
HONEST SCOPE: GPUs have no ternary ALUs. The packed model's measured gain on GPU comes from lower weight traffic (+ the
fused scan-step kernel); the add-only energy numbers are an ANALYTIC estimate, not a measurement.
"""
import argparse, json, os, threading, time, statistics
import torch

from ssm_core import load_pretrained, load_checkpoint, QuantConfig, size_report, TernaryLinear

# ---- 45nm energy per operation, pJ.  Horowitz, "Computing's energy problem", ISSCC 2014 (VERIFY before citing).
E = dict(int8_add=0.03, int16_add=0.05, int32_add=0.1, fp16_add=0.4, fp32_add=0.9,
         int8_mult=0.2, int32_mult=3.1, fp16_mult=1.1, fp32_mult=3.7,
         dram_per_byte=1300.0 / 8)         # ~1.3 nJ per 64-bit DRAM access (lower end of the 1.3-2.6 nJ range)


def analytic(mcfg, n_tern, n_total, act="fp16"):
    """Illustrative operation-energy estimate; never a substitute for device telemetry."""
    if n_tern <= 0:
        return {"proj_params": 0, "note": "no ternary projection parameters in this checkpoint"}
    mac_fp16 = n_tern * (E["fp16_mult"] + E["fp16_add"])
    add_cost = E["fp16_add"] if act == "fp16" else E["int8_add"]
    tern_ops = n_tern * (2 / 3) * add_cost                                # ~1/3 of ternary weights are 0 -> skipped
    dram_fp16 = n_tern * 2 * E["dram_per_byte"]
    dram_tern = n_tern * (2 / 8) * E["dram_per_byte"]                     # 2 bits per weight
    return {
        "proj_params": n_tern,
        "fp16_compute_uJ": mac_fp16 / 1e6, "ternary_compute_uJ": tern_ops / 1e6,
        "fp16_weight_dram_uJ": dram_fp16 / 1e6, "ternary_weight_dram_uJ": dram_tern / 1e6,
        "compute_reduction_x": mac_fp16 / tern_ops,
        "total_reduction_x": (mac_fp16 + dram_fp16) / (tern_ops + dram_tern),
    }


def memory_vs_context(mcfg, contexts=(1024, 4096, 16384, 65536, 262144)):
    """Analytic: SSM state is constant; a hypothetical Transformer of the same width/depth stores 2*d_model*layers fp16 per token."""
    ssm = mcfg.n_layer * (mcfg.nheads * mcfg.headdim * mcfg.d_state * 4 + mcfg.conv_dim * mcfg.d_conv * 4)
    kv = lambda n: n * 2 * mcfg.d_model * mcfg.n_layer * 2
    return {"ssm_state_MB": ssm / 2**20, "kv_cache_MB": {n: kv(n) / 2**20 for n in contexts}}


@torch.no_grad()
def bench_decode(model, batch, steps=200, warm=20, trials=3, nvml=False, dev="cuda"):
    model.eval()
    ids = torch.randint(0, model.mcfg.vocab, (batch,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    power, stop = [], threading.Event()
    if nvml:
        import pynvml
        pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        def poll():
            while not stop.is_set():
                power.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0); time.sleep(0.02)
        th = threading.Thread(target=poll); th.start()
    trial_ms = []
    overflow_seen = False
    try:
        for _ in range(trials):
            conv, ssm = model.init_states(batch, dev)
            for _ in range(warm):
                model.step(ids, conv, ssm)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(steps):
                model.step(ids, conv, ssm)
            e.record(); torch.cuda.synchronize()
            trial_ms.append(s.elapsed_time(e) / steps)
            trial_flags = [blk.mixer.integer_overflow_seen for blk in model.backbone.layers
                           if blk.mixer.integer_overflow_seen is not None]
            if trial_flags:
                overflow_seen = overflow_seen or any(bool(v.any().item()) for v in trial_flags)
    finally:
        if nvml:
            stop.set(); th.join()
    ms = statistics.median(trial_ms)
    out = {"batch": batch, "trials": trials, "steps_per_trial": steps,
           "ms_per_token_step_median": ms, "ms_per_token_step_mean": statistics.mean(trial_ms),
           "ms_per_token_step_std": statistics.pstdev(trial_ms), "tok_per_s": batch * 1000 / ms,
           "peak_mem_MB": torch.cuda.max_memory_allocated() / 2**20}
    if model.qcfg.integer_state:
        out["integer_state_overflow_seen"] = overflow_seen
    if nvml and power:
        w = sum(power) / len(power)
        out.update(avg_power_W=w, mJ_per_token=w * (ms / 1000) / batch * 1000)
    return out


@torch.no_grad()
def bench_prefill(model, ids, warm=2, repeats=5):
    """Compute-only prefill microbenchmark; ids are deterministic random token IDs, not eval data."""
    model.eval()
    ids = ids.to(next(model.parameters()).device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warm):
        model(ids)
    torch.cuda.synchronize()
    trial_ms = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); model(ids); end.record(); torch.cuda.synchronize()
        trial_ms.append(start.elapsed_time(end))
    med = statistics.median(trial_ms)
    tokens = ids.numel()
    return {"batch": ids.shape[0], "sequence_length": ids.shape[1], "trials": repeats,
            "ms_per_batch_median": med, "ms_per_batch_mean": statistics.mean(trial_ms),
            "ms_per_batch_std": statistics.pstdev(trial_ms), "tok_per_s": tokens * 1000 / med,
            "peak_mem_MB": torch.cuda.max_memory_allocated() / 2**20,
            "input_note": "random token IDs for compute benchmarking only; not language-model evaluation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pretrained", default="state-spaces/mamba2-130m")
    ap.add_argument("--pretrained-only", action="store_true")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--decode-steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prefill-lengths", default="1024,2048,4096")
    ap.add_argument("--prefill-batch", type=int, default=1)
    ap.add_argument("--prefill-repeats", type=int, default=5)
    ap.add_argument("--nvml", action="store_true")
    ap.add_argument("--integer-state", action="store_true",
                    help="also benchmark int32 fixed-point recurrent state with actual shift decay (decode only)")
    ap.add_argument("--state-frac-bits", type=int, default=20)
    ap.add_argument("--check-integer-overflow", action="store_true",
                    help="debug-only synchronous overflow check; do not use for latency measurements")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Decode benchmarking requires a CUDA GPU.")
    if not a.pretrained_only and not a.ckpt:
        ap.error("--ckpt is required unless --pretrained-only is set")
    if min(a.decode_steps, a.trials, a.prefill_repeats, a.prefill_batch) < 1 or a.warmup < 0:
        ap.error("decode steps, trials, prefill repeats and prefill batch must be positive; warmup must be nonnegative")
    dev = "cuda"
    res = {"gpu": torch.cuda.get_device_name(0),
           "gpu_capability": torch.cuda.get_device_capability(0),
           "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "benchmark_note": "Teacher and student non-ternary parameters are both FP16; report actual measured results only for this GPU/runtime."}

    teacher, mcfg = load_pretrained(a.pretrained, QuantConfig())
    teacher = teacher.to(dev).half()
    res["memory_vs_context"] = memory_vs_context(mcfg)
    res["benchmark_config"] = {"decode_steps": a.decode_steps, "warmup": a.warmup,
                               "trials": a.trials, "prefill_batch": a.prefill_batch,
                               "prefill_lengths": [int(v) for v in a.prefill_lengths.split(",")],
                               "prefill_repeats": a.prefill_repeats}
    gen = torch.Generator(device=dev).manual_seed(2026)
    prefill_inputs = {int(length): torch.randint(0, mcfg.vocab, (a.prefill_batch, int(length)),
                                                  device=dev, generator=gen)
                      for length in a.prefill_lengths.split(",")}
    res["teacher_decode"] = [bench_decode(teacher, int(b), steps=a.decode_steps, warm=a.warmup,
                                          trials=a.trials, nvml=a.nvml) for b in a.batches.split(",")]
    res["teacher_prefill"] = [bench_prefill(teacher, ids, repeats=a.prefill_repeats)
                              for ids in prefill_inputs.values()]
    res["teacher_size"] = size_report(teacher)
    del teacher; torch.cuda.empty_cache()

    if not a.pretrained_only:
        from kernels import convert_to_packed, HAS_TRITON
        model, ck = load_checkpoint(a.ckpt, "cpu")
        res["student_size"] = size_report(model)
        n_tern = sum(m.weight.numel() for m in model.modules() if isinstance(m, TernaryLinear))
        res["analytic_energy_per_token"] = {"A16": analytic(mcfg, n_tern, 0, "fp16"), "A8": analytic(mcfg, n_tern, 0, "int8")}
        # (a) fake-quant/dequantized PyTorch baseline: this is not expected to speed up.
        model = model.to(dev).half()
        res["student_simulated_fp_decode"] = [bench_decode(model, int(b), steps=a.decode_steps, warm=a.warmup,
                                                            trials=a.trials, nvml=a.nvml) for b in a.batches.split(",")]
        res["student_simulated_fp_prefill"] = [bench_prefill(model, ids, repeats=a.prefill_repeats)
                                               for ids in prefill_inputs.values()]
        if a.integer_state:
            if model.qcfg.decay != "shift":
                raise ValueError("--integer-state requires a checkpoint trained with --decay shift")
            if abs(model.qcfg.k_step - 1.0) > 1e-8 or abs(model.qcfg.k_min-round(model.qcfg.k_min)) > 1e-8 or abs(model.qcfg.k_max-round(model.qcfg.k_max)) > 1e-8 or model.qcfg.k_max > 30:
                raise ValueError("--integer-state requires integer exponents, --k-step 1 and --k-max<=30")
            if not 8 <= a.state_frac_bits <= 30:
                raise ValueError("--state-frac-bits must be in [8,30]")
            model.qcfg.integer_state = True
            model.qcfg.state_frac_bits = a.state_frac_bits
            model.qcfg.check_integer_overflow = a.check_integer_overflow
            res["student_integer_state_torch_reference"] = [
                bench_decode(model, int(b), steps=a.decode_steps, warm=a.warmup,
                             trials=a.trials, nvml=a.nvml) for b in a.batches.split(",")]
            if HAS_TRITON:
                model = convert_to_packed(model, use_triton_scan=False, use_integer_scan=True)
                res["student_packed_triton_integer_state"] = [
                    bench_decode(model, int(b), steps=a.decode_steps, warm=a.warmup,
                                 trials=a.trials, nvml=a.nvml) for b in a.batches.split(",")]
                res["student_packed_triton_integer_state_prefill"] = [
                    bench_prefill(model, ids, repeats=a.prefill_repeats) for ids in prefill_inputs.values()]
        elif HAS_TRITON:
            model = convert_to_packed(model)
            res["student_packed_triton_float_state"] = [
                bench_decode(model, int(b), steps=a.decode_steps, warm=a.warmup,
                             trials=a.trials, nvml=a.nvml) for b in a.batches.split(",")]
            res["student_packed_triton_float_state_prefill"] = [
                bench_prefill(model, ids, repeats=a.prefill_repeats) for ids in prefill_inputs.values()]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

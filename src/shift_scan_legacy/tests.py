"""
tests.py -- run FIRST (CPU is fine, no downloads):  python tests.py
Deterministic unit tests (random tensors here are test fixtures, never training/eval data):
  1. chunked SSD scan == naive recurrence
  2. parallel forward == recurrent decode step
  3. integer-valued power-of-two exponents and STE gradients
  4. ternary values, packing round-trip and STE gradients
  5. activation-aware scales and group-shared decay
  6. fixed-point integer-shift reference and recurrent error-bound sanity check
Optional: --official compares logits with official mamba_ssm (CUDA + mamba_ssm required).
"""
import sys
import tempfile
import os
import torch
import torch.nn.functional as F
from ssm_core import (ModelConfig, QuantConfig, Mamba2LM, TernaryLinear, ssd_scan, pack_ternary, unpack_ternary,
                      size_report, save_checkpoint, load_checkpoint)
from fixedpoint_state import fixedpoint_shift_step
from theory import recurrent_error_bound, decay_grid_report
from retention import exponent_grid, horizon_aware_exponent, retention_error_report, dyadic_levels, dyadic_decay

torch.manual_seed(0)


def naive_scan(x, logdecay, B, C):
    b, l, h, p = x.shape
    n = B.shape[-1]
    s = torch.zeros(b, h, p, n, dtype=x.dtype)
    ys = []
    for t in range(l):
        s = torch.exp(logdecay[:, t])[:, :, None, None] * s + x[:, t][..., None] * B[:, t][:, None, None, :]
        ys.append((s * C[:, t][:, None, None, :]).sum(-1))
    return torch.stack(ys, 1), s


def test_scan():
    for l, chunk in [(64, 16), (50, 16), (37, 8), (130, 64)]:
        b, h, p, n = 2, 3, 4, 5
        x, B, C = torch.randn(b, l, h, p, dtype=torch.double), torch.randn(b, l, n, dtype=torch.double), torch.randn(b, l, n, dtype=torch.double)
        k = torch.randint(1, 12, (b, l, h)).double()
        ld = torch.log1p(-2.0 ** (-k))
        y1, s1 = ssd_scan(x, ld, B, C, chunk)
        y2, s2 = naive_scan(x, ld, B, C)
        assert torch.allclose(y1, y2, atol=1e-9) and torch.allclose(s1, s2, atol=1e-9), (l, chunk)
    print("[ok] chunked scan == naive recurrence")


def tiny_cfg():
    return ModelConfig(d_model=64, n_layer=2, d_inner=128, d_state=16, nheads=4, d_conv=4, vocab=100, chunk=8)


def init_like_pretrained(model):
    for m in model.modules():
        if hasattr(m, "A_log"):
            m.A_log.data = torch.log(torch.rand_like(m.A_log) * 15 + 1)
            dt = torch.exp(torch.rand_like(m.dt_bias) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3))) + torch.log(torch.tensor(1e-3)))
            m.dt_bias.data = dt + torch.log(-torch.expm1(-dt))


def test_parallel_vs_step(decay, ternary, decay_quantizer="nearest"):
    q = QuantConfig(decay=decay, ternary=ternary, group_size=32, k_max=12,
                    decay_quantizer=decay_quantizer, decay_horizon=32)
    model = Mamba2LM(tiny_cfg(), q).eval()
    init_like_pretrained(model)
    ids = torch.randint(0, 100, (2, 21))
    with torch.no_grad():
        full = model(ids)
        conv, ssm = model.init_states(2, "cpu")
        outs = [model.step(ids[:, t], conv, ssm) for t in range(ids.shape[1])]
    step = torch.stack(outs, 1)
    err = (full - step).abs().max().item()
    assert err < 1e-3, err
    print(f"[ok] parallel == recurrent step  (decay={decay}, quantizer={decay_quantizer}, ternary={ternary}, max err {err:.2e})")


def test_dyadic_decay():
    lv1, lv2 = dyadic_levels(1, 14, 1), dyadic_levels(1, 14, 2)
    assert len(lv1) == 14 and len(lv2) > 100, (len(lv1), len(lv2))
    assert torch.all(lv2 > 0) and torch.all(lv2 < 1) and torch.all(lv2[1:] > lv2[:-1])
    assert set(lv1.tolist()) <= set(lv2.tolist())                       # 2-term grid contains 1-term grid
    a = torch.rand(500) * 0.99 + 0.005
    for mode in ("uniform", "log"):
        c1 = dyadic_decay(a, lv1, 64, mode); c2 = dyadic_decay(a, lv2, 64, mode)
        e1 = retention_error_report(a, c1, 64)["retention_rmse"]; e2 = retention_error_report(a, c2, 64)["retention_rmse"]
        assert e2 < e1, (mode, e1, e2)                                  # denser grid cannot be worse (superset)
    print(f"[ok] dyadic decay: {len(lv1)} -> {len(lv2)} levels, retention RMSE {e1:.4f} -> {e2:.4f}")


def test_shift_properties():
    model = Mamba2LM(tiny_cfg(), QuantConfig(decay="shift", k_max=12))
    init_like_pretrained(model)
    mix = model.backbone.layers[0].mixer
    mix.track_k = True
    ids = torch.randint(0, 100, (2, 16))
    loss = model(ids).logsumexp(-1).mean()
    loss.backward()
    k = mix.last_k
    assert torch.all(k == k.round()) and k.min() >= 1 and k.max() <= 12
    a = 1 - 2.0 ** (-k)
    assert torch.all((a > 0) & (a < 1))
    assert mix.dt_bias.grad is not None and mix.A_log.grad is not None and mix.A_log.grad.abs().sum() > 0
    print(f"[ok] shift decay: integer k in [{int(k.min())},{int(k.max())}], 0<a<1, STE gradients reach dt_bias/A_log")


def test_ternary():
    lin = TernaryLinear(64, 32, group_size=16)
    q, gamma = lin.ternary_parts()
    assert set(q.unique().tolist()) <= {-1.0, 0.0, 1.0}
    pk = pack_ternary(q.to(torch.int8))
    assert torch.equal(unpack_ternary(pk, 64), q.to(torch.int8))
    x = torch.randn(3, 64, requires_grad=True)
    lin(x).sum().backward()
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0
    half_lin = TernaryLinear(64, 32, group_size=16).half()
    half_out = half_lin(x.half())
    assert half_out.dtype == torch.float16 and torch.isfinite(half_out).all()
    Wd = (q.view(32, 4, 16) * gamma.unsqueeze(-1)).view(32, 64)
    assert torch.allclose(lin(x), F.linear(x, Wd), atol=1e-5)
    print(f"[ok] ternary: {(q==0).float().mean():.0%} zeros, pack/unpack exact, STE gradient flows, forward == dequantized matmul")


def test_layerwise_mixed_precision():
    q = QuantConfig(ternary=True, group_size=32, fp_layers="1")
    model = Mamba2LM(tiny_cfg(), q)
    assert isinstance(model.backbone.layers[0].mixer.in_proj, TernaryLinear)
    assert isinstance(model.backbone.layers[0].mixer.out_proj, TernaryLinear)
    assert isinstance(model.backbone.layers[1].mixer.in_proj, torch.nn.Linear)
    assert not isinstance(model.backbone.layers[1].mixer.in_proj, TernaryLinear)
    assert isinstance(model.backbone.layers[1].mixer.out_proj, torch.nn.Linear)
    assert not isinstance(model.backbone.layers[1].mixer.out_proj, TernaryLinear)
    print("[ok] selected blocks retain full-precision input/output projections")


def test_group_decay():
    cfg = tiny_cfg()
    q = QuantConfig(decay="shift", decay_group_size=2, k_max=12)
    model = Mamba2LM(cfg, q).eval()
    init_like_pretrained(model)
    mix = model.backbone.layers[0].mixer
    ids = torch.randint(0, 100, (2, 9))
    model.set_track_k(True)
    with torch.no_grad():
        model(ids)
    k = mix.last_k
    assert torch.allclose(k[..., 0], k[..., 1])
    assert torch.allclose(k[..., 2], k[..., 3])
    assert torch.all(k == k.round())
    print("[ok] group-wise decay shares integer exponents across adjacent heads")


def test_fixedpoint_and_error_bound():
    torch.manual_seed(7)
    b, h, p, n = 2, 4, 8, 6
    frac = 20
    state = torch.randn(b, h, p, n) * 0.1
    state_q = torch.round(state * (1 << frac)).to(torch.int32)
    x = torch.randn(b, h, p) * 0.1
    B = torch.randn(b, n) * 0.1
    C = torch.randn(b, n)
    k = torch.randint(2, 10, (b, h)).float()
    gain = torch.rand(b, h)
    D = torch.randn(h)
    next_q, y = fixedpoint_shift_step(state_q, x, B, C, k, gain, D, frac_bits=frac)
    ref = state * (1 - torch.exp2(-k))[..., None, None] + (gain[..., None] * x)[..., None] * B[:, None, None, :]
    ref_y = (ref * C[:, None, None, :]).sum(-1) + D[None, :, None] * x
    assert (next_q.float() / (1 << frac) - ref).abs().max() < 2e-5
    assert (y - ref_y).abs().max() < 2e-4

    A = torch.tensor([0.9, 0.8, 0.95])
    Ah = torch.tensor([0.89, 0.81, 0.94])
    hs = torch.tensor([1.0, 0.8, 0.7])
    u = torch.tensor([0.2, 0.1, 0.3])
    uh = u + torch.tensor([0.01, -0.02, 0.01])
    bound = recurrent_error_bound(A, Ah, hs, u, uh)
    exact = []
    e = torch.tensor(0.0)
    h_ref = torch.tensor(0.0)
    h_approx = torch.tensor(0.0)
    for t in range(3):
        h_ref = A[t] * h_ref + u[t]
        h_approx = Ah[t] * h_approx + uh[t]
        e = (h_ref - h_approx).abs()
        exact.append(e)
    assert torch.all(torch.stack(exact) <= bound + 1e-6)
    print("[ok] fixed-point shift reference and recurrent error bound")


def test_activation_aware_ternary():
    lin = TernaryLinear(64, 32, group_size=16, activation_aware=True)
    lin.train()
    x = torch.randn(8, 64)
    x[:, :16] *= 8.0
    before = lin.act_ema.clone()
    y = lin(x)
    assert torch.isfinite(y).all()
    assert not torch.equal(before, lin.act_ema)
    q, gamma = lin.ternary_parts()
    assert set(q.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert gamma.shape == (32, 4)
    print("[ok] activation-aware ternary scales update from channel second moments")


def test_fp_rows():
    cfg = tiny_cfg()
    model = Mamba2LM(cfg, QuantConfig(ternary=True, group_size=32, fp_rows="dt"))
    lin = model.backbone.layers[0].mixer.in_proj
    assert lin.fp_rows[-cfg.nheads:].all() and not lin.fp_rows[:-cfg.nheads].any()
    x = torch.randn(2, cfg.d_model)
    y = lin(x)
    assert torch.allclose(y[:, -cfg.nheads:], F.linear(x, lin.weight)[:, -cfg.nheads:], atol=1e-5)
    print("[ok] fp_rows: dt rows stay full precision (sensitivity-study switch works)")


def test_pot_projection():
    dt = torch.rand(1000) * 0.099 + 0.001
    A = -(torch.rand(1000) * 15 + 1)
    a = torch.exp(dt * A)
    k = (-torch.log2(-torch.expm1(dt * A))).clamp(1, 14).round()
    a_q = 1 - 2.0 ** (-k)
    rel_leak = ((1 - a_q) / (1 - a)).clamp(min=1e-9)
    print(f"[info] PoT grid vs pretrained-like decays: mean |a-a_q| = {(a - a_q).abs().mean():.4f}; "
          f"leak ratio (a_q vs a) within [0.7,1.42] for {((rel_leak > 0.7) & (rel_leak < 1.42)).float().mean():.0%} of heads")


def test_horizon_aware_decay():
    torch.manual_seed(19)
    a = torch.rand(4096, dtype=torch.float32) * 0.998 + 0.001
    grid = exponent_grid(1, 14, 1.0)
    for weight_mode in ("uniform", "log"):
        kh, ch, kn, cn = horizon_aware_exponent(a, grid, horizon=128, weight_mode=weight_mode)
        assert torch.all(kh == kh.round())
        assert torch.all(ch <= cn + 1e-7), "HARPQ must minimize its own finite-horizon objective over the same grid"
        ah, an = 1 - torch.exp2(-kh), 1 - torch.exp2(-kn)
        rh = retention_error_report(a, ah, horizon=128)
        rn = retention_error_report(a, an, horizon=128)
        assert rh["retention_rmse"] <= rn["retention_rmse"] + 1e-6
        print(f"[ok] HARPQ({weight_mode}): objective {cn.mean():.6g} -> {ch.mean():.6g}; "
              f"retention RMSE {rn['retention_rmse']:.6g} -> {rh['retention_rmse']:.6g}")


def test_separate_lm_head():
    cfg = tiny_cfg()
    cfg.tie_embeddings = False
    model = Mamba2LM(cfg, QuantConfig())
    assert model.lm_head.weight.data_ptr() != model.backbone.embedding.weight.data_ptr()
    ids = torch.randint(0, cfg.vocab, (2, 7))
    assert model(ids).shape == (2, 7, cfg.vocab)
    print("[ok] untied LM head is represented and produces correctly shaped logits")


def test_harpq_ste_gradients():
    cfg = tiny_cfg()
    model = Mamba2LM(cfg, QuantConfig(decay="shift", decay_quantizer="horizon",
                                     decay_horizon=64, decay_lag_weight="uniform"))
    init_like_pretrained(model)
    ids = torch.randint(0, cfg.vocab, (2, 13))
    loss = model(ids).square().mean()
    loss.backward()
    mix = model.backbone.layers[0].mixer
    assert mix.dt_bias.grad is not None and torch.isfinite(mix.dt_bias.grad).all()
    assert mix.A_log.grad is not None and torch.isfinite(mix.A_log.grad).all()
    assert mix.A_log.grad.abs().sum() > 0
    print("[ok] HARPQ straight-through path propagates finite gradients to dt_bias and A_log")


def test_integer_state_decode():
    cfg = tiny_cfg()
    qi = QuantConfig(decay="shift", integer_state=True, state_frac_bits=20)
    qf = QuantConfig(decay="shift", integer_state=False)
    integer_model = Mamba2LM(cfg, qi).eval()
    init_like_pretrained(integer_model)
    float_model = Mamba2LM(cfg, qf).eval()
    float_model.load_state_dict(integer_model.state_dict())
    ids = torch.randint(0, cfg.vocab, (2, 12))
    ci, si = integer_model.init_states(2, "cpu")
    cf, sf = float_model.init_states(2, "cpu")
    errors = []
    with torch.no_grad():
        for t in range(ids.shape[1]):
            yi = integer_model.step(ids[:, t], ci, si)
            yf = float_model.step(ids[:, t], cf, sf)
            errors.append((yi - yf).abs().max())
    err = torch.stack(errors).max().item()
    assert si[0].dtype == torch.int32 and err < 2e-4, err
    print(f"[ok] integer fixed-point decode fallback: int32 state, max output error {err:.2e}")


def test_checkpoint_roundtrip():
    cfg = tiny_cfg()
    cfg.tie_embeddings = False
    qcfg = QuantConfig(decay="shift", decay_quantizer="horizon", decay_horizon=32,
                       ternary=True, group_size=16)
    model = Mamba2LM(cfg, qcfg)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "roundtrip.pt")
        save_checkpoint(model, path, {"step": 7, "tokens": 1234, "optimizer_state": {"state": {}}})
        loaded, ck = load_checkpoint(path, "cpu")
        assert loaded.qcfg.decay_quantizer == "horizon"
        assert loaded.mcfg.tie_embeddings is False
        assert ck["extra"]["step"] == 7
        assert loaded(ids := torch.randint(0, cfg.vocab, (1, 5))).shape == (1, 5, cfg.vocab)
    print("[ok] checkpoint round-trip preserves HARPQ config, untied head and resume metadata")


def test_official():
    from huggingface_hub import hf_hub_download
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from ssm_core import load_pretrained
    ref = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device="cuda", dtype=torch.float32).eval()
    mine, _ = load_pretrained("state-spaces/mamba2-130m", QuantConfig())
    mine = mine.cuda().eval()
    ids = torch.randint(0, 50277, (1, 512), device="cuda")
    with torch.no_grad():
        a, b = ref(ids).logits, mine(ids)
    err = (a - b[..., : a.shape[-1]]).abs().max().item()
    print(f"[{'ok' if err < 5e-2 else 'FAIL'}] logits vs official mamba_ssm: max abs err {err:.3e}")


if __name__ == "__main__":
    test_scan()
    for decay in ("exp", "shift"):
        for ternary in (False, True):
            test_parallel_vs_step(decay, ternary)
    test_parallel_vs_step("shift", False, "horizon")
    test_parallel_vs_step("shift", True, "horizon")
    test_parallel_vs_step("shift", True, "dyadic")
    test_dyadic_decay()
    test_shift_properties()
    test_ternary()
    test_fp_rows()
    test_activation_aware_ternary()
    test_group_decay()
    test_fixedpoint_and_error_bound()
    test_horizon_aware_decay()
    test_harpq_ste_gradients()
    test_separate_lm_head()
    test_integer_state_decode()
    test_checkpoint_roundtrip()
    test_pot_projection()
    if "--official" in sys.argv:
        test_official()
    print("all tests passed")

"""
kernels.py -- experimental Triton kernels for inference/decode.

  1. ternary_gemv: packed 2-bit ternary projection with select/add/subtract.
  2. shift_scan_step: floating-point PoT decay decode step (not an integer shift).
  3. integer_shift_scan_step: int32 fixed-point state with a true signed integer
     right-shift decay operation. Injection and readout still use FP32 products.

The Triton kernels are experimental and have not been compiled or benchmarked on
this CPU-only development environment. Run `python kernels.py` on the target GPU,
compare against PyTorch references, and report only measured results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                                        # pragma: no cover
    HAS_TRITON = False

from ssm_core import TernaryLinear, pack_ternary, unpack_ternary, Mamba2Mixer

if HAS_TRITON:

    @triton.jit
    def _ternary_gemv_kernel(x_ptr, wp_ptr, s_ptr, y_ptr, O, I,
                             stride_xb, stride_wo, stride_so, stride_yb,
                             GROUP: tl.constexpr, BLOCK_O: tl.constexpr):
        pid_o = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        mask_o = offs_o < O
        offs_g = tl.arange(0, GROUP)
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        for g in range(0, I // GROUP):
            i0 = g * GROUP
            xv = tl.load(x_ptr + pid_b * stride_xb + i0 + offs_g).to(tl.float32)
            byte_idx = (i0 + offs_g) // 4
            wb = tl.load(wp_ptr + offs_o[:, None] * stride_wo + byte_idx[None, :],
                         mask=mask_o[:, None], other=1)
            code = (wb.to(tl.int32) >> ((offs_g % 4) * 2)[None, :]) & 3          # 0:-1  1:0  2:+1
            contrib = tl.where(code == 2, xv[None, :], tl.where(code == 0, -xv[None, :], 0.0))
            part = tl.sum(contrib, axis=1)
            sc = tl.load(s_ptr + offs_o * stride_so + g, mask=mask_o, other=0.0).to(tl.float32)
            acc += part * sc
        tl.store(y_ptr + pid_b * stride_yb + offs_o, acc.to(y_ptr.dtype.element_ty), mask=mask_o)

    @triton.jit
    def _integer_shift_scan_step_kernel(h_ptr, x_ptr, B_ptr, C_ptr, k_ptr, g_ptr, D_ptr,
                                       y_ptr, overflow_ptr, H, P, N: tl.constexpr,
                                       BP: tl.constexpr, NBLOCKS: tl.constexpr, SCALE: tl.constexpr):
        """Fixed-point state update: h_q <- h_q - round(h_q / 2^k) + round(u*2^F)."""
        pid = tl.program_id(0)
        pid_p = tl.program_id(1)
        b = pid // H
        hh = pid % H
        offs_p = pid_p * BP + tl.arange(0, BP)
        offs_n = tl.arange(0, N)
        mp = offs_p < P
        h_off = ((b * H + hh) * P + offs_p[:, None]) * N + offs_n[None, :]
        h = tl.load(h_ptr + h_off, mask=mp[:, None], other=0).to(tl.int64)
        k = tl.load(k_ptr + b * H + hh).to(tl.int32)
        half = tl.full((), 1, tl.int64) << (k - 1)
        pos = (h + half) >> k
        neg = -(((-h) + half) >> k)
        decay_piece = tl.where(h >= 0, pos, neg)

        gain = tl.load(g_ptr + b * H + hh).to(tl.float32)
        x = tl.load(x_ptr + (b * H + hh) * P + offs_p, mask=mp, other=0).to(tl.float32)
        Bv = tl.load(B_ptr + b * N + offs_n).to(tl.float32)
        Cv = tl.load(C_ptr + b * N + offs_n).to(tl.float32)
        inj_scaled = (gain * x)[:, None] * Bv[None, :] * SCALE
        inj_abs = tl.abs(inj_scaled)
        inj_floor = tl.floor(inj_abs)
        inj_frac = inj_abs - inj_floor
        inj_even = (inj_floor.to(tl.int64) & 1) == 0
        inj_round = inj_floor + tl.where(inj_frac > 0.5, 1.0,
                                         tl.where(inj_frac < 0.5, 0.0, tl.where(inj_even, 0.0, 1.0)))
        inj_q = tl.where(inj_scaled >= 0.0, inj_round, -inj_round).to(tl.int64)
        new64 = h - decay_piece + inj_q
        overflow = (new64 > 2147483647) | (new64 < -2147483648)
        clipped = tl.minimum(tl.maximum(new64, -2147483648), 2147483647)
        tl.store(h_ptr + h_off, clipped.to(tl.int32), mask=mp[:, None])
        state_f = clipped.to(tl.float32) / SCALE
        y = tl.sum(state_f * Cv[None, :], axis=1) + tl.load(D_ptr + hh).to(tl.float32) * x
        tl.store(y_ptr + (b * H + hh) * P + offs_p, y, mask=mp)
        overflow_count = tl.sum(tl.sum(overflow.to(tl.int32), axis=1), axis=0)
        tl.store(overflow_ptr + pid * NBLOCKS + pid_p, overflow_count)

    @triton.jit
    def _shift_scan_step_kernel(h_ptr, x_ptr, B_ptr, C_ptr, k_ptr, g_ptr, D_ptr, y_ptr,
                                H, P, N: tl.constexpr, BP: tl.constexpr):
        pid = tl.program_id(0)
        pid_p = tl.program_id(1)
        b = pid // H
        hh = pid % H
        offs_p = pid_p * BP + tl.arange(0, BP)
        offs_n = tl.arange(0, N)
        mp = offs_p < P
        h_off = ((b * H + hh) * P + offs_p[:, None]) * N + offs_n[None, :]
        h = tl.load(h_ptr + h_off, mask=mp[:, None], other=0.0)
        k = tl.load(k_ptr + b * H + hh).to(tl.float32)
        leak = tl.exp2(-k)                                               # 2^-k
        gain = tl.load(g_ptr + b * H + hh).to(tl.float32)
        x = tl.load(x_ptr + (b * H + hh) * P + offs_p, mask=mp, other=0.0).to(tl.float32)
        Bv = tl.load(B_ptr + b * N + offs_n).to(tl.float32)
        Cv = tl.load(C_ptr + b * N + offs_n).to(tl.float32)
        h = h - h * leak + (x * gain)[:, None] * Bv[None, :]
        tl.store(h_ptr + h_off, h, mask=mp[:, None])
        y = tl.sum(h * Cv[None, :], axis=1) + tl.load(D_ptr + hh).to(tl.float32) * x
        tl.store(y_ptr + (b * H + hh) * P + offs_p, y, mask=mp)


def ternary_gemv(x, packed, scales, group, out_dtype=None, block_o=32):
    """x (B, I) fp16/bf16/fp32, packed (O, I/4) uint8, scales (O, I/group)  ->  (B, O)"""
    assert HAS_TRITON and x.is_cuda
    Bsz, I = x.shape
    O = packed.shape[0]
    x = x.contiguous()
    y = torch.empty(Bsz, O, device=x.device, dtype=out_dtype or x.dtype)
    grid = (triton.cdiv(O, block_o), Bsz)
    _ternary_gemv_kernel[grid](x, packed, scales, y, O, I, x.stride(0), packed.stride(0), scales.stride(0),
                               y.stride(0), GROUP=group, BLOCK_O=block_o, num_warps=4)
    return y


def integer_shift_scan_step(h_q, x, B, C, k, gain, D, frac_bits=20, bp=16):
    """Integer-state decode kernel. State is int32 fixed point; returns float output and overflow flags.

    Only decay uses a true integer shift. Input injection and readout still use
    floating-point products and are rounded/accumulated into the fixed-point state.
    """
    assert HAS_TRITON and h_q.is_cuda
    if h_q.dtype != torch.int32:
        raise TypeError("integer state tensor must be int32")
    if frac_bits < 8 or frac_bits > 30:
        raise ValueError("frac_bits must be in [8, 30]")
    b, H, P, N = h_q.shape
    if N & (N - 1):
        raise ValueError("Triton integer kernel currently requires power-of-two d_state")
    y = torch.empty(b, H, P, device=h_q.device, dtype=torch.float32)
    nblocks = triton.cdiv(P, bp)
    overflow = torch.empty(b * H * nblocks, device=h_q.device, dtype=torch.int32)
    grid = (b * H, nblocks)
    _integer_shift_scan_step_kernel[grid](
        h_q, x.contiguous().float(), B.contiguous().float(), C.contiguous().float(),
        k.contiguous().to(torch.int32), gain.contiguous().float(), D.contiguous().float(),
        y, overflow, H, P, N=N, BP=bp, NBLOCKS=nblocks, SCALE=(1 << frac_bits), num_warps=4)
    return y, overflow


def shift_scan_step(h, x, B, C, k, gain, D, bp=16):
    """h (b,H,P,N) fp32 updated IN PLACE; x (b,H,P); B,C (b,N); k,gain (b,H); D (H) -> y (b,H,P) fp32"""
    assert HAS_TRITON and h.is_cuda
    b, H, P, N = h.shape
    y = torch.empty(b, H, P, device=h.device, dtype=torch.float32)
    grid = (b * H, triton.cdiv(P, bp))
    _shift_scan_step_kernel[grid](h, x.contiguous(), B.contiguous(), C.contiguous(), k.float().contiguous(),
                                  gain.float().contiguous(), D.float().contiguous(), y, H, P, N=N, BP=bp)
    return y


class PackedTernaryLinear(nn.Module):
    """Inference-only replacement of TernaryLinear: stores 2-bit packed weights + fp16 group scales.
    Rows flagged fp (sensitivity study) are kept in fp16 and computed separately."""

    def __init__(self, tl_: TernaryLinear):
        super().__init__()
        q, gamma = tl_.ternary_parts()
        O, I = q.shape
        self.O, self.I, self.group = O, I, tl_.group
        self.act_bits = tl_.act_bits
        if tl_.fp_rows is not None:
            q = q.clone()
            q[tl_.fp_rows] = 0
            self.register_buffer("fp_idx", tl_.fp_rows.nonzero().squeeze(-1))
            self.register_buffer("fp_w", tl_.weight.detach()[tl_.fp_rows].half())
        else:
            self.fp_idx, self.fp_w = None, None
        self.register_buffer("packed", pack_ternary(q.to(torch.int8)))
        self.register_buffer("scales", gamma.half())

    def forward(self, x):
        shp = x.shape[:-1]
        x2 = x.reshape(-1, self.I)
        if self.act_bits:
            qmax = 2 ** (self.act_bits - 1) - 1
            scale = x2.abs().amax(-1, keepdim=True).clamp_min(1e-5) / qmax
            x2 = (x2 / scale).round().clamp(-qmax - 1, qmax) * scale
        if HAS_TRITON and x2.is_cuda and x2.shape[0] <= 16:
            y = ternary_gemv(x2.half(), self.packed, self.scales, self.group)
        else:                                                            # prefill / CPU: dequantize on the fly
            q = unpack_ternary(self.packed, self.I).float()
            W = (q.view(self.O, -1, self.group) * self.scales.float().unsqueeze(-1)).view(self.O, self.I)
            y = F.linear(x2.float(), W).to(x2.dtype)
        if self.fp_idx is not None:
            y[:, self.fp_idx] = F.linear(x2.to(self.fp_w.dtype), self.fp_w).to(y.dtype)
        return y.reshape(*shp, self.O)


@torch.no_grad()
def convert_to_packed(model, use_triton_scan=True, use_integer_scan=False):
    """Swap ternary layers for packed inference modules and enable selected decode kernels."""
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, TernaryLinear):
                setattr(mod, cname, PackedTernaryLinear(child))
    for mod in model.modules():
        if isinstance(mod, Mamba2Mixer):
            mod.use_triton_scan = use_triton_scan and HAS_TRITON and model.qcfg.decay == "shift" and not model.qcfg.integer_state
            mod.use_triton_integer_scan = use_integer_scan and HAS_TRITON and model.qcfg.integer_state
    return model


# ----------------------------------------------------------------------------- self test / micro-benchmark
def _bench(fn, iters=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000                               # microseconds


if __name__ == "__main__":
    assert torch.cuda.is_available() and HAS_TRITON, "needs a CUDA GPU and `pip install triton`"
    dev = "cuda"
    torch.manual_seed(0)
    print("== ternary_gemv correctness ==")
    for (O, I, G, Bsz) in [(3072, 768, 128, 1), (1536, 1536, 128, 4), (770, 768, 128, 8)]:
        lin = TernaryLinear(I, O, group_size=G).to(dev)
        pk = PackedTernaryLinear(lin).to(dev)
        x = torch.randn(Bsz, I, device=dev, dtype=torch.float16)
        ref = lin(x.float())
        out = pk(x).float()
        print(f"  O={O} I={I} B={Bsz}: max abs err {(ref - out).abs().max().item():.3e}  (ref scale {ref.abs().mean().item():.2f})")
    print("== integer_shift_scan_step correctness ==")
    from fixedpoint_state import fixedpoint_shift_step
    frac_bits = 20
    state_float = torch.randn(2, 4, 8, 16, device=dev) * 0.05
    state_q_ref = torch.round(state_float * (1 << frac_bits)).to(torch.int32)
    state_q_tri = state_q_ref.clone()
    x_i = torch.randn(2, 4, 8, device=dev) * 0.1
    B_i = torch.randn(2, 16, device=dev) * 0.1
    C_i = torch.randn(2, 16, device=dev)
    k_i = torch.randint(1, 12, (2, 4), device=dev, dtype=torch.int32)
    g_i = torch.rand(2, 4, device=dev)
    D_i = torch.randn(4, device=dev)
    ref_state, ref_y = fixedpoint_shift_step(state_q_ref, x_i, B_i, C_i, k_i, g_i, D_i, frac_bits=frac_bits)
    tri_y, overflow = integer_shift_scan_step(state_q_tri, x_i, B_i, C_i, k_i, g_i, D_i, frac_bits=frac_bits)
    assert not bool(overflow.any().item())
    assert torch.equal(state_q_tri, ref_state), (state_q_tri.to(torch.int64) - ref_state.to(torch.int64)).abs().max().item()
    assert torch.allclose(tri_y, ref_y, atol=2e-5, rtol=2e-5)
    print(f"  int32 state exact match | output max error {(tri_y-ref_y).abs().max().item():.2e}")
    print("== shift_scan_step correctness ==")
    b, H, P, N = 2, 24, 64, 128
    h = torch.randn(b, H, P, N, device=dev)
    x = torch.randn(b, H, P, device=dev); Bv = torch.randn(b, N, device=dev); Cv = torch.randn(b, N, device=dev)
    k = torch.randint(1, 12, (b, H), device=dev).float(); g = torch.rand(b, H, device=dev); D = torch.randn(H, device=dev)
    h_ref = h * (1 - 2.0 ** (-k))[..., None, None] + (g[..., None] * x)[..., None] * Bv[:, None, None, :]
    y_ref = (h_ref * Cv[:, None, None, :]).sum(-1) + D[None, :, None] * x
    y = shift_scan_step(h, x, Bv, Cv, k, g, D)
    print(f"  state err {(h - h_ref).abs().max().item():.2e} | y err {(y - y_ref).abs().max().item():.2e}")
    print("== micro-benchmark: projection GEMV, decode (us/call) ==")
    for (O, I, name) in [(3072 + 0, 768, "in_proj-like 768->3.1k"), (768, 1536, "out_proj-like 1536->768"), (8192, 4096, "large 4096->8192")]:
        lin = TernaryLinear(I, O, group_size=128).to(dev)
        pk = PackedTernaryLinear(lin).to(dev)
        W16 = lin.weight.detach().half()
        for Bsz in (1, 8):
            x = torch.randn(Bsz, I, device=dev, dtype=torch.float16)
            t_fp = _bench(lambda: F.linear(x, W16))
            t_tn = _bench(lambda: pk(x))
            print(f"  {name:26s} B={Bsz}: fp16 {t_fp:7.1f} | ternary-triton {t_tn:7.1f} | weight bytes fp16 {W16.numel()*2/1e6:.1f}MB vs packed {(pk.packed.numel()+pk.scales.numel()*2)/1e6:.1f}MB")

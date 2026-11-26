"""
ssm_core.py
Pure-PyTorch Mamba-2 re-implementation that loads the official `state-spaces/mamba2-*` checkpoints and adds

  (1) power-of-two "leak" decay          a_t = 1 - 2^-k_t,  k_t integer  (shift-based recurrence)
  (2) grouped ternary projections         W ~ gamma_g * {-1, 0, +1}       (BitNet-b1.58 style, group-wise scale)

Layout of the official checkpoint keys is kept (backbone.embedding / backbone.layers.i.{norm,mixer.*} / backbone.norm_f)
so `load_state_dict` works directly.  Assumes ngroups == 1 (true for all released Mamba-2 models).
"""
import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- configs
@dataclass
class ModelConfig:
    d_model: int = 768
    n_layer: int = 24
    d_inner: int = 1536
    d_state: int = 128
    nheads: int = 24
    d_conv: int = 4
    vocab: int = 50288
    chunk: int = 64
    tie_embeddings: bool = True

    @property
    def headdim(self):
        return self.d_inner // self.nheads

    @property
    def conv_dim(self):
        return self.d_inner + 2 * self.d_state

    @property
    def in_proj_out(self):
        return 2 * self.d_inner + 2 * self.d_state + self.nheads


@dataclass
class QuantConfig:
    decay: str = "exp"          # 'exp' (original Mamba-2) | 'shift' (a = 1 - 2^-k)
    gain: str = "dt"            # 'dt' keep pretrained input gain dt | 'leak' use 2^-k (fully shift based, needs more finetuning)
    k_min: float = 1.0          # k>=1 -> a<=0.5 ; (k=0 would mean a=0 and log(0))
    k_max: float = 14.0         # k=14 -> a=0.99994
    k_step: float = 1.0         # 1.0 = pure power of two. 0.5 = half-octave levels (needs a 2-entry LUT, NOT pure shift)
    ternary: bool = False       # ternarize in_proj / out_proj
    group_size: int = 128       # scale group along input dim; 0 = one scale per output row
    fp_rows: str = "none"       # in_proj rows kept full precision: none | dt | bc | dt+bc   (sensitivity study)
    quantize_out_proj: bool = True
    act_bits: int = 0           # 0 = fp activations (W1.58 A16); 8 = per-token absmax int8 activations
    decay_group_size: int = 1    # share k across this many adjacent heads; 1 = per-head, nheads = layer-wise
    activation_aware: bool = False  # calibrate ternary group scales using input-channel second moments
    decay_quantizer: str = "nearest"  # nearest | horizon (HARPQ) | dyadic (multi-term PoT + HARPQ objective)
    decay_horizon: int = 64
    dyadic_terms: int = 2        # terms in the dyadic leak when decay_quantizer == 'dyadic'
    decay_lag_weight: str = "uniform"  # uniform | log | early | late
    integer_state: bool = False  # decode-only fixed-point state; requires shift decay
    state_frac_bits: int = 20
    check_integer_overflow: bool = False  # debug-only; synchronizes GPU each decode step
    fp_layers: str = ""  # comma-separated zero-based blocks whose in_proj and out_proj stay full precision


def ste_round(x):
    return x + (x.round() - x).detach()


# ----------------------------------------------------------------------------- ternary linear
class TernaryLinear(nn.Linear):
    """Latent weights with STE ternary forward values and per-row/group scales.

    Default gamma is mean|W|. In activation-aware mode, gamma is weighted by the
    EMA of input-channel second moments. Scales are not independently optimized,
    avoiding a free scale parameter collapsing during QAT-from-pretrained.
    """

    def __init__(self, in_f, out_f, group_size=128, fp_rows=None, act_bits=0, activation_aware=False):
        super().__init__(in_f, out_f, bias=False)
        self.group = group_size if group_size else in_f
        if in_f % self.group:
            raise ValueError(f"in_features {in_f} not divisible by group {self.group}")
        self.act_bits = act_bits
        self.activation_aware = activation_aware
        self.register_buffer("act_ema", torch.ones(in_f, dtype=torch.float32), persistent=True)
        self.act_ema_decay = 0.99
        self.lam = 1.0                       # quantization strength (annealing); 1.0 = fully quantized
        if fp_rows is not None:
            self.register_buffer("fp_rows", fp_rows, persistent=False)
        else:
            self.fp_rows = None

    @torch.no_grad()
    def ternary_parts(self):
        W = self.weight.detach().float()
        O, I = W.shape
        Wg = W.view(O, I // self.group, self.group)
        if self.activation_aware:
            # Weighted least-squares scale: channels with larger calibrated input
            # second moments contribute more to output reconstruction error.
            ag = self.act_ema.to(W.device).view(1, I // self.group, self.group).clamp_min(1e-8)
            gamma = ((Wg.abs() * ag).sum(-1, keepdim=True) /
                     ag.sum(-1, keepdim=True).clamp_min(1e-8)).clamp(min=1e-6)
        else:
            gamma = Wg.abs().mean(-1, keepdim=True).clamp(min=1e-6)
        q = (Wg / gamma).round().clamp(-1, 1)
        return q.view(O, I), gamma.squeeze(-1)           # (O,I) in {-1,0,1}, (O, I/group)

    def forward(self, x):
        W = self.weight
        O, I = W.shape
        if self.activation_aware and self.training:
            reduce_dims = tuple(range(x.ndim - 1))
            stat = x.detach().float().square().mean(dim=reduce_dims)
            self.act_ema.mul_(self.act_ema_decay).add_(stat.to(self.act_ema.device), alpha=1 - self.act_ema_decay)
        q, gamma = self.ternary_parts()
        Wq = (q.view(O, I // self.group, self.group) * gamma.unsqueeze(-1)).view(O, I)
        Wq = W + self.lam * (Wq.to(W.dtype) - W).detach()
        if self.fp_rows is not None:
            Wq = torch.where(self.fp_rows[:, None], W, Wq)
        if self.act_bits:
            qmax = 2 ** (self.act_bits - 1) - 1
            s = x.abs().amax(-1, keepdim=True).clamp(min=1e-5) / qmax
            xq = (x / s).round().clamp(-qmax - 1, qmax) * s
            x = x + self.lam * (xq - x).detach()
        return F.linear(x, Wq)


def make_fp_rows(mcfg: ModelConfig, mode: str):
    """in_proj output layout: [ z (d_inner) | x (d_inner) | B (d_state) | C (d_state) | dt (nheads) ]"""
    if mode == "none":
        return None
    m = torch.zeros(mcfg.in_proj_out, dtype=torch.bool)
    if "dt" in mode:
        m[-mcfg.nheads:] = True
    if "bc" in mode:
        m[2 * mcfg.d_inner: 2 * mcfg.d_inner + 2 * mcfg.d_state] = True
    return m


# ----------------------------------------------------------------------------- packing helpers
def pack_ternary(q):
    """q: int tensor (O, I) in {-1,0,1}, I % 4 == 0 -> uint8 (O, I/4); code = q+1; 4 codes per byte, LSB first."""
    O, I = q.shape
    c = (q.to(torch.int16) + 1).to(torch.uint8).view(O, I // 4, 4)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def unpack_ternary(p, I):
    shifts = torch.tensor([0, 2, 4, 6], device=p.device)
    c = (p.unsqueeze(-1).to(torch.int16) >> shifts) & 3
    return (c.reshape(p.shape[0], -1)[:, :I] - 1).to(torch.int8)


# ----------------------------------------------------------------------------- SSD scan (ngroups = 1)
def segsum(x):
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    xs = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return xs.masked_fill(~mask, -torch.inf)


def ssd_scan(x, logdecay, B, C, chunk):
    """x (b,l,h,p) [already multiplied by input gain], logdecay (b,l,h) <= 0, B,C (b,l,n)  ->  y (b,l,h,p), final state (b,h,p,n)
    Chunked state-space-duality algorithm; works for ANY per-head scalar decay in (0,1), incl. 1-2^-k."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    pad = (-l) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        logdecay = F.pad(logdecay, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    L = x.shape[1]
    c = L // chunk
    x = x.reshape(b, c, chunk, h, p)
    Bm = B.reshape(b, c, chunk, n)
    Cm = C.reshape(b, c, chunk, n)
    A = logdecay.reshape(b, c, chunk, h).permute(0, 3, 1, 2)             # b h c l
    A_cum = torch.cumsum(A, -1)
    Lmat = torch.exp(segsum(A))                                          # b h c l s
    CB = torch.einsum("bcln,bcsn->bcls", Cm, Bm)
    M = CB.unsqueeze(1) * Lmat
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", M, x)
    decay_states = torch.exp(A_cum[..., -1:] - A_cum)                    # b h c l
    x_dec = x * decay_states.permute(0, 2, 3, 1).unsqueeze(-1)
    states = torch.einsum("bcln,bclhp->bchpn", Bm, x_dec)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], 1)
    chunk_decay = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))       # b h c+1 c+1
    new_states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
    st = new_states[:, :-1]
    sdo = torch.exp(A_cum).permute(0, 2, 3, 1).unsqueeze(-1)
    Y_off = torch.einsum("bcln,bchpn->bclhp", Cm, st) * sdo
    Y = (Y_diag + Y_off).reshape(b, L, h, p)[:, :l]
    return Y, new_states[:, -1]


# ----------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class GatedRMSNorm(nn.Module):
    """Mamba-2 norm_before_gate=False:  rmsnorm(y * silu(z)) * weight"""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, y, z):
        yf = y.float() * F.silu(z.float())
        return (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(y.dtype)


def _linear(i, o, qcfg: QuantConfig, mcfg: ModelConfig, is_in_proj, protect_layer=False):
    if protect_layer or not qcfg.ternary or (not is_in_proj and not qcfg.quantize_out_proj):
        return nn.Linear(i, o, bias=False)
    fp_rows = make_fp_rows(mcfg, qcfg.fp_rows) if is_in_proj else None
    return TernaryLinear(i, o, qcfg.group_size, fp_rows, qcfg.act_bits, qcfg.activation_aware)


class Mamba2Mixer(nn.Module):
    def __init__(self, mcfg: ModelConfig, qcfg: QuantConfig, layer_idx=None):
        super().__init__()
        self.m, self.q = mcfg, qcfg
        protected_layers = {int(x.strip()) for x in qcfg.fp_layers.split(",") if x.strip()}
        protect_layer = layer_idx in protected_layers
        self.in_proj = _linear(mcfg.d_model, mcfg.in_proj_out, qcfg, mcfg, True, protect_layer)
        self.conv1d = nn.Conv1d(mcfg.conv_dim, mcfg.conv_dim, mcfg.d_conv, groups=mcfg.conv_dim,
                                padding=mcfg.d_conv - 1, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(mcfg.nheads))
        self.A_log = nn.Parameter(torch.zeros(mcfg.nheads))
        self.D = nn.Parameter(torch.ones(mcfg.nheads))
        self.norm = GatedRMSNorm(mcfg.d_inner)
        self.out_proj = _linear(mcfg.d_inner, mcfg.d_model, qcfg, mcfg, False, protect_layer)
        self.lam = 1.0
        self.track_k = False
        self.last_k = None
        self.track_state = False
        self.last_state = None
        self.track_decay = False
        self.last_decay_ref = None
        self.last_decay_hat = None
        self.use_triton_scan = False
        self.use_triton_integer_scan = False
        self.last_integer_overflow = None
        self.integer_overflow_seen = None
        if qcfg.decay_group_size < 1 or mcfg.nheads % qcfg.decay_group_size:
            raise ValueError("decay_group_size must be a positive divisor of nheads")
        if qcfg.decay == "shift" and qcfg.k_min < 1:
            raise ValueError("k_min must be >= 1 (k=0 gives decay 0 and log(0))")
        if qcfg.k_step <= 0 or qcfg.k_max < qcfg.k_min:
            raise ValueError("require k_step > 0 and k_max >= k_min")

    # ---- decay: returns (logdecay, gain, k)
    def decay(self, dt, A):
        q = self.q
        a_ref = torch.exp(dt * A).clamp(1e-8, 1.0 - 1e-8)
        if q.decay == "exp":
            if self.track_decay:
                self.last_decay_ref = a_ref.detach()
                self.last_decay_hat = a_ref.detach()
            return dt * A, dt, None
        from retention import exponent_grid, horizon_aware_exponent
        leak = (1.0 - a_ref).clamp(min=1e-8, max=1.0)
        k_cont = (-torch.log2(leak)).clamp(q.k_min, q.k_max)
        # Group sharing is performed before discrete selection, so every head in
        # a group receives the same exponent and the same representative target.
        if q.decay_group_size > 1:
            g = q.decay_group_size
            k_cont = k_cont.reshape(*k_cont.shape[:-1], -1, g).mean(-1, keepdim=True).expand(
                *k_cont.shape[:-1], -1, g).reshape_as(k_cont)
            a_group = (1.0 - torch.exp2(-k_cont)).clamp(1e-8, 1.0 - 1e-8)
        else:
            a_group = a_ref
        if q.decay_quantizer == "dyadic":                                # multi-term PoT leak (research ext.)
            from retention import dyadic_levels, dyadic_decay
            if q.integer_state:
                raise NotImplementedError("dyadic decay has no integer-state kernel yet (needs two shifts)")
            levels = dyadic_levels(int(q.k_min), int(q.k_max), q.dyadic_terms, a_ref.device)
            leak_hat = 1.0 - dyadic_decay(a_group, levels, q.decay_horizon, q.decay_lag_weight)
            leak_cont = torch.exp2(-k_cont)
            pw = leak_cont + self.lam * (leak_hat - leak_cont).detach()      # STE in leak space
            k = -torch.log2(pw)
            if self.track_k:
                self.last_k = k.detach()
            if self.track_decay:
                self.last_decay_ref = a_ref.detach()
                self.last_decay_hat = (1.0 - pw).detach()
            return torch.log1p(-pw), (dt if q.gain == "dt" else pw), k
        k_grid = exponent_grid(q.k_min, q.k_max, q.k_step, a_ref.device)
        if q.decay_quantizer == "horizon":
            k_selected, _, _, _ = horizon_aware_exponent(
                a_group, k_grid, q.decay_horizon, q.decay_lag_weight)
            k_round = k_selected
        else:
            k_round = ((k_cont / q.k_step).round() * q.k_step).clamp(q.k_min, q.k_max)
        k = k_cont + self.lam * (k_round - k_cont).detach()              # STE
        pw = torch.exp2(-k)
        a_hat = 1.0 - pw
        if self.track_k:
            self.last_k = k.detach()
        if self.track_decay:
            self.last_decay_ref = a_ref.detach()
            self.last_decay_hat = a_hat.detach()
        gain = dt if q.gain == "dt" else pw
        return torch.log1p(-pw), gain, k

    def forward(self, u):
        b, l, _ = u.shape
        m = self.m
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [m.d_inner, m.conv_dim, m.nheads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l]).transpose(1, 2)
        x, Bm, Cm = torch.split(xBC, [m.d_inner, m.d_state, m.d_state], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        logdecay, gain, _ = self.decay(dt, A)
        xh = x.float().reshape(b, l, m.nheads, m.headdim)
        y, final_state = ssd_scan(xh * gain.unsqueeze(-1), logdecay, Bm.float(), Cm.float(), m.chunk)
        if self.track_state:
            self.last_state = final_state
        y = y + xh * self.D.float().view(1, 1, -1, 1)
        y = y.reshape(b, l, m.d_inner).to(u.dtype)
        return self.out_proj(self.norm(y, z))

    # ---- single-token recurrent step (decoding). conv_state (b,conv_dim,d_conv), ssm_state (b,h,p,n) fp32
    def step(self, u, conv_state, ssm_state):
        m = self.m
        b = u.shape[0]
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(zxbcdt, [m.d_inner, m.conv_dim, m.nheads], dim=-1)
        conv_state.copy_(torch.roll(conv_state, -1, dims=-1))
        conv_state[:, :, -1] = xBC
        xBC = F.silu((conv_state * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
        x, Bm, Cm = torch.split(xBC, [m.d_inner, m.d_state, m.d_state], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        logdecay, gain, k = self.decay(dt, A)                            # (b,h)
        xh = x.float().reshape(b, m.nheads, m.headdim)
        if self.q.integer_state:
            if k is None:
                raise RuntimeError("integer-state decoding requires power-of-two shift decay")
            k_int = k.detach().round().to(torch.int32)
            if self.use_triton_integer_scan:
                from kernels import integer_shift_scan_step
                y, overflow = integer_shift_scan_step(
                    ssm_state, xh, Bm.float(), Cm.float(), k_int, gain.float(), self.D.float(),
                    frac_bits=self.q.state_frac_bits)
                self.last_integer_overflow = overflow
                if self.integer_overflow_seen is None:
                    self.integer_overflow_seen = overflow.clone()
                else:
                    self.integer_overflow_seen = torch.maximum(self.integer_overflow_seen, overflow)
                if self.q.check_integer_overflow and bool(overflow.any().item()):
                    raise OverflowError("Triton integer-state kernel saturated int32 state; reduce state_frac_bits or rescale inputs")
            else:
                from fixedpoint_state import fixedpoint_shift_step
                new_state, y = fixedpoint_shift_step(
                    ssm_state, xh, Bm.float(), Cm.float(), k_int, gain.float(), self.D.float(),
                    frac_bits=self.q.state_frac_bits)
                ssm_state.copy_(new_state)
        elif self.use_triton_scan and k is not None:
            from kernels import shift_scan_step
            y = shift_scan_step(ssm_state, xh, Bm.float(), Cm.float(), k.detach(), gain.detach(), self.D.float())
        else:
            a = torch.exp(logdecay)
            ssm_state.mul_(a[..., None, None]).add_((gain[..., None] * xh)[..., None] * Bm.float()[:, None, None, :])
            y = (ssm_state * Cm.float()[:, None, None, :]).sum(-1) + self.D.float()[None, :, None] * xh
        y = y.reshape(b, m.d_inner).to(u.dtype)
        return self.out_proj(self.norm(y, z))


class Block(nn.Module):
    def __init__(self, mcfg, qcfg, layer_idx=0):
        super().__init__()
        self.norm = RMSNorm(mcfg.d_model)
        self.mixer = Mamba2Mixer(mcfg, qcfg, layer_idx=layer_idx)

    def forward(self, x):
        return x + self.mixer(self.norm(x))

    def step(self, x, conv_state, ssm_state):
        return x + self.mixer.step(self.norm(x), conv_state, ssm_state)


class Backbone(nn.Module):
    def __init__(self, mcfg, qcfg):
        super().__init__()
        self.embedding = nn.Embedding(mcfg.vocab, mcfg.d_model)
        self.layers = nn.ModuleList([Block(mcfg, qcfg, layer_idx=i) for i in range(mcfg.n_layer)])
        self.norm_f = RMSNorm(mcfg.d_model)


class Mamba2LM(nn.Module):
    def __init__(self, mcfg: ModelConfig, qcfg: QuantConfig = None):
        super().__init__()
        self.mcfg, self.qcfg = mcfg, qcfg or QuantConfig()
        self.backbone = Backbone(mcfg, self.qcfg)
        self.lm_head = nn.Linear(mcfg.d_model, mcfg.vocab, bias=False)
        if mcfg.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, ids):
        x = self.backbone.embedding(ids)
        for blk in self.backbone.layers:
            x = blk(x)
        return self.lm_head(self.backbone.norm_f(x))

    def init_states(self, batch, device):
        m = self.mcfg
        conv = [torch.zeros(batch, m.conv_dim, m.d_conv, device=device) for _ in range(m.n_layer)]
        for blk in self.backbone.layers:
            blk.mixer.last_integer_overflow = None
            blk.mixer.integer_overflow_seen = None
        ssm_dtype = torch.int32 if self.qcfg.integer_state else torch.float32
        ssm = [torch.zeros(batch, m.nheads, m.headdim, m.d_state, device=device, dtype=ssm_dtype)
               for _ in range(m.n_layer)]
        return conv, ssm

    def step(self, ids, conv, ssm):
        x = self.backbone.embedding(ids)
        for i, blk in enumerate(self.backbone.layers):
            x = blk.step(x, conv[i], ssm[i])
        return self.lm_head(self.backbone.norm_f(x))

    def set_quant_strength(self, lam):
        for mod in self.modules():
            if hasattr(mod, "lam"):
                mod.lam = lam

    def set_track_k(self, flag):
        for mod in self.modules():
            if isinstance(mod, Mamba2Mixer):
                mod.track_k = flag

    def set_track_state(self, flag):
        for mod in self.modules():
            if isinstance(mod, Mamba2Mixer):
                mod.track_state = flag
                if not flag:
                    mod.last_state = None

    def set_track_decay(self, flag):
        for mod in self.modules():
            if isinstance(mod, Mamba2Mixer):
                mod.track_decay = flag
                if not flag:
                    mod.last_decay_ref = None
                    mod.last_decay_hat = None


# ----------------------------------------------------------------------------- loading / saving
def infer_config(sd, chunk=64) -> ModelConfig:
    vocab, d_model = sd["backbone.embedding.weight"].shape
    n_layer = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.layers."))
    p = "backbone.layers.0.mixer."
    nheads = sd[p + "A_log"].shape[0]
    d_inner = sd[p + "norm.weight"].shape[0]
    conv_dim, _, d_conv = sd[p + "conv1d.weight"].shape
    d_state = (conv_dim - d_inner) // 2
    assert sd[p + "in_proj.weight"].shape[0] == 2 * d_inner + 2 * d_state + nheads, \
        "unexpected in_proj shape (ngroups != 1 or d_mlp != 0 is not supported)"
    return ModelConfig(d_model, n_layer, d_inner, d_state, nheads, d_conv, vocab, chunk,
                       tie_embeddings=("lm_head.weight" not in sd))


def _load_hf_state_dict(repo_id: str):
    """Load a single-file or sharded HF PyTorch/safetensors checkpoint."""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safe_load
    # Prefer the canonical safetensors files, then legacy PyTorch weights.
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(repo_id, filename)
            state = safe_load(path) if filename.endswith(".safetensors") else torch.load(
                path, map_location="cpu", weights_only=True)
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            return state
        except Exception:
            pass
    # Sharded safetensors index.
    try:
        index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
        index = json.load(open(index_path))
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(safe_load(hf_hub_download(repo_id, shard)))
        return state
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model.safetensors, pytorch_model.bin, or a safetensors shard index from {repo_id}. "
            "Check the repository files and access permissions.") from exc


def load_pretrained(name="state-spaces/mamba2-130m", qcfg: QuantConfig = None, chunk=64):
    sd = _load_hf_state_dict(name)
    sd = {k: v.float() if torch.is_floating_point(v) else v for k, v in sd.items()}
    # Some HF repositories wrap the model under a 'model.' prefix; normalize only
    # when the expected embedding key is absent and the prefixed key is present.
    if "backbone.embedding.weight" not in sd and "model.backbone.embedding.weight" in sd:
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
    mcfg = infer_config(sd, chunk)
    model = Mamba2LM(mcfg, qcfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ignorable = ("fp_rows", "act_ema")
    missing = [k for k in missing if not k.endswith(ignorable)]
    # Ignore only well-known training-only metadata; unexpected model weights are
    # printed and surfaced so a checkpoint mismatch is never silently hidden.
    print(f"[load_pretrained] {name}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | "
          f"missing={missing} | unexpected={unexpected}")
    if missing:
        raise RuntimeError(f"checkpoint is missing required model weights: {missing}")
    return model, mcfg


def save_checkpoint(model, path, extra=None):
    """Save model weights/config plus optional optimizer/scaler/RNG resume state."""
    torch.save({"state_dict": model.state_dict(), "mcfg": asdict(model.mcfg), "qcfg": asdict(model.qcfg),
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Mamba2LM(ModelConfig(**ck["mcfg"]), QuantConfig(**ck["qcfg"]))
    model.load_state_dict(ck["state_dict"], strict=False)
    return model.to(device), ck


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


# ----------------------------------------------------------------------------- reporting
@torch.no_grad()
def size_report(model):
    """Bytes if exported with 2-bit packed ternary weights + fp16 group scales; everything else fp16."""
    tern_bits, scale_bits, fp_params = 0, 0, 0
    for mod in model.modules():
        if isinstance(mod, TernaryLinear):
            O, I = mod.weight.shape
            n_fp = int(mod.fp_rows.sum()) * I if mod.fp_rows is not None else 0
            tern_bits += 2 * (O * I - n_fp)
            n_fp_rows = int(mod.fp_rows.sum()) if mod.fp_rows is not None else 0
            scale_bits += 16 * (O - n_fp_rows) * (I // mod.group)
            fp_params += n_fp
    tern_params = tern_bits // 2
    total_params = sum(p.numel() for p in model.parameters())
    other = total_params - tern_params - fp_params
    fp16_bytes = total_params * 2
    packed_bytes = tern_bits / 8 + scale_bits / 8 + (other + fp_params) * 2
    return dict(total_params=total_params, ternary_params=tern_params, fp16_MB=fp16_bytes / 2**20,
                packed_MB=packed_bytes / 2**20, ratio=fp16_bytes / packed_bytes,
                effective_bits_per_param=8 * packed_bytes / total_params)

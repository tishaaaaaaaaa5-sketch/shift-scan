"""
Reference implementation of integer-state power-of-two decay.

This is an arithmetic correctness oracle, NOT a fast kernel. It demonstrates what a
true integer right-shift recurrence means and lets experiments quantify fixed-point
rounding/overflow before writing a CUDA/Triton implementation.

state_q represents state * 2**frac_bits. The decay term is rounded symmetrically
before the arithmetic right shift to avoid a systematic negative-value bias.
"""
from __future__ import annotations
import torch


def round_shift_signed(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Round signed int64 x / 2**k to nearest, ties away from zero."""
    if x.dtype not in (torch.int32, torch.int64):
        raise TypeError("x must be integer")
    k = k.to(device=x.device, dtype=torch.int64)
    while k.ndim < x.ndim:
        k = k.unsqueeze(-1)
    half = torch.bitwise_left_shift(torch.ones_like(k), (k - 1).clamp_min(0))
    pos = torch.bitwise_right_shift(x + half, k)
    neg = -torch.bitwise_right_shift(-x + half, k)
    return torch.where(x >= 0, pos, neg)


def fixedpoint_shift_step(
    state_q: torch.Tensor,
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    k: torch.Tensor,
    gain: torch.Tensor,
    D: torch.Tensor,
    frac_bits: int = 20,
    saturate_int32: bool = True,
):
    """
    One decode step.

    Shapes:
      state_q: (batch, heads, head_dim, d_state), int64/int32
      x:       (batch, heads, head_dim)
      B, C:    (batch, d_state)
      k,gain:  (batch, heads)
      D:       (heads,)

    Returns (new_state_q, y_float). B*x, C*state, gain and D remain multiplies;
    only the state-decay scaling is implemented as an integer shift.
    """
    if state_q.ndim != 4:
        raise ValueError("state_q must have shape (batch, heads, head_dim, d_state)")
    if not torch.is_floating_point(x):
        raise TypeError("x must be floating point")
    if frac_bits < 8 or frac_bits > 30:
        raise ValueError("frac_bits must be in [8, 30]")
    k_i = k.round().to(torch.int64)
    if torch.any(k_i < 1) or torch.any(k_i > 30):
        raise ValueError("integer shift exponents must be in [1, 30]")

    h = state_q.to(torch.int64)
    decay_piece = round_shift_signed(h, k_i)
    scale = float(1 << frac_bits)
    injection = (
        (gain[..., None] * x)[..., None] * B[:, None, None, :]
    )
    injection_q = torch.round(injection * scale).to(torch.int64)
    new_state = h - decay_piece + injection_q

    if saturate_int32:
        info = torch.iinfo(torch.int32)
        if torch.any(new_state < info.min) or torch.any(new_state > info.max):
            raise OverflowError(
                "fixed-point state overflowed int32; lower frac_bits, rescale inputs, "
                "or use a wider accumulator"
            )
        new_state = new_state.to(torch.int32)

    state_float = new_state.to(torch.float32) / scale
    y = (state_float * C[:, None, None, :]).sum(-1)
    y = y + D[None, :, None].float() * x.float()
    return new_state, y

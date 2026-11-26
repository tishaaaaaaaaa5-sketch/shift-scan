"""
Small analytical helpers for Shift-Scan error analysis.

For h_t = A_t h_{t-1} + u_t and hhat_t = Ahat_t hhat_{t-1} + uhat_t:
  e_t <= |A_t| e_{t-1} + |A_t-Ahat_t| |hhat_{t-1}| + |u_t-uhat_t|.
The recurrence can be iterated to produce a sample-wise deterministic bound.
"""
from __future__ import annotations
import torch


def recurrent_error_bound(A, A_hat, h_hat, u, u_hat, initial_error=None):
    """Return per-step upper bounds; h_hat is the magnitude (or norm) of the approximate state."""
    if not (A.shape == A_hat.shape == u.shape == u_hat.shape == h_hat.shape):
        raise ValueError("A, A_hat, h_hat magnitude, u and u_hat must have identical shapes")
    e = torch.zeros_like(u[0]) if initial_error is None else initial_error.abs()
    bounds = []
    for t in range(u.shape[0]):
        e = A[t].abs() * e + (A[t] - A_hat[t]).abs() * h_hat[t].abs() + (u[t] - u_hat[t]).abs()
        bounds.append(e)
    return torch.stack(bounds, dim=0)


def power_of_two_decay(leak_exponent: torch.Tensor) -> torch.Tensor:
    """Ahat = 1 - 2**(-k), preserving input dtype/device."""
    return 1.0 - torch.exp2(-leak_exponent)


def decay_grid_report(A: torch.Tensor, k_min=1, k_max=14, step=1.0):
    """Compare FP decay A in (0,1) with nearest allowed 1-2^-k grid."""
    if torch.any(A <= 0) or torch.any(A >= 1):
        raise ValueError("A must contain decay values strictly between zero and one")
    leak = (1 - A).clamp_min(torch.finfo(A.dtype).tiny)
    k_cont = (-torch.log2(leak)).clamp(k_min, k_max)
    k = ((k_cont / step).round() * step).clamp(k_min, k_max)
    A_hat = power_of_two_decay(k)
    return {
        "k": k,
        "A_hat": A_hat,
        "mae": (A - A_hat).abs().mean(),
        "max_abs_error": (A - A_hat).abs().max(),
        "mean_relative_leak_error": ((1 - A_hat) - (1 - A)).abs().div(1 - A).mean(),
    }

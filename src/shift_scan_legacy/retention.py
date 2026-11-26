"""Horizon-aware power-of-two retention quantization (HARPQ).

Research proposal implemented here: select a discrete leak exponent by minimizing
an integrated finite-horizon retention-curve distortion, rather than minimizing
only one-step leak error. This is a local-stationary approximation: each token's
continuous decay is treated as constant over the calibration horizon. It is a
hypothesis to test, not a claim of established novelty.
"""
from __future__ import annotations
import torch


def exponent_grid(k_min: float = 1, k_max: float = 14, step: float = 1.0,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    if step <= 0 or k_max < k_min:
        raise ValueError("invalid exponent grid")
    n = int(round((k_max - k_min) / step)) + 1
    grid = torch.arange(n, device=device, dtype=dtype) * step + k_min
    return grid[grid <= k_max + 1e-6]


def horizon_weights(horizon: int, mode: str = "uniform", device=None,
                    dtype=torch.float32) -> torch.Tensor:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    lag = torch.arange(1, horizon + 1, device=device, dtype=dtype)
    if mode == "uniform":
        w = torch.ones_like(lag)
    elif mode == "log":
        w = 1.0 / torch.log2(lag + 1.0)
    elif mode == "early":
        w = torch.exp(-(lag - 1.0) / max(1.0, horizon / 8.0))
    elif mode == "late":
        w = lag / lag.sum()
    else:
        raise ValueError("weight mode must be uniform|log|early|late")
    return w / w.sum()


def nearest_leak_exponent(a: torch.Tensor, k_grid: torch.Tensor) -> torch.Tensor:
    """Nearest one-step leak in log2-leak space (the conventional baseline)."""
    leak = (1.0 - a.float()).clamp_min(1e-12)
    k_cont = -torch.log2(leak)
    return k_grid[(k_cont[..., None] - k_grid).abs().argmin(dim=-1)]


def _geometric_sum(z: torch.Tensor, horizon: int) -> torch.Tensor:
    """sum_{r=1..H} z**r, stable for z close to one."""
    z = z.clamp(0.0, 1.0)
    denom = (1.0 - z).clamp_min(1e-12)
    value = z * (1.0 - z.pow(horizon)) / denom
    return torch.where((1.0 - z).abs() < 1e-7, z * horizon, value)


def horizon_aware_exponent(a: torch.Tensor, k_grid: torch.Tensor,
                           horizon: int = 64, weight_mode: str = "uniform",
                           chunk_size: int = 4096):
    """Choose k minimizing weighted squared retention error over lags 1..H.

    Uniform lag weights use a closed-form geometric-series identity and are
    efficient enough for the training hot path. Other weight profiles use a
    chunked exact calculation to bound temporary memory.
    """
    a = a.float().clamp(1e-8, 1.0 - 1e-8)
    grid = k_grid.to(device=a.device, dtype=torch.float32)
    candidates = 1.0 - torch.exp2(-grid)
    flat = a.reshape(-1)
    costs = []
    if weight_mode == "uniform":
        # sum_r (a^r-c^r)^2 = S_H(a^2)+S_H(c^2)-2*S_H(ac)
        sa = _geometric_sum(flat.square(), horizon).unsqueeze(-1)
        sc = _geometric_sum(candidates.square(), horizon).unsqueeze(0)
        sac = _geometric_sum(flat.unsqueeze(-1) * candidates.unsqueeze(0), horizon)
        all_costs = ((sa + sc - 2.0 * sac) / horizon).clamp_min(0.0)
    else:
        lag = torch.arange(1, horizon + 1, device=a.device, dtype=torch.float32)
        weights = horizon_weights(horizon, weight_mode, a.device, torch.float32)
        for lo in range(0, flat.numel(), chunk_size):
            part = flat[lo:lo + chunk_size]
            target = part[:, None, None].pow(lag[None, None, :])
            approx = candidates[None, :, None].pow(lag[None, None, :])
            costs.append(((target - approx).square() * weights).sum(-1))
        all_costs = torch.cat(costs, dim=0)
    idx = all_costs.argmin(-1)
    selected = grid[idx].reshape_as(a)
    best_cost = all_costs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).reshape_as(a)
    nearest = nearest_leak_exponent(a, grid)
    nidx = (grid.view(*([1] * a.ndim), -1) == nearest.unsqueeze(-1)).to(torch.int64).argmax(-1)
    nearest_cost = all_costs.gather(-1, nidx.reshape(-1, 1)).squeeze(-1).reshape_as(a)
    return selected, best_cost, nearest, nearest_cost


def retention_error_report(a: torch.Tensor, a_hat: torch.Tensor, horizon: int = 256):
    """One-step and stationary multi-lag retention distortion statistics."""
    a = a.float().clamp(0, 1)
    ah = a_hat.float().clamp(0, 1)
    lag = torch.arange(1, horizon + 1, device=a.device, dtype=torch.float32)
    ref = a.unsqueeze(-1).pow(lag)
    pred = ah.unsqueeze(-1).pow(lag)
    err = pred - ref
    return {
        "one_step_mae": (a - ah).abs().mean().item(),
        "retention_mae": err.abs().mean().item(),
        "retention_rmse": err.square().mean().sqrt().item(),
        "retention_max_abs": err.abs().max().item(),
        "horizon": int(horizon),
    }


def nonstationary_retention(decays: torch.Tensor, max_lag: int = 256):
    """Exact products for observed nonstationary decay sequences.

    decays has time on dimension 0. Returns retention[t, lag-1, ...] for all
    available start positions, with NaN where a full lag is unavailable.
    Intended for analysis, not the training hot path.
    """
    if decays.ndim < 1:
        raise ValueError("decays must have a time dimension")
    T = decays.shape[0]
    H = min(max_lag, T - 1)
    out = []
    for lag in range(1, H + 1):
        prod = torch.ones_like(decays[:-lag])
        for j in range(1, lag + 1):
            prod = prod * decays[j:j + T - lag]
        pad = torch.full_like(decays, float("nan"))
        pad[lag:] = prod
        out.append(pad)
    return torch.stack(out, dim=1) if out else decays.new_empty((T, 0, *decays.shape[1:]))


# ---------------------------------------------------------------------------
# Dyadic (multi-term) power-of-two leak: leak = sum_i 2^-k_i  =>  a_hat = 1 - sum_i 2^-k_i.
# Hardware: h - (h>>k1) - (h>>k2): two shifts + two subtracts, still multiplier-free.
# 14 one-term levels -> ~118 two-term levels on k in [1,14] (see research_suite.py S1).
# ---------------------------------------------------------------------------
def dyadic_levels(k_min: int = 1, k_max: int = 14, terms: int = 2,
                  device=None) -> torch.Tensor:
    """Sorted unique decay levels 1 - sum 2^-k_i (1..terms terms, k non-decreasing), all in (0,1)."""
    ks = [2.0 ** -k for k in range(int(k_min), int(k_max) + 1)]
    leaks = {0.0}
    for _ in range(terms):
        leaks = leaks | {l + p for l in leaks for p in ks if l + p < 1.0}   # union keeps fewer-term levels
    leaks.discard(0.0)
    return torch.tensor(sorted(1.0 - l for l in leaks), device=device, dtype=torch.float32)


def retention_costs(a: torch.Tensor, cand: torch.Tensor, horizon: int,
                    weight_mode: str = "uniform", chunk_size: int = 2048) -> torch.Tensor:
    """(n, m) weighted retention distortion between targets a:(n,) and candidates cand:(m,)."""
    if weight_mode == "uniform":
        sa = _geometric_sum(a.square(), horizon).unsqueeze(-1)
        sc = _geometric_sum(cand.square(), horizon).unsqueeze(0)
        sac = _geometric_sum(a.unsqueeze(-1) * cand.unsqueeze(0), horizon)
        return ((sa + sc - 2.0 * sac) / horizon).clamp_min(0.0)
    lag = torch.arange(1, horizon + 1, device=a.device, dtype=torch.float32)
    w = horizon_weights(horizon, weight_mode, a.device, torch.float32)
    out = []
    for lo in range(0, a.numel(), chunk_size):
        part = a[lo:lo + chunk_size]
        d = part[:, None, None].pow(lag) - cand[None, :, None].pow(lag)
        out.append((d.square() * w).sum(-1))
    return torch.cat(out, 0)


def dyadic_decay(a: torch.Tensor, levels: torch.Tensor, horizon: int = 64,
                 weight_mode: str = "uniform") -> torch.Tensor:
    """Pick the level in `levels` minimising weighted retention distortion; returns a_hat, shape of a."""
    a32 = a.float().clamp(1e-8, 1.0 - 1e-8)
    lv = levels.to(device=a.device, dtype=torch.float32)
    idx = retention_costs(a32.reshape(-1), lv, horizon, weight_mode).argmin(-1)
    return lv[idx].reshape_as(a32)

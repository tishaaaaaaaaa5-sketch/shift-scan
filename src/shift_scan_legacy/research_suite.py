"""Shift-Scan analysis suite (NumPy only; runs on CPU in seconds).

Every study is a controlled numerical experiment on a *stated prior* (Mamba-2 initialisation
distribution) or on exact arithmetic -- NOT a result from a trained checkpoint. Use
calibrate_decay.py for the real-checkpoint version of Study 1.

  S1  decay-grid study: nearest vs HARPQ vs dyadic two-term PoT (new), horizon sweep
  S2  fixed-point dead zone: analytic floor 2^(k-1-F) vs simulation, rounding vs truncation
  S3  ternary scale rules vs output error under heterogeneous activation variance
  S4  recurrent error bound: tightness and growth vs sequence length
  S5  packed-size accounting for Mamba-2 130M
Usage: python research_suite.py --out results/suite
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
K = np.arange(1, 15, dtype=np.float64)           # leak-exponent grid
CAND = 1.0 - 2.0 ** -K                             # 1 - 2^-k


# ---------- shared helpers ----------
def mamba2_init_decay(n=20000):
    """a = exp(-dt*A) under Mamba-2 init: A~U(1,16), dt~logU(1e-3,0.1)."""
    A = rng.uniform(1, 16, n)
    dt = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), n))
    return np.exp(-dt * A)


def lag_weights(H, mode="uniform"):
    r = np.arange(1, H + 1, dtype=np.float64)
    w = {"uniform": np.ones(H), "log": 1 / np.log2(r + 1),
         "early": np.exp(-(r - 1) / max(1, H / 8)), "late": r}[mode]
    return w / w.sum()


def retention_cost(a, c, H, mode="uniform"):
    """weighted sum_r w_r (a^r - c^r)^2 ; a:(n,1), c:(1,m) -> (n,m)"""
    r = np.arange(1, H + 1)
    w = lag_weights(H, mode)
    ar = a[..., None] ** r
    cr = c[..., None] ** r
    return (((ar - cr) ** 2) * w).sum(-1)


def pick(a, cands, H, mode="uniform"):
    out = np.empty(len(a))
    for lo in range(0, len(a), 2000):
        out[lo:lo + 2000] = cands[retention_cost(a[lo:lo + 2000, None], cands[None, :], H, mode).argmin(1)]
    return out


def rmse(a, ah, H=256):
    r = np.arange(1, H + 1)
    return float(np.sqrt(np.mean((a[:, None] ** r - ah[:, None] ** r) ** 2)))


# ---------- S1 ----------
def study1(out):
    a = mamba2_init_decay(4000)
    nearest = CAND[np.abs(-np.log2(1 - a)[:, None] - K).argmin(1)]
    # dyadic: leak = sum of <=2 terms 2^-k, k in 1..14 (same construction as retention.dyadic_levels)
    ks = [2.0 ** -k for k in range(1, 15)]
    L = {0.0}
    L = L | {l + p for l in L for p in ks}
    L = L | {l + p for l in L for p in ks if l + p < 1.0}
    L.discard(0.0)
    dy = np.array(sorted(1.0 - l for l in L))
    res = {"n_levels": {"pot_1term": len(CAND), "dyadic_2term": len(dy)}, "horizon": {}}
    for H in (16, 32, 64, 128, 256):
        har = pick(a, CAND, H)
        dya = pick(a, dy, H)
        nea = nearest
        res["horizon"][H] = {
            "nearest": rmse(a, nea), "harpq": rmse(a, har), "dyadic2": rmse(a, dya),
            "frac_k_changed": float(np.mean(har != nea))}
    # lag-weight sweep at H=128
    res["lag_weight"] = {m: rmse(a, pick(a, CAND, 128, m)) for m in ("uniform", "log", "early", "late")}
    # retention curve for 3 representative decays
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    Hs = list(res["horizon"])
    for name, c in (("nearest", "tab:gray"), ("harpq", "tab:blue"), ("dyadic2", "tab:green")):
        ax[0].plot(Hs, [res["horizon"][h][name] for h in Hs], "o-", label=name, color=c)
    ax[0].set(xlabel="calibration horizon H", ylabel="retention RMSE (lags<=256)",
              title="S1a  retention error vs horizon", yscale="log"); ax[0].legend()
    r = np.arange(1, 513)
    for a0 in (0.90, 0.99, 0.999):
        i = np.argmin(np.abs(a - a0))
        ax[1].plot(r, a[i] ** r, color="k", lw=1.5, alpha=.8)
        ax[1].plot(r, nearest[i] ** r, "--", color="tab:gray")
        ax[1].plot(r, pick(a[i:i + 1], CAND, 128)[0] ** r, "-.", color="tab:blue")
        ax[1].plot(r, pick(a[i:i + 1], dy, 128)[0] ** r, ":", color="tab:green")
    ax[1].set(xlabel="lag", ylabel="retention a^r", title="S1b  curves: FP(black) near(gray) HARPQ(blue) dyadic(green)")
    ax[2].hist(-np.log2(1 - a), bins=60, alpha=.7)
    ax[2].set(xlabel="continuous k = -log2(1-a)", ylabel="count", title="S1c  where Mamba-2 init decays sit")
    plt.tight_layout(); plt.savefig(f"{out}/s1_decay_grid.png", dpi=140); plt.close()
    return res


# ---------- S2 ----------
def fixed_decay(h, k, mode):
    if mode == "trunc":
        return h - (h >> k)
    half = 1 << (k - 1)
    return h - np.where(h >= 0, (h + half) >> k, -((-h + half) >> k))


def study2(out):
    res = {}
    ks = [4, 6, 8, 10, 12]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for F in (16, 20, 24):
        floors = []
        for k in ks:
            h = np.array([1 << (F + 2)], dtype=np.int64)       # start at 4.0
            for _ in range(60000):
                n = fixed_decay(h, k, "round")
                if n[0] == h[0]:
                    break
                h = n
            floors.append(abs(int(h[0])) / 2.0 ** F)
        res[f"F{F}"] = {"k": ks, "sim_floor": floors,
                        "analytic": [2.0 ** (k - 1 - F) for k in ks]}
        ax[0].plot(ks, floors, "o-", label=f"sim F={F}")
        ax[0].plot(ks, [2.0 ** (k - 1 - F) for k in ks], "k:", lw=1)
    ax[0].set(yscale="log", xlabel="shift k", ylabel="residual |state| (dead zone)",
              title="S2a  decay stalls at 2^(k-1-F)  (dotted = analytic)"); ax[0].legend()
    # truncation vs rounding: drift of a decaying state that should reach 0
    k, F = 8, 20
    traj = {}
    for mode in ("trunc", "round"):
        h = np.array([1 << (F - 2), -(1 << (F - 2))], dtype=np.int64)
        tr = []
        for _ in range(4000):
            h = fixed_decay(h, k, mode); tr.append(h / 2.0 ** F)
        traj[mode] = np.array(tr)
    res["trunc_final"] = traj["trunc"][-1].tolist(); res["round_final"] = traj["round"][-1].tolist()
    ideal = 0.25 * (1 - 2.0 ** -k) ** np.arange(1, 4001)
    ax[1].plot(ideal, "k", label="ideal +0.25"); ax[1].plot(traj["trunc"][:, 0], label="trunc +")
    ax[1].plot(traj["trunc"][:, 1], label="trunc -"); ax[1].plot(traj["round"][:, 0], "--", label="round +")
    ax[1].set(xlabel="step", ylabel="state", title="S2b  truncation is asymmetric: +/- states stop at different floors (k=8,F=20)")
    ax[1].legend(); plt.tight_layout(); plt.savefig(f"{out}/s2_deadzone.png", dpi=140); plt.close()
    return res


# ---------- S3 ----------
def ternary(W, act_var, mode, g=128):
    """per-group ternary; scale rules: absmean | twn (0.7*mean|w| thr) | actaware (act-weighted)"""
    O, I = W.shape
    Wg = W.reshape(O, I // g, g)
    v = act_var.reshape(1, I // g, g)
    if mode == "absmean":
        s = np.abs(Wg).mean(-1, keepdims=True)
        Q = np.clip(np.round(Wg / s), -1, 1)
        return (Q * s).reshape(O, I)
    if mode == "twn":
        thr = 0.7 * np.abs(Wg).mean(-1, keepdims=True)
        Q = np.where(np.abs(Wg) > thr, np.sign(Wg), 0.0)
        m = np.abs(Q).sum(-1, keepdims=True).clip(1)
        s = (np.abs(Wg) * np.abs(Q)).sum(-1, keepdims=True) / m
        return (Q * s).reshape(O, I)
    if mode == "twn_act":
        thr = 0.7 * np.abs(Wg).mean(-1, keepdims=True)
        Q = np.where(np.abs(Wg) > thr, np.sign(Wg), 0.0)
        s = (v * Wg * Q).sum(-1, keepdims=True) / (v * Q * Q).sum(-1, keepdims=True).clip(1e-12)
        return (Q * s).reshape(O, I)
    # activation-aware: Q by absmean, scale minimises E[(x.(w-sQ))^2] = sum v*(w-sQ)^2 -> s=sum v*w*Q/sum v*Q^2
    s0 = np.abs(Wg).mean(-1, keepdims=True)
    Q = np.clip(np.round(Wg / s0), -1, 1)
    s = (v * Wg * Q).sum(-1, keepdims=True) / (v * Q * Q).sum(-1, keepdims=True).clip(1e-12)
    return (Q * s).reshape(O, I)


def study3(out, trials=5):
    O, I, N = 512, 1024, 4096
    res = {}
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))
    for sigma in (0.0, 0.5, 1.0, 1.5, 2.0):          # log-std of per-channel activation scale
        errs = {m: [] for m in ("absmean", "twn", "actaware", "twn_act")}
        for t in range(trials):
            W = rng.standard_t(5, (O, I)) * 0.02       # heavy-ish tails
            sc = np.exp(rng.normal(0, sigma, I))
            X = rng.normal(0, 1, (N, I)) * sc
            var = (X ** 2).mean(0)
            Y = X @ W.T
            for m in errs:
                Yh = X @ ternary(W, var, m).T
                errs[m].append(np.linalg.norm(Y - Yh) / np.linalg.norm(Y))
        res[f"sigma{sigma}"] = {m: float(np.mean(v)) for m, v in errs.items()}
    for m in ("absmean", "twn", "actaware", "twn_act"):
        ax.plot([float(k[5:]) for k in res], [res[k][m] for k in res], "o-", label=m)
    ax.set(xlabel="activation-scale heterogeneity (log-std per channel)", ylabel="relative output error",
           title="S3  ternary scale rule vs output error"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{out}/s3_ternary.png", dpi=140); plt.close()
    return res


# ---------- S4 ----------
def study4(out):
    n, Ts = 64, (64, 256, 1024, 4096)
    res, fig_data = {}, {}
    for T in Ts:
        a = np.clip(mamba2_init_decay(n), 0, 0.99999)
        ah = CAND[np.abs(-np.log2(1 - a)[:, None] - K).argmin(1)]
        u = rng.normal(0, 1, (T, n)); uh = u + rng.normal(0, 0.01, (T, n))   # ternary-like input noise
        h = np.zeros(n); hh = np.zeros(n); e_b = np.zeros(n); act, bnd = [], []
        for t in range(T):
            e_b = a * e_b + np.abs(a - ah) * np.abs(hh) + np.abs(u[t] - uh[t])   # uses hhat_{t-1}
            h = a * h + u[t]; hh = ah * hh + uh[t]
            act.append(np.abs(h - hh)); bnd.append(e_b.copy())
        act, bnd = np.array(act), np.array(bnd)
        res[T] = {"mean_actual": float(act.mean()), "mean_bound": float(bnd.mean()),
                  "bound_over_actual": float(bnd.mean() / act.mean()),
                  "bound_valid": bool(np.all(bnd >= act - 1e-9))}
        fig_data[T] = (act.mean(1), bnd.mean(1))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    t = max(Ts)
    ax.plot(fig_data[t][0], label="actual |h-ĥ|"); ax.plot(fig_data[t][1], label="deterministic bound")
    ax.set(yscale="log", xlabel="step", ylabel="mean error", title="S4  bound validity & looseness (T=4096)"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{out}/s4_bound.png", dpi=140); plt.close()
    return res


# ---------- S5 ----------
def study5():
    d, L, di, ds, nh, V, g = 768, 24, 1536, 128, 24, 50288, 128
    inp = d * (2 * di + 2 * ds + nh); outp = di * d
    per_layer = inp + outp
    tern_w = L * per_layer
    emb = V * d
    def bytes_tern(n, pack):            # pack: 2 bit/weight or base-3 (5 trits/byte)
        bw = n * 2 / 8 if pack == "2bit" else n / 5
        scales = n / g * 2              # fp16 scale per group
        return bw + scales
    row = lambda name, b: {"scheme": name, "MB": round(b / 2 ** 20, 1)}
    fp16 = (tern_w + emb) * 2
    return {"ternary_params_M": tern_w / 1e6, "embedding_params_M": emb / 1e6,
            "rows": [row("FP16 all", fp16),
                     row("ternary proj 2-bit + fp16 emb", bytes_tern(tern_w, "2bit") + emb * 2),
                     row("ternary proj base-3 + fp16 emb", bytes_tern(tern_w, "b3") + emb * 2),
                     row("ternary proj 2-bit + int8 emb", bytes_tern(tern_w, "2bit") + emb)],
            "note": "excludes conv, norms, A/D/dt_bias, FP rows kept for sensitive projections"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="results/suite"); a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {"S1": study1(a.out), "S2": study2(a.out), "S3": study3(a.out), "S4": study4(a.out), "S5": study5()}
    json.dump(R, open(f"{a.out}/suite_results.json", "w"), indent=1)
    print(json.dumps(R, indent=1)[:6000])

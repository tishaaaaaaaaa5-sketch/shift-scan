#!/usr/bin/env python
"""seed_variance_study.py -- how much of a quantization/QAT "gain" survives a different seed?

Subcommands
  run       train small quantized ResNet-20 models on CIFAR-10 for many seeds (resumable, appends JSONL)
  analyze   seed-variance statistics + figures + report from a run file OR any CSV of (method, seed, metric)
  selftest  check the statistics code on synthetic numbers (needs numpy/scipy/pandas/matplotlib only)

Design: weight-only QAT (first conv and final FC stay FP32), per-tensor scales, SGD+cosine, final-epoch test
accuracy (no checkpoint selection on the test set). The seed controls init AND data order/augmentation.
`--reps 2` repeats the same seed to separate GPU nondeterminism from seed variance.

  python seed_variance_study.py run --methods fp32,uniform4,lsq4,uniform2,lsq2,bitnet158,twn --seeds 0-4
  python seed_variance_study.py analyze --input results/cifar_runs.jsonl --metric test_acc --higher-better
  python seed_variance_study.py analyze --input my.csv --method-col variant --metric student_ppl --lower-better
Nothing here produces results by itself: every number comes from runs you execute. `--synthetic` is a pipeline
test on random data; its rows are flagged and `analyze` refuses them unless --allow-synthetic.
"""
import argparse, itertools, json, math, os, platform, sys, tempfile, time
import numpy as np

METHODS = {"fp32": None, "uniform4": ("uniform", 4), "lsq4": ("lsq", 4), "uniform2": ("uniform", 2),
           "lsq2": ("lsq", 2), "bitnet158": ("bitnet158", 0), "twn": ("twn", 0)}
# claim templates (A vs B) in the style of published comparisons; verdicts are about seed noise only
CLAIMS = [("lsq2", "uniform2", "learned step size beats fixed max-abs scale at 2 bits"),
          ("lsq4", "uniform4", "learned step size beats fixed max-abs scale at 4 bits"),
          ("bitnet158", "twn", "absmean ternary vs threshold ternary (TWN)"),
          ("lsq4", "fp32", "4-bit QAT matches FP32"),
          ("lsq2", "bitnet158", "2-bit LSQ vs 1.58-bit absmean")]


def parse_seeds(s):
    out = []
    for part in str(s).split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip(): out.append(int(part))
    return out


# =============================== statistics ===============================
def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (m - rank))); adj[i] = run
    return adj


def boot_ci(x, rng, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def pooled_sd(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    return math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))


def perm_p(a, b, rng, n=20000):
    """two-sided permutation p for difference of means; exact when the number of splits is small"""
    a, b = np.asarray(a, float), np.asarray(b, float); obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    if math.comb(len(pool), na) <= n:
        d = [abs(pool[list(i)].mean() - np.delete(pool, list(i)).mean()) for i in itertools.combinations(range(len(pool)), na)]
        return float((np.array(d) >= obs - 1e-12).mean())
    d = np.empty(n)
    for k in range(n):
        rng.shuffle(pool); d[k] = abs(pool[:na].mean() - pool[na:].mean())
    return float(((d >= obs - 1e-12).sum() + 1) / (n + 1))


def n_for_power(delta, sd, alpha=0.05, power=0.8):
    """seeds per method for a two-sample comparison (normal approximation; optimistic for small n)"""
    from scipy.stats import norm
    if not np.isfinite(sd) or abs(delta) < 1e-12: return np.inf
    return max(3, int(math.ceil(2 * ((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / abs(delta)) ** 2)))   # never below 3


def flip_prob(a, b):
    """P(one random seed of A vs one random seed of B gives the opposite sign to the n-seed mean difference)"""
    d = np.subtract.outer(np.asarray(a, float), np.asarray(b, float)); m = np.mean(a) - np.mean(b)
    if m == 0: return 0.5
    s = np.sign(m); return float(((np.sign(d) != s) * (np.sign(d) != 0) + 0.5 * (d == 0)).mean())


def pairwise_table(groups, orient, rng):
    from scipy.stats import ttest_ind, ttest_rel
    rows = []
    for A, B in itertools.combinations(groups, 2):
        a, b = groups[A], groups[B]; va, vb = a.values, b.values
        r = dict(A=A, B=B, n_A=len(va), n_B=len(vb))
        if len(va) < 2 or len(vb) < 2: rows.append(r); continue
        adv = orient * (va.mean() - vb.mean()); sd = pooled_sd(va, vb)
        common = a.index.intersection(b.index); d = orient * (a[common].values - b[common].values)
        lo, hi = boot_ci(d, rng) if len(d) >= 2 else (np.nan, np.nan)
        single = orient * np.subtract.outer(va, vb).ravel()
        r.update(adv_A_over_B=adv, pooled_sd=sd, gap_over_sd=adv / sd if sd > 0 else np.inf,
                 welch_p=float(ttest_ind(va, vb, equal_var=False).pvalue), perm_p=perm_p(va, vb, rng),
                 flip_single_seed=flip_prob(va, vb) if orient > 0 else flip_prob(-va, -vb),
                 seeds_for_80pct_power=n_for_power(adv, sd), single_gap_p05=np.percentile(single, 5),
                 single_gap_p95=np.percentile(single, 95), paired_n=len(common), paired_mean=d.mean() if len(d) else np.nan,
                 paired_ci_lo=lo, paired_ci_hi=hi,
                 paired_p=float(ttest_rel(a[common].values, b[common].values).pvalue) if len(common) >= 3 else np.nan)
        rows.append(r)
    import pandas as pd
    t = pd.DataFrame(rows)
    if "perm_p" in t:
        ok = t.perm_p.notna(); t.loc[ok, "perm_p_holm"] = holm(t.loc[ok, "perm_p"].values)
    return t


# =============================== analyze ===============================
def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def analyze(a):
    import pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.read_json(a.input, lines=True) if a.input.endswith(".jsonl") else pd.read_csv(a.input)
    if "data" in df and (df["data"] == "synthetic").any() and not a.allow_synthetic:
        sys.exit("File contains synthetic-data rows (pipeline tests). Remove them or pass --allow-synthetic.")
    df = df.rename(columns={a.method_col: "method"})
    if "rep" not in df: df["rep"] = 0
    orient = 1 if a.higher_better else -1
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
    S = df.groupby(["method", "seed"])[a.metric].mean().reset_index()
    groups = {m: g.set_index("seed")[a.metric].sort_index() for m, g in S.groupby("method")}
    summ = pd.DataFrame([dict(method=m, n_seeds=len(g), mean=g.mean(), sd=g.std(ddof=1) if len(g) > 1 else np.nan,
                              min=g.min(), max=g.max(), range=g.max() - g.min(),
                              ci_lo=boot_ci(g.values, rng)[0], ci_hi=boot_ci(g.values, rng)[1]) for m, g in groups.items()])
    pw = pairwise_table(groups, orient, rng)
    summ.to_csv(f"{a.out}/summary.csv", index=False); pw.to_csv(f"{a.out}/pairwise.csv", index=False)

    rep = [f"# Seed-variance report: `{a.metric}` ({'higher' if a.higher_better else 'lower'} is better)\n",
           f"Source: `{a.input}`. Seed-level values (reps averaged). **n per method: {({k: int(v) for k, v in summ.set_index('method').n_seeds.items()})}**\n"]
    if summ.n_seeds.min() < 5:
        rep.append("> WARNING: fewer than 5 seeds for some methods. sd, p-values and power numbers are unstable at this n; "
                   "report every individual value and avoid the word 'significant'.\n")
    rep += ["## Per-method seed spread\n", md_table(summ), "\n"]
    if "adv_A_over_B" in pw:
        t = pw.dropna(subset=["adv_A_over_B"])
        rep += ["## Pairwise comparisons\n",
                "`adv_A_over_B` is oriented so positive = A better. `flip_single_seed` = chance that a one-seed-each comparison "
                "reverses the sign of the n-seed mean gap. `seeds_for_80pct_power` is a normal-approximation estimate.\n",
                md_table(t[["A", "B", "adv_A_over_B", "pooled_sd", "gap_over_sd", "perm_p", "perm_p_holm", "flip_single_seed",
                            "single_gap_p05", "single_gap_p95", "seeds_for_80pct_power"]]), "\n"]
        rep.append("## Claim templates\n")
        for A, B, txt in CLAIMS:
            row = t[((t.A == A) & (t.B == B)) | ((t.A == B) & (t.B == A))]
            if row.empty: continue
            r = row.iloc[0]; adv = r.adv_A_over_B if r.A == A else -r.adv_A_over_B; flip = r.flip_single_seed
            nmin = min(r.n_A, r.n_B)
            if nmin < 3: v = "n too small for a verdict"
            elif r.perm_p_holm < 0.05 and adv > 0: v = "gap exceeds seed noise"
            elif r.perm_p_holm < 0.05: v = "gap exceeds seed noise but has the OPPOSITE sign"
            else: v = f"not distinguishable from seed noise (needs ~{r.seeds_for_80pct_power} seeds/method at this gap)"
            rep.append(f"- **{A} vs {B}** ({txt}): advantage {adv:+.3g}, single-seed flip chance {flip:.0%} -> {v}")
        rep.append("\nA 'matches' claim is an equivalence claim: judge it from the paired CI in `pairwise.csv`, not from p.\n")
    # nondeterminism: spread between reps of the same seed vs spread across seeds
    if df.groupby(["method", "seed"]).rep.nunique().max() > 1:
        rows = []
        for m, g in df.groupby("method"):
            within = [x[a.metric].var(ddof=1) for _, x in g.groupby("seed") if len(x) > 1]
            if within: rows.append(dict(method=m, sd_same_seed=math.sqrt(np.mean(within)),
                                        sd_across_seeds=summ.set_index("method").loc[m, "sd"]))
        rep += ["## Same-seed repeats (implementation nondeterminism) vs across-seed spread\n", md_table(pd.DataFrame(rows)), "\n"]
    open(f"{a.out}/report.md", "w").write("\n".join(rep))

    # figures
    ms = list(summ.method); fig, ax = plt.subplots(figsize=(1.1 * len(ms) + 3, 4))
    for i, m in enumerate(ms):
        y = groups[m].values; ax.scatter(i + rng.uniform(-.12, .12, len(y)), y, alpha=.7)
        ci = summ.set_index("method").loc[m]; ax.errorbar(i, ci["mean"], yerr=[[ci["mean"] - ci.ci_lo], [ci.ci_hi - ci["mean"]]],
                                                          color="k", capsize=4, marker="_", ms=14)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=30); ax.set_ylabel(a.metric)
    ax.set_title("each dot = one seed; bar = mean with bootstrap 95% CI"); plt.tight_layout(); plt.savefig(f"{a.out}/fig_seeds.png", dpi=140); plt.close()
    if "flip_single_seed" in pw and pw.flip_single_seed.notna().any():
        M = pd.DataFrame(np.nan, index=ms, columns=ms)
        for _, r in pw.dropna(subset=["flip_single_seed"]).iterrows(): M.loc[r.A, r.B] = M.loc[r.B, r.A] = r.flip_single_seed
        fig, ax = plt.subplots(figsize=(1 * len(ms) + 3, .9 * len(ms) + 2)); im = ax.imshow(M.values.astype(float), vmin=0, vmax=.5, cmap="viridis_r")
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=45); ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms)
        for i in range(len(ms)):
            for j in range(len(ms)):
                if np.isfinite(M.values[i, j]): ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, label="P(single-seed comparison reverses mean ordering)"); plt.tight_layout()
        plt.savefig(f"{a.out}/fig_flip_matrix.png", dpi=140); plt.close()
    print("\n".join(rep)); print(f"\nwrote {a.out}/ (report.md, summary.csv, pairwise.csv, fig_*.png)")


# =============================== training (needs torch + torchvision) ===============================
def run(a):
    import torch, torch.nn as nn, torch.nn.functional as F
    if a.deterministic: os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu" and not a.synthetic: sys.exit("No GPU. Use a GPU runtime (or --synthetic for a CPU pipeline test).")
    if a.deterministic:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else: torch.backends.cudnn.benchmark = True

    def ste(q, w): return w + (q - w).detach()
    def round_ste(x): return (x.round() - x).detach() + x
    def grad_scale(x, s): return (x - x * s).detach() + x * s

    class Uniform(nn.Module):
        def __init__(s, bits): super().__init__(); s.Qn, s.Qp = 2 ** (bits - 1), 2 ** (bits - 1) - 1
        def forward(s, w):
            sc = w.detach().abs().max().clamp_min(1e-8) / s.Qp
            return ste(torch.clamp(torch.round(w / sc), -s.Qn, s.Qp) * sc, w)

    class LSQ(nn.Module):
        def __init__(s, w, bits):
            super().__init__(); s.Qn, s.Qp = 2 ** (bits - 1), 2 ** (bits - 1) - 1
            s.step = nn.Parameter(2 * w.abs().mean() / math.sqrt(s.Qp))
        def forward(s, w):
            st = grad_scale(s.step, 1.0 / math.sqrt(w.numel() * s.Qp))
            return round_ste(torch.clamp(w / st, -s.Qn, s.Qp)) * st

    class Absmean158(nn.Module):
        def forward(s, w):
            g = w.detach().abs().mean().clamp_min(1e-5)
            return ste(torch.clamp(torch.round(w / g), -1, 1) * g, w)

    class TWN(nn.Module):
        def forward(s, w):
            wa = w.detach().abs(); mask = (wa > 0.7 * wa.mean()).float()
            alpha = (wa * mask).sum() / mask.sum().clamp_min(1)
            return ste(torch.sign(w) * mask * alpha, w)

    def make_q(spec, w):
        kind, bits = spec
        return {"uniform": lambda: Uniform(bits), "lsq": lambda: LSQ(w, bits),
                "bitnet158": Absmean158, "twn": TWN}[kind]()

    class QConv(nn.Conv2d):
        def __init__(s, *args, **kw):
            super().__init__(*args, **kw); s.q = None          # instance attr (not class attr) so a module can replace it
        def forward(s, x):
            return F.conv2d(x, s.q(s.weight) if s.q is not None else s.weight, None, s.stride, s.padding)

    class Block(nn.Module):
        def __init__(s, cin, cout, stride):
            super().__init__()
            s.c1 = QConv(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False); s.b1 = nn.BatchNorm2d(cout)
            s.c2 = QConv(cout, cout, kernel_size=3, stride=1, padding=1, bias=False); s.b2 = nn.BatchNorm2d(cout)
            s.sc = None if stride == 1 and cin == cout else nn.Sequential(
                QConv(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False), nn.BatchNorm2d(cout))
        def forward(s, x):
            o = s.b2(s.c2(F.relu(s.b1(s.c1(x)))))
            return F.relu(o + (x if s.sc is None else s.sc(x)))

    class ResNet20(nn.Module):
        def __init__(s, spec):
            super().__init__()
            s.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); s.bn1 = nn.BatchNorm2d(16)
            cfg = [(16, 16, 1)] * 3 + [(16, 32, 2), (32, 32, 1), (32, 32, 1)] + [(32, 64, 2), (64, 64, 1), (64, 64, 1)]
            s.blocks = nn.Sequential(*[Block(*c) for c in cfg]); s.fc = nn.Linear(64, 10)
            for m in s.modules():
                if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if spec is not None:                                   # attach AFTER init so LSQ steps see the real weights
                for m in s.modules():
                    if isinstance(m, QConv): m.q = make_q(spec, m.weight.data)
        def forward(s, x):
            x = s.blocks(F.relu(s.bn1(s.conv1(x))))
            return s.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def load_data():
        if a.synthetic:
            g = torch.Generator().manual_seed(1234)
            xtr = torch.randint(0, 256, (2048, 3, 32, 32), dtype=torch.uint8, generator=g); ytr = torch.randint(0, 10, (2048,), generator=g)
            xte = torch.randint(0, 256, (512, 3, 32, 32), dtype=torch.uint8, generator=g); yte = torch.randint(0, 10, (512,), generator=g)
        else:
            import torchvision
            tr = torchvision.datasets.CIFAR10(a.data, train=True, download=True); te = torchvision.datasets.CIFAR10(a.data, train=False, download=True)
            xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous(); ytr = torch.tensor(tr.targets)
            xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous(); yte = torch.tensor(te.targets)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1) * 255; std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1) * 255
        nz = lambda x: ((x.float() - mean) / std)
        xtr = F.pad(nz(xtr), (4, 4, 4, 4)).half().to(dev); xte = nz(xte).half().to(dev)
        return xtr, ytr.to(dev), xte, yte.to(dev)

    def batches(xp, y, bs, gen):
        N = xp.shape[0]; perm = torch.randperm(N, generator=gen, device=dev); ar = torch.arange(32, device=dev)
        for i in range(0, N - bs + 1, bs):
            idx = perm[i:i + bs]; B = idx.numel()
            dy = torch.randint(0, 9, (B,), generator=gen, device=dev); dx = torch.randint(0, 9, (B,), generator=gen, device=dev)
            rows = (dy[:, None] + ar)[:, None, :, None]; cols = (dx[:, None] + ar)[:, None, None, :]
            xb = xp[idx][torch.arange(B, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], rows, cols]
            flip = torch.rand(B, generator=gen, device=dev) < 0.5
            yield torch.where(flip[:, None, None, None], xb.flip(3), xb).float(), y[idx]

    @torch.no_grad()
    def evaluate(model, xte, yte):
        model.eval(); ok = 0
        for i in range(0, xte.shape[0], 1000):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                ok += (model(xte[i:i + 1000].float()).argmax(1) == yte[i:i + 1000]).sum().item()
        return 100.0 * ok / xte.shape[0]

    def one(method, seed, data):
        xtr, ytr, xte, yte = data
        import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if dev.type == "cuda": torch.cuda.manual_seed_all(seed)
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        model = ResNet20(METHODS[method]).to(dev)
        dec = [p for n, p in model.named_parameters() if p.ndim > 1]
        nod = [p for n, p in model.named_parameters() if p.ndim <= 1]          # BN params, biases, LSQ steps
        opt = torch.optim.SGD([{"params": dec, "weight_decay": a.wd}, {"params": nod, "weight_decay": 0.0}],
                              lr=a.lr, momentum=0.9, nesterov=True)
        spe = xtr.shape[0] // a.batch; total = a.epochs * spe; warm = max(1, spe)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); t0 = time.time(); last = 0.0
        for ep in range(a.epochs):
            model.train(); tot = torch.zeros((), device=dev); n = 0
            for xb, yb in batches(xtr, ytr, a.batch, gen):
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.detach(); n += 1
            last = (tot / max(n, 1)).item()
        return dict(test_acc=evaluate(model, xte, yte), train_loss=last, time_s=round(time.time() - t0, 1))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["method"], r["seed"], r["rep"], r["epochs"], r["data"]))
    seeds = parse_seeds(a.seeds); methods = a.methods.split(","); data_name = "synthetic" if a.synthetic else "cifar10"
    bad = [m for m in methods if m not in METHODS]
    if bad: sys.exit(f"unknown methods {bad}; choose from {list(METHODS)}")
    jobs = [(m, s, r) for s in seeds for r in range(a.reps) for m in methods]       # seed-major: partial results = complete seeds
    todo = [j for j in jobs if (j[0], j[1], j[2], a.epochs, data_name) not in done]
    print(f"{len(todo)} of {len(jobs)} runs to do on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})", flush=True)
    data = load_data(); t_start = time.time()
    for k, (m, s, r) in enumerate(todo):
        res = one(m, s, data)
        row = dict(method=m, seed=s, rep=r, epochs=a.epochs, data=data_name, lr=a.lr, batch=a.batch, wd=a.wd,
                   deterministic=a.deterministic, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                   torch=torch.__version__, python=platform.python_version(), **res)
        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
        el = time.time() - t_start
        print(f"[{k + 1}/{len(todo)}] {m} seed={s} rep={r}: acc {res['test_acc']:.2f}  ({res['time_s']}s)  eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min", flush=True)


# =============================== selftest ===============================
def selftest():
    import pandas as pd
    rng = np.random.default_rng(1)
    assert parse_seeds("0-2,5") == [0, 1, 2, 5]
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]); print("[ok] Holm correction")
    a = rng.normal(0, 1, 8) + 3; b = rng.normal(0, 1, 8)
    assert perm_p(a, b, rng) < 0.01; same = perm_p(rng.normal(0, 1, 8), rng.normal(0, 1, 8), rng); assert same > 0.05
    print("[ok] permutation p separates a 3-sd gap from no gap")
    assert flip_prob([10, 10.1, 9.9], [0, 0.1, -0.1]) == 0.0 and flip_prob([1, 3], [2, 4]) > 0
    print("[ok] single-seed flip probability")
    assert n_for_power(1, 1) == 16 and n_for_power(2, 1) < n_for_power(1, 1) and n_for_power(0, 1) == np.inf
    print("[ok] seeds-for-power:", n_for_power(1, 1), "per group for gap = 1 sd")
    d = tempfile.mkdtemp(); rows = []
    for s in range(6):                                          # SYNTHETIC numbers, only to exercise the code
        rows += [dict(method="A", seed=s, rep=0, acc=90 + rng.normal(0, .3)), dict(method="B", seed=s, rep=0, acc=89 + rng.normal(0, .3)),
                 dict(method="C", seed=s, rep=0, acc=89.1 + rng.normal(0, .3)), dict(method="A", seed=s, rep=1, acc=90 + rng.normal(0, .3))]
    pd.DataFrame(rows).to_csv(f"{d}/t.csv", index=False)
    analyze(argparse.Namespace(input=f"{d}/t.csv", metric="acc", higher_better=True, method_col="method", out=f"{d}/o", allow_synthetic=True))
    for f in ("report.md", "summary.csv", "pairwise.csv", "fig_seeds.png", "fig_flip_matrix.png"): assert os.path.exists(f"{d}/o/{f}"), f
    pw = pd.read_csv(f"{d}/o/pairwise.csv").set_index(["A", "B"])
    assert pw.loc[("A", "B"), "perm_p"] < 0.05 and pw.loc[("B", "C"), "perm_p"] > 0.05
    print("\n[ok] analyze on synthetic numbers: files written, A>B detected, B~C not\nselftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--methods", default="fp32,uniform4,lsq4,uniform2,lsq2,bitnet158,twn")
    r.add_argument("--seeds", default="0-4"); r.add_argument("--reps", type=int, default=1); r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--batch", type=int, default=128); r.add_argument("--lr", type=float, default=0.1); r.add_argument("--wd", type=float, default=5e-4)
    r.add_argument("--data", default="./data"); r.add_argument("--out", default="results/cifar_runs.jsonl")
    r.add_argument("--deterministic", action="store_true"); r.add_argument("--synthetic", action="store_true")
    an = sp.add_parser("analyze"); an.add_argument("--input", required=True); an.add_argument("--metric", default="test_acc")
    g = an.add_mutually_exclusive_group(required=True); g.add_argument("--higher-better", dest="higher_better", action="store_true")
    g.add_argument("--lower-better", dest="higher_better", action="store_false")
    an.add_argument("--method-col", default="method"); an.add_argument("--out", default="results/analysis"); an.add_argument("--allow-synthetic", action="store_true")
    sp.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "analyze": analyze, "selftest": lambda _: selftest()}[a.cmd](a)

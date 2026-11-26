"""Create/run a reproducible Shift-Scan ablation suite.

Generate commands only (safe default):
  python run_ablation.py --tokens 2000000 --out-dir runs/ablation --print
Execute sequentially:
  python run_ablation.py --tokens 2000000 --out-dir runs/ablation --run
Use a small budget only for smoke tests; final research should use matched budgets
and >=3 seeds per condition.
"""
import argparse, os, shlex, subprocess, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=float, default=100e6)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0, help="base seed used when --seeds is omitted")
    p.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2 for repeated runs")
    p.add_argument("--out-dir", default="runs/ablation")
    p.add_argument("--pretrained", default="state-spaces/mamba2-130m")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--micro-bs", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--run", action="store_true", help="actually execute all jobs sequentially")
    p.add_argument("--print", action="store_true", help="print shell-escaped commands")
    a = p.parse_args()
    # Matched configurations isolate the effects of decay quantization,
    # ternary projections, HARPQ and state-aware distillation.
    configs = [
        ("A_fp_reference", ["--decay", "exp", "--ternary", "0", "--state-weight", "0"]),
        ("B_pot_nearest", ["--decay", "shift", "--decay-quantizer", "nearest", "--ternary", "0"]),
        ("C_ternary_only", ["--decay", "exp", "--ternary", "1", "--state-weight", "0.05"]),
        ("D_joint_nearest", ["--decay", "shift", "--decay-quantizer", "nearest", "--ternary", "1", "--state-weight", "0.05"]),
        ("E_harpq_h32", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "32", "--decay-lag-weight", "uniform", "--ternary", "1", "--state-weight", "0.05"]),
        ("F_harpq_h64", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "64", "--decay-lag-weight", "uniform", "--ternary", "1", "--state-weight", "0.05"]),
        ("G_harpq_h128", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "128", "--decay-lag-weight", "uniform", "--ternary", "1", "--state-weight", "0.05"]),
        ("H_harpq_h256", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "256", "--decay-lag-weight", "uniform", "--ternary", "1", "--state-weight", "0.05"]),
        ("I_harpq_activation_aware", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "128", "--ternary", "1", "--activation-aware", "1", "--state-weight", "0.05"]),
        ("J_harpq_no_state_kd", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "128", "--ternary", "1", "--activation-aware", "1", "--state-weight", "0", "--state-loss-type", "hybrid"]),
        ("K_harpq_cosine_state_kd", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "128", "--ternary", "1", "--activation-aware", "1", "--state-weight", "0.05", "--state-loss-type", "cosine"]),
        ("L_harpq_group4", ["--decay", "shift", "--decay-quantizer", "horizon", "--decay-horizon", "128", "--decay-group-size", "4", "--ternary", "1", "--state-weight", "0.05"]),
        ("M_dyadic2_h128", ["--decay", "shift", "--decay-quantizer", "dyadic", "--dyadic-terms", "2", "--decay-horizon", "128", "--ternary", "1", "--activation-aware", "1", "--state-weight", "0.05"]),
        ("N_dyadic3_h128", ["--decay", "shift", "--decay-quantizer", "dyadic", "--dyadic-terms", "3", "--decay-horizon", "128", "--ternary", "1", "--activation-aware", "1", "--state-weight", "0.05"]),
    ]
    os.makedirs(a.out_dir, exist_ok=True)
    commands = []
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    for seed in seeds:
        for name, opts in configs:
            run_name = f"{name}_seed{seed}"
            cmd = [sys.executable, "train_qat.py", "--name", run_name, "--out", a.out_dir,
                   "--pretrained", a.pretrained, "--tokens", str(a.tokens), "--seq-len", str(a.seq_len),
                   "--seed", str(seed), "--dtype", a.dtype, "--micro-bs", str(a.micro_bs),
                   "--accum", str(a.accum)] + opts
            commands.append(cmd)
    for cmd in commands:
        print(shlex.join(cmd))
        if a.run:
            subprocess.run(cmd, check=True)
    if not a.run and not a.print:
        print("No jobs were launched. Add --run to execute; --print only displays commands.")


if __name__ == "__main__":
    main()

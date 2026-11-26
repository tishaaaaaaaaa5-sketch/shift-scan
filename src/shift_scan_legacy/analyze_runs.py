"""
Aggregate Shift-Scan QAT logs into a reproducible CSV summary.

Example:
  python analyze_runs.py --runs runs --eval-dir results --out results/ablation_summary.csv

The script never fabricates missing metrics: absent evaluation JSON fields remain blank.
"""
import argparse
import csv
import glob
import json
import os


def read_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--eval-dir", default="results")
    ap.add_argument("--out", default="results/ablation_summary.csv")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs, "*"))):
        name = os.path.basename(run_dir)
        log_path = os.path.join(run_dir, "log.jsonl")
        if not os.path.isfile(log_path):
            continue
        records = [json.loads(line) for line in open(log_path) if line.strip()]
        eval_records = [r for r in records if "student_ppl" in r]
        latest = eval_records[-1] if eval_records else {}
        best = min(eval_records, key=lambda r: r["student_ppl"]) if eval_records else {}
        config = read_json(os.path.join(run_dir, "args.json"))
        size = read_json(os.path.join(run_dir, "size_report.json"))
        evaluation = read_json(os.path.join(args.eval_dir, f"{name}.json"))
        rows.append({
            "run": name,
            "decay": config.get("decay", ""),
            "decay_quantizer": config.get("decay_quantizer", ""),
            "decay_horizon": config.get("decay_horizon", ""),
            "decay_lag_weight": config.get("decay_lag_weight", ""),
            "ternary": config.get("ternary", ""),
            "activation_aware": config.get("activation_aware", ""),
            "decay_group_size": config.get("decay_group_size", ""),
            "state_weight": config.get("state_weight", ""),
            "state_loss_type": config.get("state_loss_type", ""),
            "tokens_requested": config.get("tokens", ""),
            "seed": config.get("seed", ""),
            "latest_val_ppl": latest.get("student_ppl", ""),
            "teacher_val_ppl": latest.get("teacher_ppl", ""),
            "best_val_ppl": best.get("student_ppl", ""),
            "state_mse": latest.get("state_mse", ""),
            "k_mean": latest.get("k_mean", ""),
            "frac_at_kmin": latest.get("frac_at_kmin", ""),
            "frac_at_kmax": latest.get("frac_at_kmax", ""),
            "packed_size_MB_estimate": size.get("packed_MB", ""),
            "fp16_size_MB": size.get("fp16_MB", ""),
            "eval_tasks_present": ",".join(sorted(evaluation.get("results", {}).keys()))
                if isinstance(evaluation.get("results"), dict) else "",
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not rows:
        raise SystemExit(f"No run logs found under {args.runs!r}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} run summaries -> {args.out}")


if __name__ == "__main__":
    main()

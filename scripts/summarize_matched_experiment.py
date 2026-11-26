#!/usr/bin/env python3
"""Summarize per-seed lm-evaluation-harness JSON files."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.input_dir)
    groups = {"A_full_precision_student": [], "B_uniform_ternary": [],
              "C_sensitivity_mixed": [], "D_random_mixed": []}
    for path in sorted(root.glob("*.json")):
        if path.name in {"layer_selection.json", "summary.json"}:
            continue
        if path.name.startswith("A_full_precision_student_s"):
            key = "A_full_precision_student"
        elif path.name.startswith("B_uniform_ternary_s"):
            key = "B_uniform_ternary"
        elif path.name.startswith("C_sensitivity_mixed_s"):
            key = "C_sensitivity_mixed"
        elif path.name.startswith("D_random_mixed_s"):
            key = "D_random_mixed"
        else:
            continue
        groups[key].append(json.loads(path.read_text()))

    summary = {"conditions": {}, "comparisons": {
        "ternary_gap": "B_uniform_ternary minus A_full_precision_student",
        "sensitivity_benefit": "C_sensitivity_mixed minus B_uniform_ternary",
        "sensitivity_vs_random": "C_sensitivity_mixed minus D_random_mixed",
    }}
    for key, runs in groups.items():
        if not runs:
            summary["conditions"][key] = {"n": 0, "status": "missing"}
            continue
        task_names = set.intersection(*(set(run.get("results", {})) for run in runs))
        metrics = {}
        for task in sorted(task_names):
            metric_names = set.intersection(*(set(run["results"][task]) for run in runs))
            metrics[task] = {}
            for metric in sorted(metric_names):
                vals = [run["results"][task][metric] for run in runs]
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                    metrics[task][metric] = {
                        "mean": statistics.mean(vals),
                        "stdev": statistics.stdev(vals) if len(vals) > 1 else None,
                        "n": len(vals),
                        "values": vals,
                    }
        summary["conditions"][key] = {"n": len(runs), "metrics": metrics}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()

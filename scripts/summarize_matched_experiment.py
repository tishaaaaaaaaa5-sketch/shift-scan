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
    seed_groups = {key: {} for key in groups}
    for key, runs in groups.items():
        if not runs:
            summary["conditions"][key] = {"n": 0, "status": "missing"}
            continue
        metrics = {}
        for task in sorted(set.intersection(*(set(run.get("results", {})) for run in runs))):
            names = set.intersection(*(set(run["results"][task]) for run in runs))
            metrics[task] = {}
            for metric in sorted(names):
                vals = [run["results"][task][metric] for run in runs]
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                    metrics[task][metric] = {
                        "mean": statistics.mean(vals),
                        "stdev": statistics.stdev(vals) if len(vals) > 1 else None,
                        "n": len(vals),
                        "values": vals,
                    }
        summary["conditions"][key] = {"n": len(runs), "metrics": metrics}
        for path in sorted(root.glob(key + "_s*.json")):
            try:
                seed = int(path.stem.rsplit("_s", 1)[1])
            except ValueError:
                continue
            seed_groups[key][seed] = json.loads(path.read_text())

    # Paired per-seed deltas are more informative than subtracting condition means.
    pair_specs = {
        "ternary_gap": ("B_uniform_ternary", "A_full_precision_student"),
        "sensitivity_benefit": ("B_uniform_ternary", "C_sensitivity_mixed"),
        "sensitivity_vs_random": ("D_random_mixed", "C_sensitivity_mixed"),
    }
    for label, (left, right) in pair_specs.items():
        common_seeds = sorted(set(seed_groups[left]) & set(seed_groups[right]))
        deltas = {}
        if common_seeds:
            task_sets = [set(seed_groups[left][s].get("results", {})) &
                         set(seed_groups[right][s].get("results", {})) for s in common_seeds]
            for task in sorted(set.intersection(*task_sets)):
                metric_sets = [set(seed_groups[left][s]["results"][task]) &
                               set(seed_groups[right][s]["results"][task]) for s in common_seeds]
                for metric in sorted(set.intersection(*metric_sets)):
                    pairs = []
                    for seed in common_seeds:
                        lv = seed_groups[left][seed]["results"][task][metric]
                        rv = seed_groups[right][seed]["results"][task][metric]
                        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)) and math.isfinite(lv) and math.isfinite(rv):
                            pairs.append({"seed": seed, "delta": rv - lv})
                    if pairs:
                        deltas.setdefault(task, {})[metric] = {
                            "mean_delta": statistics.mean(x["delta"] for x in pairs),
                            "stdev_delta": statistics.stdev(x["delta"] for x in pairs) if len(pairs) > 1 else None,
                            "paired_seeds": pairs,
                            "n": len(pairs),
                        }
        summary["comparisons"][label] = {
            "definition": f"{right} minus {left}",
            "n_paired_seeds": len(common_seeds),
            "paired_deltas": deltas,
        }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()

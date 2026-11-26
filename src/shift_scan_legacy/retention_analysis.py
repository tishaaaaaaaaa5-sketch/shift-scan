"""Summarize calibration JSONs from calibrate_decay.py; never fabricates results."""
import argparse, csv, json, os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for path in a.inputs:
        d = json.load(open(path))
        rows.append({
            "file": os.path.basename(path),
            "model": d.get("source", {}).get("model", ""),
            "dataset": d.get("source", {}).get("dataset", ""),
            "tokens_seen": d.get("source", {}).get("tokens_seen", ""),
            "horizon": d.get("method", {}).get("horizon", ""),
            "nearest_retention_rmse": d.get("nearest_one_step", {}).get("retention_rmse", ""),
            "harpq_retention_rmse": d.get("horizon_aware", {}).get("retention_rmse", ""),
            "objective_reduction": d.get("objective_cost", {}).get("relative_reduction", ""),
            "fraction_k_changed": d.get("fraction_exponents_changed_vs_nearest", ""),
        })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} calibration summaries to {a.out}")


if __name__ == "__main__":
    main()

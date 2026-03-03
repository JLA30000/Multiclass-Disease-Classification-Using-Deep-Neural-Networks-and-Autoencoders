import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from typing import List


METRIC_RE = re.compile(r"GRID_METRIC val_macro_f1 ([0-9]*\.?[0-9]+)")


def parse_list(raw: str, cast_fn):
    vals = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if token:
            vals.append(cast_fn(token))
    if not vals:
        raise ValueError(f"Failed to parse values from: {raw!r}")
    return vals


def run_id_value(v):
    return str(v).replace(".", "p")


def build_run_id(lr, md, sub, col, lam):
    return (
        f"lr{run_id_value(lr)}_md{run_id_value(md)}_"
        f"sub{run_id_value(sub)}_col{run_id_value(col)}_lam{run_id_value(lam)}"
    )


def run_one_config(args, lr, md, sub, col, lam):
    run_id = build_run_id(lr, md, sub, col, lam)
    log_file = os.path.join(args.outdir, f"{run_id}.log")

    cmd = [
        sys.executable,
        args.script_path,
        "--data_path",
        args.data_path,
        "--indices_file",
        args.indices_file,
        "--seeds",
        args.seeds,
        "--early_stopping_rounds",
        str(args.early_stopping_rounds),
        "--n_estimators",
        str(args.n_estimators),
        "--learning_rate",
        str(lr),
        "--max_depth",
        str(md),
        "--min_child_weight",
        str(args.min_child_weight),
        "--gamma",
        str(args.gamma),
        "--subsample",
        str(sub),
        "--colsample_bytree",
        str(col),
        "--reg_lambda",
        str(lam),
        "--reg_alpha",
        str(args.reg_alpha),
        "--tree_method",
        args.tree_method,
        "--device",
        args.device,
        "--n_jobs",
        str(args.n_jobs),
    ]

    if args.no_plots:
        cmd.append("--no_plots")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write("\n")
        f.write(proc.stderr)

    metric_match = METRIC_RE.search(proc.stdout or "")
    metric = float(metric_match.group(1)) if metric_match else None

    row = {
        "run_id": run_id,
        "learning_rate": lr,
        "max_depth": md,
        "subsample": sub,
        "colsample_bytree": col,
        "reg_lambda": lam,
        "val_macro_f1": metric,
        "return_code": proc.returncode,
        "log_file": log_file,
    }
    return row


def write_csv(path: str, rows: List[dict]):
    fieldnames = [
        "run_id",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "val_macro_f1",
        "return_code",
        "log_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        "Kaggle/Linux-safe grid search wrapper for xgboost_weighted.py"
    )
    parser.add_argument("--script_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--indices_file", required=True)
    parser.add_argument("--outdir", default="/kaggle/working/runs_xgb_tuning")

    parser.add_argument("--learning_rates", default="0.05,0.1")
    parser.add_argument("--max_depths", default="5")
    parser.add_argument("--subsamples", default="0.9")
    parser.add_argument("--colsamples", default="0.9")
    parser.add_argument("--reg_lambdas", default="0.5")

    parser.add_argument("--seeds", default="1")
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--n_estimators", type=int, default=4000)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--tree_method", default="hist", choices=["hist", "approx", "exact"])
    parser.add_argument("--device", default="cuda", help="GPU default. Use cpu/auto if needed.")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--no_plots", action="store_true", help="Skip plots for faster tuning.")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    lrs = parse_list(args.learning_rates, float)
    mds = parse_list(args.max_depths, int)
    subs = parse_list(args.subsamples, float)
    cols = parse_list(args.colsamples, float)
    lams = parse_list(args.reg_lambdas, float)

    total = len(lrs) * len(mds) * len(subs) * len(cols) * len(lams)
    print(f"Starting grid with {total} runs")
    print(f"Device default: {args.device}")

    rows = []
    for i, (lr, md, sub, col, lam) in enumerate(
        itertools.product(lrs, mds, subs, cols, lams), start=1
    ):
        run_id = build_run_id(lr, md, sub, col, lam)
        print(f"[{i}/{total}] Running {run_id}")
        row = run_one_config(args, lr, md, sub, col, lam)
        rows.append(row)
        if row["val_macro_f1"] is None:
            print(f"  WARNING: metric missing; return_code={row['return_code']}")
        else:
            print(f"  val_macro_f1={row['val_macro_f1']:.6f}")

    sorted_rows = sorted(
        rows,
        key=lambda r: (-1 if r["val_macro_f1"] is None else 0, -(r["val_macro_f1"] or -1.0)),
    )

    summary_csv = os.path.join(args.outdir, "summary.csv")
    write_csv(summary_csv, sorted_rows)

    valid = [r for r in sorted_rows if r["val_macro_f1"] is not None]
    if not valid:
        print("No valid metric parsed. Check logs in outdir.")
        return

    best = valid[0]
    best_json = os.path.join(args.outdir, "best_run.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBEST RUN")
    print(json.dumps(best, indent=2))
    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved best run: {best_json}")


if __name__ == "__main__":
    main()

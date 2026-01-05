from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from math import sqrt
from typing import Dict, List, Tuple


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: List[float]) -> float:
    m = mean(xs)
    return sqrt(sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="results")
    ap.add_argument("--out", type=str, default="aggregate.json")
    args = ap.parse_args()

    summaries = glob.glob(os.path.join(args.runs_dir, "*", "summary.json"))
    if not summaries:
        raise SystemExit(f"No summary.json files found under {args.runs_dir}")

    # Deduplicate by (env, algo, reward, seed): keep latest by mtime
    latest: Dict[Tuple[str, str, str, int], str] = {}
    for p in summaries:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        key = (s["env_id"], s["algo"], s["reward"], int(s["seed"]))
        if key not in latest or os.path.getmtime(p) > os.path.getmtime(latest[key]):
            latest[key] = p

    grouped = defaultdict(list)
    for key, p in latest.items():
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        grouped[(s["env_id"], s["algo"], s["reward"])].append(s)

    table = []
    for (env_id, algo, reward), rows in sorted(grouped.items()):
        rets = [float(r["final_eval_mean_return"]) for r in rows]
        times = [float(r["train_time_seconds"]) for r in rows]
        entry = {
            "env_id": env_id,
            "algo": algo,
            "reward": reward,
            "n_seeds": len(rows),
            "mean_return": mean(rets),
            "std_return": std(rets) if len(rets) > 1 else 0.0,
            "mean_train_time_s": mean(times),
        }
        table.append(entry)

    out_obj = {"deduped_runs": len(latest), "groups": table}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, sort_keys=False)

    print("Wrote:", args.out)
    for r in table:
        print(f"{r['env_id']:14s} {r['algo']:4s} {r['reward']:2s}  n={r['n_seeds']}  mean±std={r['mean_return']:.2f}±{r['std_return']:.2f}")


if __name__ == "__main__":
    main()

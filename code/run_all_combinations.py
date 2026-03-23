"""
run_all_combinations.py
=======================
Runs all 27 (3^3) knowledge strategy combinations for the Moon Survival
agent teaming experiment.

Knowledge levels per agent: none | quarter | half
3 agents  →  3 x 3 x 3 = 27 combinations

Usage
-----
# Run all 27 combinations with defaults (k=3, 3 iterations, 1 discussion round):
    python run_all_combinations.py

# Custom settings:
    python run_all_combinations.py --k 3 --iterations 3 --discussion-rounds 2

# Only run a subset (e.g. skip combinations where all agents have none):
    python run_all_combinations.py --skip-all-none

# Dry-run: print what would run without calling the API:
    python run_all_combinations.py --dry-run
"""

import argparse
import itertools
import os
import sys

from experiment_runner import run_experiment

LEVELS = ["none", "quarter", "half"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 27 knowledge strategy combinations for Moon Survival."
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of candidate rankings per iteration (default: 3)."
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of big iterations (default: 3)."
    )
    parser.add_argument(
        "--discussion-rounds", type=int, default=1,
        help="Number of discussion rounds per iteration (default: 1)."
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for knowledge assignment (default: 42)."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Directory to save result files."
    )
    parser.add_argument(
        "--skip-all-none", action="store_true",
        help="Skip the [none, none, none] combination (no knowledge at all)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all combinations that would run without calling the API."
    )
    args = parser.parse_args()

    # ── Generate all 27 combinations ─────────────────────────────────────────
    all_combos = list(itertools.product(LEVELS, LEVELS, LEVELS))

    if args.skip_all_none:
        all_combos = [c for c in all_combos if c != ("none", "none", "none")]

    total = len(all_combos)

    print(f"\n{'#'*70}")
    print(f"  FULL COMBINATION SWEEP")
    print(f"  Total strategies : {total}")
    print(f"  k={args.k}  |  iterations={args.iterations}  |  discussion_rounds={args.discussion_rounds}")
    print(f"  Model: {args.model}")
    print(f"{'#'*70}\n")

    if args.dry_run:
        print("  DRY RUN — strategies that would be executed:\n")
        for i, combo in enumerate(all_combos, 1):
            strategy_str = "+".join(combo)
            print(f"  {i:>2}/{total}  {strategy_str}")
        print()
        return

    # ── Run each combination ──────────────────────────────────────────────────
    results_summary = []

    for i, combo in enumerate(all_combos, 1):
        strategy = list(combo)
        strategy_str = "+".join(strategy)
        print(f"\n{'='*70}")
        print(f"  [{i}/{total}]  Strategy: {strategy_str}")
        print(f"{'='*70}")

        try:
            log = run_experiment(
                knowledge_strategy=strategy,
                k=args.k,
                num_iterations=args.iterations,
                num_discussion_rounds=args.discussion_rounds,
                model=args.model,
                knowledge_seed=args.seed,
                output_dir=args.output_dir,
                verbose=True,
            )
            results_summary.append({
                "strategy": strategy_str,
                "best_sad": log.get("best_sad"),
                "final_iteration_best_sad": log.get("final_iteration_best_sad"),
                "status": "OK",
            })
        except Exception as e:
            print(f"\n  ERROR for strategy {strategy_str}: {e}")
            results_summary.append({
                "strategy": strategy_str,
                "best_sad": None,
                "final_iteration_best_sad": None,
                "status": f"ERROR: {e}",
            })

    # ── Print final comparison table ──────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  SWEEP COMPLETE — RESULTS COMPARISON")
    print(f"{'#'*70}")
    print(f"  {'Strategy':<30} {'Best SAD':<12} {'Final Iter Best':<18} Status")
    print(f"  {'-'*65}")
    for row in sorted(results_summary, key=lambda r: (r["best_sad"] is None, r["best_sad"])):
        print(
            f"  {row['strategy']:<30} "
            f"{str(row['best_sad']):<12} "
            f"{str(row['final_iteration_best_sad']):<18} "
            f"{row['status']}"
        )

    # ── Save comparison CSV ───────────────────────────────────────────────────
    import csv
    from datetime import datetime
    csv_name = f"sweep_k{args.k}_iter{args.iterations}_disc{args.discussion_rounds}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "best_sad", "final_iteration_best_sad", "status"])
        writer.writeheader()
        writer.writerows(results_summary)

    print(f"\n  Comparison CSV saved: {csv_path}")


if __name__ == "__main__":
    main()

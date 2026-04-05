"""
run_all_combinations.py
=======================
Runs the full sweep of knowledge configurations for the Moon Survival
agent teaming experiment.

Core sweep: 3 configs × 4 overlaps = 12 conditions (per source variant).

Usage
-----
# Run all 12 core conditions (source=all) with defaults:
    python run_all_combinations.py

# Include source variants (3 × 12 = 36 total):
    python run_all_combinations.py --include-source-variants

# Custom settings:
    python run_all_combinations.py --k 3 --iterations 3 --discussion-rounds 2

# Dry-run: print what would run without calling the API:
    python run_all_combinations.py --dry-run
"""

import argparse
import os
import sys

from experiment_runner import run_experiment

# 3 knowledge configs
CONFIGS = [
    [0, 2, 4],  # Config 1: High diversity
    [1, 2, 3],  # Config 2: Medium diversity
    [2, 2, 2],  # Config 3: Low/equal diversity
]

OVERLAPS = ["nested", "disjoint", "O3", "O4"]
SOURCES_CORE = ["all"]
SOURCES_ALL = ["all", "top", "bottom"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full configuration sweep for Moon Survival."
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
        "--model", type=str, default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for knowledge assignment (default: 42)."
    )
    parser.add_argument(
        "--setting", type=int, default=1, choices=[1, 2, 3, 4],
        help="Experimental setting 1–4 (default: 1)."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Directory to save result files."
    )
    parser.add_argument(
        "--include-source-variants", action="store_true",
        help="Also sweep source=top and source=bottom (36 conditions instead of 12)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all combinations that would run without calling the API."
    )
    args = parser.parse_args()

    sources = SOURCES_ALL if args.include_source_variants else SOURCES_CORE

    # Build all combinations
    combos = []
    for source in sources:
        for counts in CONFIGS:
            for overlap in OVERLAPS:
                combos.append((counts, source, overlap))

    total = len(combos)

    print(f"\n{'#'*70}")
    print(f"  FULL CONFIGURATION SWEEP")
    print(f"  Total conditions : {total}")
    print(f"  Setting: {args.setting}  |  k={args.k}  |  iterations={args.iterations}  |  discussion_rounds={args.discussion_rounds}")
    print(f"  Model: {args.model}")
    print(f"{'#'*70}\n")

    if args.dry_run:
        print("  DRY RUN — conditions that would be executed:\n")
        for i, (counts, source, overlap) in enumerate(combos, 1):
            label = f"counts={counts}, source={source}, overlap={overlap}"
            print(f"  {i:>2}/{total}  {label}")
        print()
        return

    # Run each combination
    results_summary = []

    for i, (counts, source, overlap) in enumerate(combos, 1):
        label = f"counts={counts}, source={source}, overlap={overlap}"
        print(f"\n{'='*70}")
        print(f"  [{i}/{total}]  {label}")
        print(f"{'='*70}")

        try:
            log = run_experiment(
                counts=counts,
                source=source,
                overlap=overlap,
                setting=args.setting,
                k=args.k,
                num_iterations=args.iterations,
                num_discussion_rounds=args.discussion_rounds,
                model=args.model,
                knowledge_seed=args.seed,
                output_dir=args.output_dir,
                verbose=True,
            )
            results_summary.append({
                "config": label,
                "best_sad": log.get("best_sad"),
                "final_iteration_best_sad": log.get("final_iteration_best_sad"),
                "status": "OK",
            })
        except Exception as e:
            print(f"\n  ERROR for {label}: {e}")
            results_summary.append({
                "config": label,
                "best_sad": None,
                "final_iteration_best_sad": None,
                "status": f"ERROR: {e}",
            })

    # Print final comparison table
    print(f"\n\n{'#'*70}")
    print(f"  SWEEP COMPLETE — RESULTS COMPARISON")
    print(f"{'#'*70}")
    print(f"  {'Config':<55} {'Best SAD':<12} {'Final Best':<12} Status")
    print(f"  {'-'*85}")
    for row in sorted(results_summary, key=lambda r: (r["best_sad"] is None, r["best_sad"])):
        print(
            f"  {row['config']:<55} "
            f"{str(row['best_sad']):<12} "
            f"{str(row['final_iteration_best_sad']):<12} "
            f"{row['status']}"
        )

    # Save comparison CSV
    import csv
    from datetime import datetime
    csv_name = f"sweep_s{args.setting}_k{args.k}_iter{args.iterations}_disc{args.discussion_rounds}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "best_sad", "final_iteration_best_sad", "status"])
        writer.writeheader()
        writer.writerows(results_summary)

    print(f"\n  Comparison CSV saved: {csv_path}")


if __name__ == "__main__":
    main()

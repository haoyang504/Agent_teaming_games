"""
run_experiment.py
=================
Main entry point for running Moon Survival agent teaming experiments.

Usage examples
--------------
# Run a named config:
    python run_experiment.py --config high_div --setting 1 --k 3 --iterations 3

# Run with custom parameters:
    python run_experiment.py --counts 1,2,3 --source top --overlap disjoint --setting 1

# Quick single-iteration sanity check:
    python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1
"""

import argparse
import os
import sys

from experiment_runner import run_experiment

# ── Predefined configurations (3 configs × 4 overlaps = 12 core conditions) ──
CONFIGS = {
    # Config 1: High diversity [0, 2, 4]
    "high_div":          {"counts": [0, 2, 4], "source": "all", "overlap": "nested"},
    "high_div_disjoint": {"counts": [0, 2, 4], "source": "all", "overlap": "disjoint"},
    "high_div_O3":       {"counts": [0, 2, 4], "source": "all", "overlap": "O3"},
    "high_div_O4":       {"counts": [0, 2, 4], "source": "all", "overlap": "O4"},
    # Config 2: Medium diversity [1, 2, 3]
    "med_div":           {"counts": [1, 2, 3], "source": "all", "overlap": "nested"},
    "med_div_disjoint":  {"counts": [1, 2, 3], "source": "all", "overlap": "disjoint"},
    "med_div_O3":        {"counts": [1, 2, 3], "source": "all", "overlap": "O3"},
    "med_div_O4":        {"counts": [1, 2, 3], "source": "all", "overlap": "O4"},
    # Config 3: Low diversity [2, 2, 2]
    "low_div":           {"counts": [2, 2, 2], "source": "all", "overlap": "nested"},
    "low_div_disjoint":  {"counts": [2, 2, 2], "source": "all", "overlap": "disjoint"},
    "low_div_O3":        {"counts": [2, 2, 2], "source": "all", "overlap": "O3"},
    "low_div_O4":        {"counts": [2, 2, 2], "source": "all", "overlap": "O4"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Moon Survival Agent Teaming Experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Named configuration from CONFIGS (e.g. 'high_div', 'med_div_disjoint'). "
            "Use --counts for fully custom runs."
        ),
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help="Comma-separated item counts for [A,B,C], e.g. '0,2,4'.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["all", "top", "bottom"],
        help="Source pool override (default: 'all'). Values: all, top, bottom.",
    )
    parser.add_argument(
        "--overlap",
        type=str,
        default=None,
        choices=["nested", "disjoint", "O3", "O4"],
        help="Overlap pattern override (default: 'nested'). Values: nested, disjoint, O3, O4.",
    )
    parser.add_argument(
        "--setting",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Experimental setting 1–4 (default: 1). Wired up in Phase 2.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of candidate rankings per iteration (default: 3).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterative rounds (default: 3).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for knowledge assignment (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Directory to save experiment result JSON files.",
    )
    parser.add_argument(
        "--discussion-rounds",
        type=int,
        default=1,
        help=(
            "Number of discussion rounds per iteration (default: 1). "
            "Each round = all 3 agents speak once."
        ),
    )
    args = parser.parse_args()

    # ── Resolve configuration ─────────────────────────────────────────────
    if args.counts:
        # Custom counts specified
        parts = args.counts.split(",")
        if len(parts) != 3:
            print("ERROR: --counts must be exactly 3 comma-separated integers.")
            sys.exit(1)
        counts = [int(p.strip()) for p in parts]
        source = args.source or "all"
        overlap = args.overlap or "nested"
    elif args.config:
        if args.config not in CONFIGS:
            print(f"ERROR: Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        cfg = CONFIGS[args.config]
        counts = cfg["counts"]
        source = args.source or cfg["source"]
        overlap = args.overlap or cfg["overlap"]
    else:
        # Default to high_div
        cfg = CONFIGS["high_div"]
        counts = cfg["counts"]
        source = args.source or cfg["source"]
        overlap = args.overlap or cfg["overlap"]

    # ── Run ──────────────────────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  Config: counts={counts}, source={source}, overlap={overlap}, setting={args.setting}")
    print(f"{'#'*70}")
    run_experiment(
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


if __name__ == "__main__":
    main()

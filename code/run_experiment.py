"""
run_experiment.py
=================
Main entry point for running Moon Survival agent teaming experiments.

Usage examples
--------------
# Run all 4 predefined strategies (k=3, 3 iterations):
    python run_experiment.py

# Run a specific strategy:
    python run_experiment.py --strategy none+quarter+half --k 3 --iterations 3

# Quick single-iteration sanity check:
    python run_experiment.py --strategy none+quarter+half --k 1 --iterations 1
"""

import argparse
import os
import sys

from experiment_runner import run_experiment

# ── Predefined knowledge strategy sets ────────────────────────────────────────
STRATEGIES = {
    "ascending":     ["none", "quarter", "half"],    # [None, 1/4, 1/2]
    "descending":    ["half", "half", "none"],        # [1/2, 1/2, None]
    "all_half":      ["half", "half", "half"],        # [1/2, 1/2, 1/2]
    "all_quarter":   ["quarter", "quarter", "quarter"],  # [1/4, 1/4, 1/4]
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Moon Survival Agent Teaming Experiment."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=(
            "Knowledge strategy as a '+'-separated list of 3 levels "
            "(none|quarter|half|full), e.g. 'none+quarter+half'. "
            "If omitted, all 4 predefined strategies are run sequentially."
        ),
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
            "Each round = all 3 agents speak once (1->2->3)."
        ),
    )
    args = parser.parse_args()

    # ── Resolve strategies ─────────────────────────────────────────────────
    if args.strategy:
        parts = args.strategy.split("+")
        if len(parts) != 3:
            print("ERROR: --strategy must contain exactly 3 levels separated by '+'.")
            sys.exit(1)
        strategies_to_run = {"custom": parts}
    else:
        strategies_to_run = STRATEGIES

    # ── Run ──────────────────────────────────────────────────────────────────
    for name, levels in strategies_to_run.items():
        print(f"\n\n{'#'*70}")
        print(f"  Strategy: {name}  ({levels})")
        print(f"{'#'*70}")
        run_experiment(
            knowledge_strategy=levels,
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

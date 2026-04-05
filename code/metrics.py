"""
Metrics
=======
Standalone analysis module for Moon Survival experiment JSON logs.

Computes:
  - Novelty: how many item-rank pairs in each iteration's final output are new
  - Recombination: whether final rankings draw from multiple agents' proposals
  - Dominance: which agent's proposals are most represented in final output

Does not modify any existing files or require API calls.
"""

import glob
import json
import os
from typing import Any, Dict, List, Set, Tuple


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_pairs(candidate: Dict[str, int]) -> Set[Tuple[str, int]]:
    """Extract the set of (item, rank) pairs from a candidate ranking."""
    return {(item, rank) for item, rank in candidate.items()}


# ── Novelty ──────────────────────────────────────────────────────────────────

def compute_novelty(experiment_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute novelty for each iteration's final ranking(s).

    Novelty = number of item-rank pairs in this iteration's final output
    that have NEVER appeared in any prior iteration's final output.

    Returns:
        List of dicts with iteration, candidate_index, total_pairs,
        novel_pairs, and novelty_ratio.
    """
    results = []
    seen_pairs: Set[Tuple[str, int]] = set()

    for iter_log in experiment_log["iterations"]:
        iteration = iter_log["iteration"]
        final_candidates = iter_log["final_selection"]["candidates"]
        iteration_pairs: Set[Tuple[str, int]] = set()

        for ci, candidate in enumerate(final_candidates):
            pairs = extract_pairs(candidate)
            total = len(pairs)
            novel = len(pairs - seen_pairs)
            results.append({
                "iteration": iteration,
                "candidate_index": ci,
                "total_pairs": total,
                "novel_pairs": novel,
                "novelty_ratio": novel / total if total > 0 else 0.0,
            })
            iteration_pairs |= pairs

        seen_pairs |= iteration_pairs

    return results


# ── Recombination ────────────────────────────────────────────────────────────

def compute_recombination(experiment_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute recombination for each iteration's final ranking(s).

    Measures whether the final ranking draws item-rank pairs from multiple
    agents' proposals rather than from a single agent.

    Returns:
        List of dicts with iteration, candidate_index, source_agents,
        pairs_per_agent, max_agent_contribution, recombination_score,
        and is_recombination.
    """
    results = []

    for iter_log in experiment_log["iterations"]:
        iteration = iter_log["iteration"]

        # Build per-agent pair sets from proposals
        agent_pairs: Dict[int, Set[Tuple[str, int]]] = {}
        for proposal in iter_log["proposals"]:
            aid = proposal["agent_id"]
            pairs: Set[Tuple[str, int]] = set()
            for candidate in proposal["candidates"]:
                pairs |= extract_pairs(candidate)
            agent_pairs[aid] = pairs

        # Determine proposal order for attribution (lowest agent_id first)
        agent_order = sorted(agent_pairs.keys())

        final_candidates = iter_log["final_selection"]["candidates"]
        selecting_agent = iter_log["final_selection"]["agent_id"]

        for ci, candidate in enumerate(final_candidates):
            pairs = extract_pairs(candidate)
            total = len(pairs)
            attribution: Dict[int, int] = {}

            for pair in pairs:
                attributed = False
                for aid in agent_order:
                    if pair in agent_pairs[aid]:
                        attribution[aid] = attribution.get(aid, 0) + 1
                        attributed = True
                        break
                if not attributed:
                    # Attribute to selecting agent
                    attribution[selecting_agent] = attribution.get(selecting_agent, 0) + 1

            source_agents = sorted(attribution.keys())
            max_contrib = max(attribution.values()) if attribution else 0

            results.append({
                "iteration": iteration,
                "candidate_index": ci,
                "source_agents": source_agents,
                "pairs_per_agent": attribution,
                "max_agent_contribution": max_contrib,
                "recombination_score": 1 - (max_contrib / total) if total > 0 else 0.0,
                "is_recombination": len(source_agents) >= 2,
            })

    return results


# ── Dominance ────────────────────────────────────────────────────────────────

def compute_dominance(experiment_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute agent dominance for each iteration.

    Traces each item-rank pair in the final output back to the agent who
    FIRST proposed that exact pair.

    Returns:
        List of dicts with iteration, candidate_index, dominance,
        most_dominant, least_dominant, and unattributed.
    """
    results = []

    for iter_log in experiment_log["iterations"]:
        iteration = iter_log["iteration"]

        # Build pair-to-agent mapping (first proposer wins)
        pair_origin: Dict[Tuple[str, int], int] = {}
        for proposal in sorted(iter_log["proposals"], key=lambda p: p["agent_id"]):
            aid = proposal["agent_id"]
            for candidate in proposal["candidates"]:
                for pair in extract_pairs(candidate):
                    if pair not in pair_origin:
                        pair_origin[pair] = aid

        # Collect all agent IDs in the experiment
        all_agents = sorted({p["agent_id"] for p in iter_log["proposals"]})

        final_candidates = iter_log["final_selection"]["candidates"]

        for ci, candidate in enumerate(final_candidates):
            pairs = extract_pairs(candidate)
            dominance: Dict[int, int] = {aid: 0 for aid in all_agents}
            unattributed = 0

            for pair in pairs:
                if pair in pair_origin:
                    dominance[pair_origin[pair]] += 1
                else:
                    unattributed += 1

            most_dominant = max(dominance, key=lambda a: dominance[a])
            least_dominant = min(dominance, key=lambda a: dominance[a])

            results.append({
                "iteration": iteration,
                "candidate_index": ci,
                "dominance": dominance,
                "most_dominant": most_dominant,
                "least_dominant": least_dominant,
                "unattributed": unattributed,
            })

    return results


# ── Summary ──────────────────────────────────────────────────────────────────

def analyze_experiment(experiment_log: Dict[str, Any]) -> Dict[str, Any]:
    """Run all metrics on an experiment log and return a combined summary."""
    novelty = compute_novelty(experiment_log)
    recombination = compute_recombination(experiment_log)
    dominance = compute_dominance(experiment_log)

    # Aggregate summary
    mean_novelty = (
        sum(e["novelty_ratio"] for e in novelty) / len(novelty)
        if novelty else 0.0
    )
    mean_recombo = (
        sum(e["recombination_score"] for e in recombination) / len(recombination)
        if recombination else 0.0
    )

    overall_dominance: Dict[int, int] = {}
    for entry in dominance:
        for aid, count in entry["dominance"].items():
            overall_dominance[aid] = overall_dominance.get(aid, 0) + count

    most_dominant_agent = (
        max(overall_dominance, key=lambda a: overall_dominance[a])
        if overall_dominance else None
    )

    return {
        "experiment_id": experiment_log.get("experiment_id", "unknown"),
        "novelty": novelty,
        "recombination": recombination,
        "dominance": dominance,
        "summary": {
            "mean_novelty_ratio": round(mean_novelty, 4),
            "mean_recombination_score": round(mean_recombo, 4),
            "overall_dominance": overall_dominance,
            "most_dominant_agent": most_dominant_agent,
        },
    }


# ── Batch ────────────────────────────────────────────────────────────────────

def analyze_batch(results_dir: str) -> List[Dict[str, Any]]:
    """Analyze all experiment logs in a directory.

    Returns:
        List of analysis results, one per experiment.
    """
    analyses = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if path.endswith("_summary.json") or path.endswith("_analysis.json"):
            continue
        with open(path) as f:
            log = json.load(f)
        if "iterations" not in log:
            continue
        analyses.append(analyze_experiment(log))
    return analyses


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Moon Survival experiment logs.")
    parser.add_argument("log_file", help="Path to experiment JSON log file.")
    parser.add_argument("--output", help="Path to save analysis JSON (optional).")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results.")
    args = parser.parse_args()

    with open(args.log_file) as f:
        experiment_log = json.load(f)

    results = analyze_experiment(experiment_log)

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"  ANALYSIS: {results['experiment_id']}")
        print(f"{'='*60}")

        print(f"\n  --- Novelty ---")
        for entry in results["novelty"]:
            print(f"  Iter {entry['iteration']} Cand {entry['candidate_index']}: "
                  f"{entry['novel_pairs']}/{entry['total_pairs']} novel "
                  f"(ratio={entry['novelty_ratio']:.2f})")

        print(f"\n  --- Recombination ---")
        for entry in results["recombination"]:
            agents = entry["source_agents"]
            print(f"  Iter {entry['iteration']} Cand {entry['candidate_index']}: "
                  f"from {len(agents)} agents, score={entry['recombination_score']:.2f}")

        print(f"\n  --- Dominance ---")
        for entry in results["dominance"]:
            print(f"  Iter {entry['iteration']} Cand {entry['candidate_index']}: "
                  f"{entry['dominance']}, most={entry['most_dominant']}, "
                  f"least={entry['least_dominant']}, unattributed={entry['unattributed']}")

        print(f"\n  --- Summary ---")
        s = results["summary"]
        print(f"  Mean novelty ratio: {s['mean_novelty_ratio']:.2f}")
        print(f"  Mean recombination score: {s['mean_recombination_score']:.2f}")
        print(f"  Overall dominance: {s['overall_dominance']}")
        print(f"  Most dominant agent: {s['most_dominant_agent']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Analysis saved: {args.output}")

"""
Experiment Runner
=================
Orchestrates the iterative multi-agent Moon Survival experiment.

One iteration:
  1. All agents propose k candidates (given prior results).
  2. Agents conduct a round-robin discussion.
  3. The last agent selects k final candidates.
  4. All final candidates are evaluated (SAD score).

Results are accumulated across iterations and saved to JSON.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from moon_survival_env import (
    evaluate_ranking,
    format_results_summary,
    ranking_dict_to_list,
)
from knowledge_manager import (
    generate_knowledge_assignment,
    knowledge_level_label,
    format_knowledge_counts,
)
from agent import MoonSurvivalAgent
import config


# ── Iteration ─────────────────────────────────────────────────────────────────

def run_iteration(
    agents: List[MoonSurvivalAgent],
    k: int,
    iteration: int,
    previous_candidates: Optional[List[Dict[str, int]]],
    previous_scores: Optional[List[int]],
    num_discussion_rounds: int = 1,
    verbose: bool = True,
) -> Tuple[List[Dict[str, int]], List[int], Dict[str, Any]]:
    """Run a single iteration of the agent teaming experiment.

    Args:
        agents:               List of MoonSurvivalAgent instances (order = round-robin order).
        k:                    Number of candidates to propose / select.
        iteration:            Current iteration number (1-indexed).
        previous_candidates:  Candidates from the prior iteration (None = first iter).
        previous_scores:      SAD scores corresponding to previous_candidates.
        verbose:              If True, print progress to stdout.

    Returns:
        Tuple of:
          - final_candidates:  List of k {item_name: rank} dicts selected by last agent.
          - final_scores:      Corresponding SAD scores.
          - log:               Dict capturing all prompts, responses, and scores.
    """
    log: Dict[str, Any] = {
        "iteration": iteration,
        "proposals": [],
        "discussion": [],
        "final_selection": None,
        "final_scores": [],
    }

    # ── Format previous results for prompt ──────────────────────────────────
    prev_summary: Optional[str] = None
    if previous_candidates and previous_scores:
        prev_summary = format_results_summary(previous_candidates, previous_scores)

    # ── Phase 1 : Proposals ──────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}  —  Phase 1: Proposals")
        print(f"{'='*60}")

    all_proposals: Dict[int, List[Dict[str, int]]] = {}
    discussion_history: List[Dict[str, str]] = []

    for agent in agents:
        if verbose:
            print(f"\n  Agent {agent.agent_id} proposing {k} candidate(s)...")
        candidates, raw = agent.propose_candidates(k, prev_summary)
        all_proposals[agent.agent_id] = candidates

        proposal_entry = {
            "agent_id": agent.agent_id,
            "raw_response": raw,
            "candidates": [dict(c) for c in candidates],
        }
        log["proposals"].append(proposal_entry)

        # Append to discussion history for subsequent phases
        discussion_history.append({
            "role": "user",
            "content": (
                f"=== Agent {agent.agent_id}'s Proposals (Iteration {iteration}) ===\n"
                f"{raw}"
            ),
        })

    # ── Phase 2 : Round-Robin Discussion ────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}  —  Phase 2: Round-Robin Discussion")
        print(f"  ({num_discussion_rounds} discussion round(s), {len(agents)} agents each)")
        print(f"{'='*60}")

    for disc_round in range(1, num_discussion_rounds + 1):
        if verbose and num_discussion_rounds > 1:
            print(f"\n  -- Discussion Round {disc_round}/{num_discussion_rounds} --")
        for agent in agents:
            if verbose:
                print(f"\n  Agent {agent.agent_id} discussing (round {disc_round})...")
            response = agent.discuss(discussion_history, iteration)
            if verbose:
                print(f"    → {response[:200]}{'...' if len(response)>200 else ''}")

            discussion_entry = {
                "agent_id": agent.agent_id,
                "discussion_round": disc_round,
                "response": response,
            }
            log["discussion"].append(discussion_entry)

            discussion_history.append({
                "role": "assistant",
                "content": (
                    f"=== Agent {agent.agent_id} Discussion Round {disc_round} ===\n{response}"
                ),
            })

    # ── Phase 3 : Final Selection (last agent) ──────────────────────────────
    last_agent = agents[-1]
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}  —  Phase 3: Final Selection (Agent {last_agent.agent_id})")
        print(f"{'='*60}")

    final_candidates, raw_selection = last_agent.select_final_candidates(
        k, discussion_history, iteration
    )

    # ── Phase 4 : Evaluation ────────────────────────────────────────────────
    final_scores: List[int] = []
    for i, cand in enumerate(final_candidates):
        ranks = ranking_dict_to_list(cand)
        score = evaluate_ranking(ranks)
        final_scores.append(score)
        if verbose:
            print(f"  Final Candidate {i+1}: SAD = {score}")

    log["final_selection"] = {
        "agent_id": last_agent.agent_id,
        "raw_response": raw_selection,
        "candidates": [dict(c) for c in final_candidates],
    }
    log["final_scores"] = final_scores

    return final_candidates, final_scores, log


# ── Full Experiment ───────────────────────────────────────────────────────────

def run_experiment(
    counts: List[int],
    source: str = "all",
    overlap: str = "nested",
    setting: int = 1,
    k: int = 3,
    num_iterations: int = 3,
    num_discussion_rounds: int = 1,
    model: str = "gpt-4o-mini",
    knowledge_seed: int = 42,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full multi-iteration Moon Survival agent teaming experiment.

    Args:
        counts:             List of 3 integers — items each agent knows [A, B, C].
        source:             Source pool: "all", "top", or "bottom".
        overlap:            Overlap pattern: "nested", "disjoint", "O3", "O4".
        setting:            Experimental setting 1–4 (passed through, wired in Phase 2).
        k:                  Number of candidates per iteration.
        num_iterations:     Number of iterations to run.
        model:              OpenAI model identifier.
        knowledge_seed:     Seed for reproducible knowledge assignment.
        output_dir:         Directory to save JSON result logs.
        verbose:            Print progress if True.

    Returns:
        Aggregated experiment log dict.
    """
    assert len(counts) == 3, "Exactly 3 agents are supported."

    # ── Setup ────────────────────────────────────────────────────────────────
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    knowledge_assignments = generate_knowledge_assignment(
        counts, source=source, overlap=overlap, seed=knowledge_seed
    )

    agents = [
        MoonSurvivalAgent(
            agent_id=i + 1,
            knowledge=knowledge_assignments[i],
            client=client,
            model=model,
            num_agents=3,
        )
        for i in range(3)
    ]

    counts_label = "-".join(str(c) for c in counts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = (
        f"c{counts_label}__{source}__{overlap}__s{setting}"
        f"__k{k}__iter{num_iterations}__disc{num_discussion_rounds}__{timestamp}"
    )

    # ── Knowledge dump ───────────────────────────────────────────────────────
    agent_labels = ["A", "B", "C"]
    agent_knowledge_log = []
    for i, (count, knowledge) in enumerate(zip(counts, knowledge_assignments)):
        agent_entry = {
            "agent_id": i + 1,
            "agent_label": agent_labels[i],
            "count": count,
            "label": knowledge_level_label(count),
            "items": [
                {"item": item, "expert_rank": rank, "explanation": expl}
                for item, rank, expl in knowledge
            ],
        }
        agent_knowledge_log.append(agent_entry)

    if verbose:
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT: {experiment_id}")
        print(f"  Knowledge: {format_knowledge_counts(counts)}")
        print(f"  Source: {source}, Overlap: {overlap}, Setting: {setting}")
        print(f"  k={k}, iterations={num_iterations}, model={model}")
        print(f"{'#'*60}")
        print(f"\n  === KNOWLEDGE ASSIGNMENTS ===")
        for entry in agent_knowledge_log:
            print(f"\n  Agent {entry['agent_label']} [{entry['label']}]:")
            if entry["items"]:
                for kv in entry["items"]:
                    print(f"    #{kv['expert_rank']:2d}  {kv['item']}")
            else:
                print(f"    (no specialised knowledge)")
        print(f"\n{'#'*60}")

    experiment_log: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "counts": counts,
        "source": source,
        "overlap": overlap,
        "setting": setting,
        "k": k,
        "num_iterations": num_iterations,
        "num_discussion_rounds": num_discussion_rounds,
        "model": model,
        "knowledge_seed": knowledge_seed,
        "agent_knowledge": agent_knowledge_log,
        "iterations": [],
    }

    # ── Iterative loop ───────────────────────────────────────────────────────
    prev_candidates: Optional[List[Dict[str, int]]] = None
    prev_scores: Optional[List[int]] = None

    for it in range(1, num_iterations + 1):
        final_candidates, final_scores, iter_log = run_iteration(
            agents=agents,
            k=k,
            iteration=it,
            previous_candidates=prev_candidates,
            previous_scores=prev_scores,
            num_discussion_rounds=num_discussion_rounds,
            verbose=verbose,
        )
        experiment_log["iterations"].append(iter_log)

        prev_candidates = final_candidates
        prev_scores = final_scores

    # ── Build iteration summary ──────────────────────────────────────────────
    iteration_summary = []
    for iter_log in experiment_log["iterations"]:
        scores = iter_log["final_scores"]
        iteration_summary.append({
            "iteration": iter_log["iteration"],
            "scores": scores,
            "best_sad": min(scores),
            "worst_sad": max(scores),
            "mean_sad": round(sum(scores) / len(scores), 2),
        })

    all_scores = [s for row in iteration_summary for s in row["scores"]]
    experiment_log["iteration_summary"] = iteration_summary
    experiment_log["best_sad"] = min(all_scores) if all_scores else None
    experiment_log["final_iteration_best_sad"] = min(prev_scores) if prev_scores else None

    if verbose:
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'#'*60}")
        print(f"\n  {'Iter':<6} {'Best SAD':<10} {'Mean SAD':<10} {'All Scores'}")
        print(f"  {'-'*50}")
        for row in iteration_summary:
            scores_str = str(row["scores"])
            print(f"  {row['iteration']:<6} {row['best_sad']:<10} {row['mean_sad']:<10} {scores_str}")
        print(f"\n  Best SAD across all iterations : {experiment_log['best_sad']}")
        print(f"  Best SAD in final iteration    : {experiment_log['final_iteration_best_sad']}")
        print(f"{'#'*60}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{experiment_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2)

    # ── Save compact summary CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{experiment_id}_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("iteration,best_sad,mean_sad,worst_sad,all_scores\n")
        for row in iteration_summary:
            f.write(
                f"{row['iteration']},{row['best_sad']},{row['mean_sad']},"
                f"{row['worst_sad']},\"{row['scores']}\"\n"
            )

    if verbose:
        print(f"\n  Full log : {out_path}")
        print(f"  Summary  : {csv_path}")

    return experiment_log


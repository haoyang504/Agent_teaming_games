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
import random
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
    generate_mixed_knowledge_assignment,
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
    discussion_seed: Optional[int] = None,
    discussion_order: str = "random",
    results_context: str = "the previous iteration",
    verbose: bool = True,
) -> Tuple[List[Dict[str, int]], List[int], Dict[str, Any]]:
    """Run a single iteration of the agent teaming experiment.

    Args:
        agents:               List of MoonSurvivalAgent instances.
        k:                    Number of candidates to propose / select.
        iteration:            Current iteration number (1-indexed).
        previous_candidates:  Candidates from the prior iteration (None = first iter).
        previous_scores:      SAD scores corresponding to previous_candidates.
        discussion_seed:      Seed for reproducible discussion order shuffling.
        discussion_order:     "random", "ABC" (1→2→3), or "CBA" (3→2→1).
        results_context:      Human-readable label for what the feedback covers.
        verbose:              If True, print progress to stdout.

    Returns:
        Tuple of:
          - final_candidates:  List of k {item_name: rank} dicts selected by last agent.
          - final_scores:      Corresponding SAD scores.
          - log:               Dict capturing all prompts, responses, and scores.
    """
    disc_rng = random.Random(discussion_seed) if discussion_seed is not None else random.Random()

    log: Dict[str, Any] = {
        "iteration": iteration,
        "feedback_provided": previous_candidates is not None,
        "proposals": [],
        "discussion": [],
        "discussion_orders": {},
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
        candidates, raw = agent.propose_candidates(k, prev_summary, results_context=results_context)
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
        # Determine speaking order for this round
        if discussion_order == "ABC":
            round_order = sorted(agents, key=lambda a: a.agent_id)
        elif discussion_order == "CBA":
            round_order = sorted(agents, key=lambda a: a.agent_id, reverse=True)
        else:
            round_order = list(agents)
            disc_rng.shuffle(round_order)
        log["discussion_orders"][disc_round] = [a.agent_id for a in round_order]

        if verbose:
            order_str = " → ".join(str(a.agent_id) for a in round_order)
            print(f"\n  -- Discussion Round {disc_round}/{num_discussion_rounds} (order: {order_str}) --")

        for agent in round_order:
            if verbose:
                print(f"\n  Agent {agent.agent_id} discussing (round {disc_round})...")
            response = agent.discuss(discussion_history, iteration)
            if verbose:
                print(f"    → {response[:200]}{'...' if len(response)>200 else ''}")

            discussion_entry = {
                "agent_id": agent.agent_id,
                "discussion_round": disc_round,
                "speaking_order": [a.agent_id for a in round_order],
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
    feedback_mode: str = "F1",
    incorrect_pattern: Optional[str] = None,
    incorrect_warning: bool = False,
    incorrect_seed: int = 99,
    k: int = 3,
    num_iterations: int = 3,
    num_discussion_rounds: int = 1,
    discussion_order: str = "random",
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
        setting:            Experimental setting 1–4.
        feedback_mode:      Feedback mode: "F1"–"F5".
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

    incorrect_assignments = None
    if incorrect_pattern:
        correct_assignments, incorrect_assignments = generate_mixed_knowledge_assignment(
            counts=counts,
            incorrect_pattern=incorrect_pattern,
            seed=knowledge_seed,
            incorrect_seed=incorrect_seed,
        )
        knowledge_assignments = [
            correct + incorrect
            for correct, incorrect in zip(correct_assignments, incorrect_assignments)
        ]
    else:
        knowledge_assignments = generate_knowledge_assignment(
            counts, source=source, overlap=overlap, seed=knowledge_seed
        )

    # ── Compute setting-dependent info ───────────────────────────────────
    team_knowledge_info = None
    leader_id = None

    if setting in (2, 4):
        team_knowledge_info = {
            "A": counts[0],
            "B": counts[1],
            "C": counts[2],
        }

    if setting in (3, 4):
        leader_id = 3

    agents = [
        MoonSurvivalAgent(
            agent_id=i + 1,
            knowledge=knowledge_assignments[i],
            client=client,
            model=model,
            num_agents=3,
            team_knowledge_info=team_knowledge_info,
            leader_id=leader_id,
            knowledge_warning=incorrect_warning,
        )
        for i in range(3)
    ]

    counts_label = "-".join(str(c) for c in counts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = (
        f"c{counts_label}__{source}__{overlap}__s{setting}__fb{feedback_mode}"
        f"__do{discussion_order}__k{k}__iter{num_iterations}__disc{num_discussion_rounds}__{timestamp}"
    )
    if incorrect_pattern:
        experiment_id += f"__inc{incorrect_pattern}"
        if incorrect_warning:
            experiment_id += "_warn"

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

    if incorrect_assignments:
        for i, incorrect in enumerate(incorrect_assignments):
            agent_knowledge_log[i]["incorrect_items"] = [
                {"item": item, "incorrect_rank": rank, "explanation": expl}
                for item, rank, expl in incorrect
            ]

    if verbose:
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT: {experiment_id}")
        print(f"  Knowledge: {format_knowledge_counts(counts)}")
        print(f"  Source: {source}, Overlap: {overlap}, Setting: {setting}")
        print(f"  Feedback mode: {feedback_mode}")
        if incorrect_pattern:
            print(f"  Incorrect pattern: {incorrect_pattern}, Warning: {incorrect_warning}")
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
        if setting >= 2:
            print(f"\n  Team info provided: {team_knowledge_info is not None}")
            if leader_id:
                print(f"  Leader designated: Agent C (id={leader_id})")
            else:
                print(f"  No leader designated")
        print(f"\n{'#'*60}")

    experiment_log: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "counts": counts,
        "source": source,
        "overlap": overlap,
        "setting": setting,
        "feedback_mode": feedback_mode,
        "incorrect_pattern": incorrect_pattern,
        "incorrect_warning": incorrect_warning,
        "incorrect_seed": incorrect_seed,
        "team_knowledge_info": team_knowledge_info,
        "leader_id": leader_id,
        "k": k,
        "num_iterations": num_iterations,
        "num_discussion_rounds": num_discussion_rounds,
        "discussion_order": discussion_order,
        "model": model,
        "knowledge_seed": knowledge_seed,
        "history_scope": "last_iteration_only",
        "agent_knowledge": agent_knowledge_log,
        "iterations": [],
    }

    # ── Iterative loop ───────────────────────────────────────────────────────
    prev_candidates: Optional[List[Dict[str, int]]] = None
    prev_scores: Optional[List[int]] = None

    # For F4/F5: accumulate all candidates and scores across iterations
    all_candidates_history: List[Dict[str, int]] = []
    all_scores_history: List[int] = []

    # Determine feedback frequency for F1-F3
    feedback_frequency = {"F1": 1, "F2": 2, "F3": 4}.get(feedback_mode, 1)

    # Determine results context label
    if feedback_mode in ("F1", "F2", "F3"):
        results_context = "the previous iteration"
    elif feedback_mode == "F4":
        results_context = "all prior iterations"
    elif feedback_mode == "F5":
        results_context = "all prior iterations (showing top 8 candidates only)"
    else:
        results_context = "the previous iteration"

    for it in range(1, num_iterations + 1):
        # ── Determine what feedback to pass this iteration ──
        if feedback_mode in ("F1", "F2", "F3"):
            if it == 1:
                iter_candidates = None
                iter_scores = None
            elif it % feedback_frequency == 0:
                iter_candidates = prev_candidates
                iter_scores = prev_scores
            else:
                iter_candidates = None
                iter_scores = None

        elif feedback_mode == "F4":
            if all_candidates_history:
                iter_candidates = list(all_candidates_history)
                iter_scores = list(all_scores_history)
            else:
                iter_candidates = None
                iter_scores = None

        elif feedback_mode == "F5":
            if all_candidates_history:
                paired = list(zip(all_scores_history, all_candidates_history))
                paired.sort(key=lambda x: x[0])
                top_8 = paired[:8]
                iter_scores = [s for s, _ in top_8]
                iter_candidates = [c for _, c in top_8]
            else:
                iter_candidates = None
                iter_scores = None

        else:
            iter_candidates = prev_candidates
            iter_scores = prev_scores

        final_candidates, final_scores, iter_log = run_iteration(
            agents=agents,
            k=k,
            iteration=it,
            previous_candidates=iter_candidates,
            previous_scores=iter_scores,
            num_discussion_rounds=num_discussion_rounds,
            discussion_seed=knowledge_seed + it,
            discussion_order=discussion_order,
            results_context=results_context,
            verbose=verbose,
        )
        experiment_log["iterations"].append(iter_log)

        # Update tracking
        prev_candidates = final_candidates
        prev_scores = final_scores

        # Accumulate for F4/F5
        all_candidates_history.extend(final_candidates)
        all_scores_history.extend(final_scores)

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


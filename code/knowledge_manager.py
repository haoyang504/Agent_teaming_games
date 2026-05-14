"""
Knowledge Manager
=================
Handles asymmetric distribution of expert knowledge to each agent.

Two-dimensional knowledge assignment:
  - counts:  how many items each agent knows (2 configs after Phase 8)
  - overlap: how knowledge sets relate across agents (nested/disjoint/O3/O4/O5)

Source pool variants (top/bottom) were removed in Phase 8; the source is
always the full 15-item ranking.

Agent C (index 2) is always sampled first, then B, then A.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

from moon_survival_env import ITEMS, GROUND_TRUTH_RANKS, EXPLANATIONS

INCORRECT_RANKING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "incorrect_ranking.json"
)


# ── Public API ───────────────────────────────────────────────────────────────

def generate_knowledge_assignment(
    counts: List[int],
    overlap: str = "nested",
    seed: int = 42,
) -> List[List[Tuple[str, int, str]]]:
    """Assign knowledge items to each agent based on counts and overlap pattern.

    Agent C (index 2) is sampled first, then B (index 1), then A (index 0).

    Args:
        counts:  List of 3 integers — items each agent knows [A, B, C].
        overlap: One of:
                   - "nested"   — A ⊆ B ⊆ C
                   - "disjoint" — A, B, C pairwise disjoint
                   - "O3"       — B ⊆ C; A disjoint from both B and C
                   - "O4"       — A ⊆ C; B disjoint from both A and C
                   - "O5"       — A = B (smaller is subset of larger when counts
                                  differ); C disjoint from both
        seed:    Random seed for reproducibility.

    Returns:
        List of 3 lists of (item_name, ground_truth_rank, explanation) tuples.

    Raises:
        ValueError: if the configuration is impossible.
    """
    if len(counts) != 3:
        raise ValueError(f"counts must have exactly 3 elements, got {len(counts)}")

    rng = random.Random(seed)

    pool_list = list(range(len(ITEMS)))

    count_a, count_b, count_c = counts

    # Validate total doesn't exceed pool for disjoint
    if overlap == "disjoint":
        total_needed = count_a + count_b + count_c
        if total_needed > len(pool_list):
            raise ValueError(
                f"Disjoint overlap requires {total_needed} items but the pool "
                f"only has {len(pool_list)} items."
            )

    if overlap == "nested":
        # A ⊆ B ⊆ C
        items_c = _sample(rng, pool_list, count_c)
        items_b = _sample(rng, items_c, count_b)
        items_a = _sample(rng, items_b, count_a)

    elif overlap == "disjoint":
        # No overlap between any pair
        remaining = list(pool_list)
        rng.shuffle(remaining)
        items_c = remaining[:count_c]
        remaining = remaining[count_c:]
        items_b = remaining[:count_b]
        remaining = remaining[count_b:]
        items_a = remaining[:count_a]

    elif overlap == "O3":
        # B ⊆ C; A disjoint from both B and C
        items_c = _sample(rng, pool_list, count_c)
        if count_b > count_c:
            raise ValueError(
                f"O3 overlap: B (count={count_b}) cannot be a subset of C (count={count_c})."
            )
        items_b = _sample(rng, items_c, count_b)
        c_set = set(items_c)
        non_c = [i for i in pool_list if i not in c_set]
        if count_a > len(non_c):
            raise ValueError(
                f"O3 overlap: need {count_a} items for Agent A disjoint from C, "
                f"but only {len(non_c)} available."
            )
        items_a = _sample(rng, non_c, count_a)

    elif overlap == "O4":
        # A ⊆ C; B disjoint from both A and C
        items_c = _sample(rng, pool_list, count_c)
        if count_a > count_c:
            raise ValueError(
                f"O4 overlap: A (count={count_a}) cannot be a subset of C (count={count_c})."
            )
        items_a = _sample(rng, items_c, count_a)
        c_set = set(items_c)
        non_c = [i for i in pool_list if i not in c_set]
        if count_b > len(non_c):
            raise ValueError(
                f"O4 overlap: need {count_b} items for Agent B disjoint from C, "
                f"but only {len(non_c)} available."
            )
        items_b = _sample(rng, non_c, count_b)

    elif overlap == "O5":
        # A = B (smaller is subset of larger when counts differ); C disjoint
        items_c = _sample(rng, pool_list, count_c)
        c_set = set(items_c)
        non_c = [i for i in pool_list if i not in c_set]
        larger_count = max(count_a, count_b)
        if larger_count > len(non_c):
            raise ValueError(
                f"O5 overlap: need {larger_count} items disjoint from C, "
                f"but only {len(non_c)} available."
            )
        larger_set = _sample(rng, non_c, larger_count)
        if count_a >= count_b:
            items_a = larger_set
            items_b = _sample(rng, items_a, count_b)
        else:
            items_b = larger_set
            items_a = _sample(rng, items_b, count_a)

    else:
        raise ValueError(
            f"Unknown overlap '{overlap}'. Valid: nested, disjoint, O3, O4, O5"
        )

    # Build assignments in [A, B, C] order
    assignments = []
    for agent_items in [items_a, items_b, items_c]:
        knowledge = [
            (ITEMS[i], GROUND_TRUTH_RANKS[i], EXPLANATIONS[i])
            for i in agent_items
        ]
        assignments.append(knowledge)

    return assignments


def format_knowledge_for_prompt(
    knowledge: List[Tuple[str, int, str]],
) -> str:
    """Convert an agent's knowledge list into a prompt-friendly string block."""
    if not knowledge:
        return (
            "You have no specialised prior knowledge about these items — "
            "rely on general reasoning and the discussion with your teammates."
        )

    lines = [
        "You have specialised knowledge (from NASA training) about the "
        "following items. Use this to inform your rankings and arguments:\n"
    ]
    for item, rank, explanation in knowledge:
        lines.append(
            f"  • {item} [Expert rank: #{rank}]\n"
            f"    {explanation}"
        )
    return "\n".join(lines)


def knowledge_level_label(count: int) -> str:
    """Return a human-readable label for a knowledge count."""
    return f"{count}/15 items"


def format_knowledge_counts(counts: List[int]) -> str:
    """Return a human-readable summary of agent knowledge counts."""
    labels = ["Agent A", "Agent B", "Agent C"]
    return ", ".join(f"{label}: {count}" for label, count in zip(labels, counts))


def generate_incorrect_ranking(seed: int = 99) -> List[Tuple[str, int, str]]:
    """Return the fixed incorrect ranking (item, rank, explanation) tuples.

    Loads from `incorrect_ranking.json` when present so the same ranking is
    reused across runs (Phase 8 requirement — ES wants one canonical incorrect
    ranking, not a fresh derangement per run). If the file is missing, falls
    back to derangement-based generation seeded by `seed`.

    Args:
        seed: Only used as the fallback derangement seed when the JSON file
              is absent. The persisted ranking is independent of this argument.
    """
    if os.path.exists(INCORRECT_RANKING_FILE):
        with open(INCORRECT_RANKING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        explanation = data["_meta"]["explanation"]
        return [
            (entry["item"], entry["rank"], explanation)
            for entry in data["ranking"]
        ]

    # Fallback: derive from seed (used the first time, or if the JSON is gone).
    rng = random.Random(seed)
    n = len(ITEMS)
    incorrect_ranks = _derangement(rng, n)
    explanation = (
        "Based on preliminary analysis, this item has been assessed at this priority level."
    )
    return [
        (ITEMS[i], incorrect_ranks[i], explanation)
        for i in range(n)
    ]


def generate_mixed_knowledge_assignment(
    counts: List[int],
    incorrect_pattern: str,
    seed: int = 42,
    incorrect_seed: int = 99,
) -> Tuple[List[List[Tuple[str, int, str]]], List[List[Tuple[str, int, str]]]]:
    """Generate per-agent knowledge with the requested incorrect-knowledge pattern.

    Phase 8 semantics: incorrect items REPLACE correct items at the same
    per-agent count (the agent's total knowledge count is unchanged).

    Pattern semantics:
      - I1: Agent A's items are all incorrect; B, C keep all correct.
      - I2: Agent B's items are all incorrect; A, C keep all correct.
      - I3: Agent C's items are all incorrect; A, B keep all correct.
      - I4: Each agent has half-correct + half-incorrect. For even counts the
            split is exact (count // 2 each). For odd counts the incorrect
            half is the ceiling (count - count//2). Agents with count=0 get 0.

    Each agent's incorrect items are sampled from the fixed incorrect ranking
    with the constraint that no incorrect (item, rank) pair clashes with any
    correct (item, rank) tuple already assigned to the *same agent*.

    Fixed correct-knowledge overlap is "disjoint" (so no clash across agents).

    Args:
        counts:            [A, B, C] item counts.
        incorrect_pattern: "I1", "I2", "I3", or "I4".
        seed:              Seed for correct knowledge assignment.
        incorrect_seed:    Seed for the random choice of which slots get
                           replaced when fewer than all are.

    Returns:
        Tuple of (correct_assignments, incorrect_assignments) where each
        agent's two lists partition into the merged knowledge the runner gives
        them. Total per-agent count = counts[i] (replacement, not append).
    """
    if incorrect_pattern not in ("I1", "I2", "I3", "I4"):
        raise ValueError(
            f"Unknown incorrect_pattern '{incorrect_pattern}'. Valid: I1, I2, I3, I4"
        )

    correct_assignments = generate_knowledge_assignment(
        counts, overlap="disjoint", seed=seed
    )
    incorrect_ranking = generate_incorrect_ranking(seed=incorrect_seed)

    # Per-pattern: how many of each agent's slots are incorrect.
    if incorrect_pattern == "I1":
        incorrect_counts = [counts[0], 0, 0]
    elif incorrect_pattern == "I2":
        incorrect_counts = [0, counts[1], 0]
    elif incorrect_pattern == "I3":
        incorrect_counts = [0, 0, counts[2]]
    else:  # I4 — half each
        incorrect_counts = [c - (c // 2) for c in counts]  # ceil half

    rng = random.Random(seed + incorrect_seed)
    final_correct: List[List[Tuple[str, int, str]]] = []
    final_incorrect: List[List[Tuple[str, int, str]]] = []
    for i in range(3):
        agent_correct = list(correct_assignments[i])
        n_incorrect = incorrect_counts[i]
        if n_incorrect == 0:
            final_correct.append(agent_correct)
            final_incorrect.append([])
            continue

        # Drop n_incorrect random slots from the correct assignment.
        rng.shuffle(agent_correct)
        kept_correct = agent_correct[n_incorrect:]
        # Sample replacements that don't clash on item or rank with kept_correct.
        agent_incorrect = _sample_incorrect_for_agent(
            rng, incorrect_ranking, kept_correct, n_incorrect
        )
        final_correct.append(kept_correct)
        final_incorrect.append(agent_incorrect)

    return final_correct, final_incorrect


# ── Internal helpers ─────────────────────────────────────────────────────────

def _derangement(rng: random.Random, n: int) -> List[int]:
    """Generate a random derangement of [1..n] (no fixed points)."""
    while True:
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        if all(perm[i] != i + 1 for i in range(n)):
            return perm


def _sample_incorrect_for_agent(
    rng: random.Random,
    incorrect_ranking: List[Tuple[str, int, str]],
    correct_knowledge: List[Tuple[str, int, str]],
    count: int,
) -> List[Tuple[str, int, str]]:
    """Sample incorrect items for one agent, avoiding overlap with their correct knowledge."""
    if count == 0:
        return []

    correct_items = {item for item, _, _ in correct_knowledge}
    correct_ranks = {rank for _, rank, _ in correct_knowledge}

    eligible = [
        (item, rank, expl) for item, rank, expl in incorrect_ranking
        if item not in correct_items and rank not in correct_ranks
    ]

    if len(eligible) < count:
        raise ValueError(
            f"Cannot find {count} non-overlapping incorrect items. "
            f"Only {len(eligible)} eligible after filtering."
        )

    return rng.sample(eligible, count)

def _sample(rng: random.Random, population: List[int], k: int) -> List[int]:
    """Sample k items from population without replacement. Returns [] if k == 0."""
    if k == 0:
        return []
    if k > len(population):
        raise ValueError(
            f"Cannot sample {k} items from a pool of {len(population)}"
        )
    return rng.sample(population, k)



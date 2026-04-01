"""
Knowledge Manager
=================
Handles asymmetric distribution of expert knowledge to each agent.

Three-dimensional knowledge assignment:
  - counts:  how many items each agent knows (3 configs)
  - source:  which portion of the ranking pool to draw from (all/top/bottom)
  - overlap: how knowledge sets relate across agents (nested/disjoint/O3/O4)

Agent C (index 2) is always sampled first, then B, then A.
"""

import random
from typing import Dict, List, Optional, Tuple

from moon_survival_env import ITEMS, GROUND_TRUTH_RANKS, EXPLANATIONS


# ── Public API ───────────────────────────────────────────────────────────────

def generate_knowledge_assignment(
    counts: List[int],
    source: str = "all",
    overlap: str = "nested",
    seed: int = 42,
) -> List[List[Tuple[str, int, str]]]:
    """Assign knowledge items to each agent based on counts, source pool, and overlap pattern.

    Agent C (index 2) is sampled first, then B (index 1), then A (index 0).

    Args:
        counts:  List of 3 integers — items each agent knows [A, B, C].
        source:  "all" (ranks 1–15), "top" (ranks 1–8), "bottom" (ranks 8–15).
        overlap: "nested" (A⊆B⊆C), "disjoint" (no overlap), "O3", or "O4".
        seed:    Random seed for reproducibility.

    Returns:
        List of 3 lists of (item_name, ground_truth_rank, explanation) tuples.

    Raises:
        ValueError: if the configuration is impossible.
    """
    if len(counts) != 3:
        raise ValueError(f"counts must have exactly 3 elements, got {len(counts)}")

    rng = random.Random(seed)

    # Build source pool — list of indices into ITEMS
    pool_indices = _get_source_pool(source)
    pool_set = set(pool_indices)

    count_a, count_b, count_c = counts

    # Validate total doesn't exceed pool for disjoint
    if overlap == "disjoint":
        total_needed = count_a + count_b + count_c
        if total_needed > len(pool_indices):
            raise ValueError(
                f"Disjoint overlap requires {total_needed} items but source "
                f"pool '{source}' only has {len(pool_indices)} items."
            )

    # Sample C first, then B, then A
    pool_list = list(pool_indices)

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
        # B overlaps C (guaranteed); A disjoint from C
        items_c = _sample(rng, pool_list, count_c)
        c_set = set(items_c)
        non_c = [i for i in pool_list if i not in c_set]
        if count_a > len(non_c):
            raise ValueError(
                f"O3 overlap: need {count_a} items for Agent A disjoint from C, "
                f"but only {len(non_c)} available."
            )
        items_a = _sample(rng, non_c, count_a)
        # B: guarantee at least 1 item from C, rest from full pool
        items_b = _sample_with_guaranteed_overlap(rng, pool_list, items_c, count_b)

    elif overlap == "O4":
        # B disjoint from C; A overlaps C (guaranteed)
        items_c = _sample(rng, pool_list, count_c)
        c_set = set(items_c)
        non_c = [i for i in pool_list if i not in c_set]
        if count_b > len(non_c):
            raise ValueError(
                f"O4 overlap: need {count_b} items for Agent B disjoint from C, "
                f"but only {len(non_c)} available."
            )
        items_b = _sample(rng, non_c, count_b)
        # A: guarantee at least 1 item from C, rest from full pool
        items_a = _sample_with_guaranteed_overlap(rng, pool_list, items_c, count_a)

    else:
        raise ValueError(
            f"Unknown overlap '{overlap}'. Valid: nested, disjoint, O3, O4"
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


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_source_pool(source: str) -> List[int]:
    """Return list of item indices for the given source pool."""
    if source == "all":
        return list(range(len(ITEMS)))
    elif source == "top":
        return [i for i in range(len(ITEMS)) if GROUND_TRUTH_RANKS[i] <= 8]
    elif source == "bottom":
        return [i for i in range(len(ITEMS)) if GROUND_TRUTH_RANKS[i] >= 8]
    else:
        raise ValueError(f"Unknown source '{source}'. Valid: all, top, bottom")


def _sample(rng: random.Random, population: List[int], k: int) -> List[int]:
    """Sample k items from population without replacement. Returns [] if k == 0."""
    if k == 0:
        return []
    if k > len(population):
        raise ValueError(
            f"Cannot sample {k} items from a pool of {len(population)}"
        )
    return rng.sample(population, k)


def _sample_with_guaranteed_overlap(
    rng: random.Random,
    pool: List[int],
    overlap_source: List[int],
    k: int,
) -> List[int]:
    """Sample k items from pool, guaranteeing at least 1 is from overlap_source.

    If k == 0, returns []. If overlap_source is empty or k can't include an
    overlap item, falls back to plain sampling.
    """
    if k == 0:
        return []
    if not overlap_source:
        return _sample(rng, pool, k)

    # Pick 1 guaranteed overlap item
    shared = rng.sample(overlap_source, 1)
    if k == 1:
        return shared

    # Fill remaining from full pool (excluding the guaranteed item)
    remaining_pool = [i for i in pool if i != shared[0]]
    rest = _sample(rng, remaining_pool, k - 1)
    return shared + rest

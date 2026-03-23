"""
Knowledge Manager
=================
Handles asymmetric distribution of expert knowledge to each agent.

Knowledge levels
----------------
  "full"   → all 15 items
  "half"   → 8 items (items 0-7 of a fixed shuffle)
  "quarter" → 4 items (items 0-3 of the same fixed shuffle)
  "none"   → no expert knowledge

All agents that share the same level see the SAME set of items (fixed by seed).
"""

import random
from typing import Dict, List, Optional, Tuple

from moon_survival_env import ITEMS, GROUND_TRUTH_RANKS, EXPLANATIONS

# ── Knowledge level definitions ──────────────────────────────────────────────

LEVEL_COUNTS: Dict[str, Optional[int]] = {
    "full":    15,
    "half":    8,
    "quarter": 4,
    "none":    0,
}

VALID_LEVELS = frozenset(LEVEL_COUNTS.keys())


# ── Public API ───────────────────────────────────────────────────────────────

def generate_knowledge_assignment(
    levels: List[str],
    seed: int = 42,
) -> List[List[Tuple[str, int, str]]]:
    """Assign fixed knowledge items to each agent based on their experience level.

    Items are chosen from a single shuffled order (fixed by `seed`), so agents
    sharing the same level always know about the same items.

    Args:
        levels: List of knowledge level strings for each agent, e.g.
                ["none", "quarter", "half"].  Length == number of agents.
        seed:   Random seed for reproducibility.

    Returns:
        List (one entry per agent) of knowledge tuples:
            (item_name, ground_truth_rank, explanation)
        An agent with "none" level receives an empty list.

    Raises:
        ValueError: if an unknown level string is provided.
    """
    for lvl in levels:
        if lvl not in VALID_LEVELS:
            raise ValueError(
                f"Unknown knowledge level '{lvl}'. Valid levels: {VALID_LEVELS}"
            )

    # Create a fixed shuffled ordering of item indices
    rng = random.Random(seed)
    shuffled_indices: List[int] = list(range(len(ITEMS)))
    rng.shuffle(shuffled_indices)

    assignments: List[List[Tuple[str, int, str]]] = []
    for level in levels:
        count = LEVEL_COUNTS[level]
        if count == 0:
            assignments.append([])
        else:
            selected_indices = shuffled_indices[:count]
            knowledge = [
                (ITEMS[i], GROUND_TRUTH_RANKS[i], EXPLANATIONS[i])
                for i in selected_indices
            ]
            assignments.append(knowledge)

    return assignments


def format_knowledge_for_prompt(
    knowledge: List[Tuple[str, int, str]],
) -> str:
    """Convert an agent's knowledge list into a prompt-friendly string block.

    Args:
        knowledge: List of (item_name, ground_truth_rank, explanation) tuples.

    Returns:
        Formatted string, or a note saying the agent has no prior expertise.
    """
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


def knowledge_level_label(level: str) -> str:
    """Return a human-readable label for a knowledge level."""
    labels = {
        "full":    "Full knowledge (15/15 items)",
        "half":    "Half knowledge (8/15 items)",
        "quarter": "Quarter knowledge (4/15 items)",
        "none":    "No specialised knowledge",
    }
    return labels.get(level, level)

"""
Moon Survival Environment
=========================
Defines the core task: items, ground truth ranking, NASA expert explanations,
and evaluation utilities.
"""

import re
from typing import Dict, List, Optional, Tuple

# ── Items in the order used throughout the system ──────────────────────────
ITEMS: List[str] = [
    "Two 100-lb oxygen tanks",
    "20 liters of water",
    "Stellar map",
    "Food concentrate",
    "Solar-powered FM receiver-transmitter",
    "50 feet of nylon rope",
    "First aid kit with injection needles",
    "Parachute silk",
    "Self-inflating life raft",
    "Signal flares",
    "Two .45 caliber pistols",
    "One case of dehydrated milk",
    "Portable heating unit",
    "Magnetic compass",
    "Box of matches",
]

# Ground truth expert ranking (index i → rank for ITEMS[i], 1 = most important)
GROUND_TRUTH_RANKS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# NASA expert explanations, one per item (same order as ITEMS)
EXPLANATIONS: List[str] = [
    "The Moon's gravitational pull is one-sixth of Earth's, so the 100-lb oxygen tanks are only ~17 lb each. Humans obviously need oxygen to survive in the vacuum of the Moon.",
    "The 20 litres of water is critical for replacing the tremendous amount of liquid lost through dehydration on the sunlit lunar surface.",
    "The stellar map will be the primary navigation tool; star patterns appear almost identical to those seen from Earth, so standard navigation techniques apply.",
    "Food concentrate is an extremely efficient means of replacing calories and supplying energy for the 200-mile trek, and is light to carry.",
    "The solar-powered FM receiver-transmitter is the primary means of communication with the mother ship, essential for coordinating rescue efforts.",
    "The 50 feet of nylon rope is useful for scaling cliffs, securing the injured, and assisting movement across rough terrain.",
    "The first aid kit, including injection needles connected to vials of vitamins and medicines, fits through a special aperture in a NASA spacesuit.",
    "Parachute silk can protect the crew from the intense radiation and heat of direct sunlight on the lunar surface.",
    "The self-inflating life raft has a CO2 bottle that can be used for propulsion across the lunar surface.",
    "Signal flares are a primary means of signaling the mother ship or a rescue vessel.",
    "The two .45 caliber pistols can serve as a possible means of self-propulsion in the low-gravity environment.",
    "One case of dehydrated milk is a bulkier, less efficient version of the food concentrate.",
    "The portable heating unit is NOT needed because the crew is on the sunlit (light) side of the Moon, not the dark side.",
    "The magnetic compass is worthless on the Moon because the lunar magnetic field is not polarised — magnetic compasses do not work there.",
    "The box of matches is completely worthless on the Moon; there is no free oxygen to sustain combustion.",
]

# Maximum possible Sum of Absolute Differences (perfect reverse ranking, n=15)
MAX_SAD: int = sum(abs(i - (16 - i)) for i in range(1, 16))  # = 112


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_ranking(proposed_ranks: List[int]) -> int:
    """Compute the Sum of Absolute Differences (SAD) between a proposed
    ranking and the ground truth.

    Args:
        proposed_ranks: List of 15 integers where proposed_ranks[i] is the
                        rank assigned to ITEMS[i] (1 = most important).

    Returns:
        SAD score.  0 = perfect match, MAX_SAD = worst possible.

    Raises:
        ValueError: if the provided list does not contain exactly 15 ranks.
    """
    if len(proposed_ranks) != len(ITEMS):
        raise ValueError(
            f"Expected {len(ITEMS)} ranks, got {len(proposed_ranks)}."
        )
    return sum(abs(p - g) for p, g in zip(proposed_ranks, GROUND_TRUTH_RANKS))


def ranking_dict_to_list(ranking_dict: Dict[str, int]) -> List[int]:
    """Convert a {item_name: rank} dict to a list aligned with ITEMS order.

    Args:
        ranking_dict: Mapping from canonical item name to assigned rank.

    Returns:
        List of ranks in ITEMS order.

    Raises:
        ValueError: if any item is missing.
    """
    ranks: List[int] = []
    for item in ITEMS:
        if item not in ranking_dict:
            raise ValueError(f"Item missing from ranking: '{item}'")
        ranks.append(ranking_dict[item])
    return ranks


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_ranking(text: str) -> Dict[str, int]:
    """Extract a numbered ranking from an agent's free-form text response.

    Supports formats like "1. Item name" or "1) Item name".
    Uses fuzzy substring matching to map to canonical item names.

    Args:
        text: Raw string output from an LLM agent.

    Returns:
        Dictionary mapping canonical item name → assigned rank.

    Raises:
        ValueError: if a complete ranking of all 15 items cannot be parsed.
    """
    # Focus on the section after "MY RANKING:" if present
    upper = text.upper()
    marker = "MY RANKING:"
    if marker in upper:
        text = text[upper.index(marker) + len(marker):]

    ranking: Dict[str, int] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)[.)]\s*(.+)", line)
        if not match:
            continue
        rank = int(match.group(1))
        item_text = match.group(2).strip().lower()

        best_item: Optional[str] = None
        best_score: int = 0

        for item in ITEMS:
            if item in ranking:
                continue
            item_lower = item.lower()
            if item_lower == item_text:
                best_item = item
                break
            if item_text in item_lower:
                score = len(item_text)
                if score > best_score:
                    best_score = score
                    best_item = item
            elif item_lower in item_text:
                score = len(item_lower)
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item:
            ranking[best_item] = rank

    # ── Recovery: if exactly 1 item is missing, infer its rank ─────────────
    if len(ranking) == len(ITEMS) - 1:
        missing_items = [it for it in ITEMS if it not in ranking]
        used_ranks = set(ranking.values())
        all_ranks = set(range(1, len(ITEMS) + 1))
        missing_ranks = list(all_ranks - used_ranks)
        if len(missing_items) == 1 and len(missing_ranks) == 1:
            inferred_item = missing_items[0]
            inferred_rank = missing_ranks[0]
            print(
                f"  [parse_ranking] Inferred missing item '{inferred_item}' "
                f"→ rank {inferred_rank}"
            )
            ranking[inferred_item] = inferred_rank

    if len(ranking) != len(ITEMS):
        missing = [it for it in ITEMS if it not in ranking]
        raise ValueError(
            f"Incomplete ranking: parsed {len(ranking)}/{len(ITEMS)}.  "
            f"Missing: {missing}"
        )
    return ranking


# ── Helpers ─────────────────────────────────────────────────────────────────

def format_items_list() -> str:
    """Return the 15 items as a numbered list string (for use in prompts)."""
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(ITEMS))


def format_results_summary(
    candidates: List[Dict[str, int]],
    scores: List[int],
) -> str:
    """Format a list of candidate rankings and their scores for display in
    the next iteration's prompt.

    Args:
        candidates: List of {item_name: rank} dicts.
        scores: Corresponding SAD scores (same length).

    Returns:
        Formatted string block.
    """
    lines = ["Previous iteration results (lower score is better, 0 = perfect):"]
    for idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
        # Build ranked list sorted by rank value
        ranked = sorted(cand.items(), key=lambda x: x[1])
        ranked_str = ", ".join(f"{r}. {it}" for it, r in ranked)
        lines.append(f"  Candidate {idx} (SAD score = {score}): {ranked_str}")
    return "\n".join(lines)

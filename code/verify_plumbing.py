# -*- coding: utf-8 -*-
"""
verify_plumbing.py
==================
Sanity checks for the Moon Survival environment and knowledge manager.
Does NOT call the OpenAI API.

Run:
    python verify_plumbing.py
"""

import sys
from moon_survival_env import (
    ITEMS,
    GROUND_TRUTH_RANKS,
    evaluate_ranking,
    ranking_dict_to_list,
    parse_ranking,
    format_results_summary,
    MAX_SAD,
)
from knowledge_manager import generate_knowledge_assignment, format_knowledge_for_prompt

PASS_STR = "  PASS"
FAIL_STR = "  FAIL"
errors = 0


def check(condition, label):
    global errors
    if condition:
        print("{0}  {1}".format(PASS_STR, label))
    else:
        print("{0}  {1}  <- FAILED".format(FAIL_STR, label))
        errors += 1


print("\n=== moon_survival_env ===\n")

# 1. Correct number of items
check(len(ITEMS) == 15, "ITEMS has 15 entries (got {0})".format(len(ITEMS)))

# 2. Ground truth is identity permutation
check(
    GROUND_TRUTH_RANKS == list(range(1, 16)),
    "GROUND_TRUTH_RANKS is [1..15]"
)

# 3. Perfect ranking -> SAD = 0
perfect_ranks = list(range(1, 16))
check(evaluate_ranking(perfect_ranks) == 0, "evaluate_ranking: perfect = 0")

# 4. Completely reversed ranking -> SAD = MAX_SAD = 112
reversed_ranks = list(range(15, 0, -1))
sad = evaluate_ranking(reversed_ranks)
check(sad == 112, "evaluate_ranking: fully reversed = 112 (got {0})".format(sad))
check(MAX_SAD == 112, "MAX_SAD = 112 (got {0})".format(MAX_SAD))

# 5. Off-by-one ranking
off_by_one = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 15]
expected_obo = sum(abs(o - g) for o, g in zip(off_by_one, GROUND_TRUTH_RANKS))
check(
    evaluate_ranking(off_by_one) == expected_obo,
    "evaluate_ranking: off-by-one = {0}".format(expected_obo)
)

# 6. ranking_dict_to_list round-trip
ranking_dict = dict(zip(ITEMS, range(1, 16)))
as_list = ranking_dict_to_list(ranking_dict)
check(as_list == list(range(1, 16)), "ranking_dict_to_list: identity round-trip")

# 7. parse_ranking - standard format
sample_text = "MY RANKING:\n" + "\n".join(
    "{0}. {1}".format(i + 1, item) for i, item in enumerate(ITEMS)
)
try:
    parsed = parse_ranking(sample_text)
    check(len(parsed) == 15, "parse_ranking: 15 items from standard format")
    check(
        all(parsed[item] == rank for item, rank in zip(ITEMS, range(1, 16))),
        "parse_ranking: correct ranks from standard format"
    )
except ValueError as e:
    check(False, "parse_ranking raised: {0}".format(e))

# 8. parse_ranking - parenthesis format
paren_text = "\n".join("{0}) {1}".format(i + 1, item) for i, item in enumerate(ITEMS))
try:
    parsed2 = parse_ranking(paren_text)
    check(len(parsed2) == 15, "parse_ranking: 15 items from parenthesis format")
except ValueError as e:
    check(False, "parse_ranking (paren) raised: {0}".format(e))

# 9. format_results_summary
candidates = [ranking_dict]
scores = [0]
summary = format_results_summary(candidates, scores)
check("SAD score = 0" in summary, "format_results_summary: contains score")
check("Two 100-lb oxygen tanks" in summary, "format_results_summary: contains item name")

print("\n=== knowledge_manager ===\n")

# 10. half knowledge -> 8 items per agent
half_assignment = generate_knowledge_assignment(["half", "half", "half"])
for i, agent_know in enumerate(half_assignment):
    check(len(agent_know) == 8,
          "Agent {0} half knowledge -> 8 items (got {1})".format(i + 1, len(agent_know)))

# 11. quarter knowledge -> 4 items per agent
quarter_assignment = generate_knowledge_assignment(["quarter", "quarter", "quarter"])
for i, agent_know in enumerate(quarter_assignment):
    check(len(agent_know) == 4,
          "Agent {0} quarter knowledge -> 4 items (got {1})".format(i + 1, len(agent_know)))

# 12. none knowledge -> 0 items
none_assignment = generate_knowledge_assignment(["none"])
check(none_assignment[0] == [], "none knowledge -> empty list")

# 13. Same level -> same items
a1_half = generate_knowledge_assignment(["half"], seed=42)[0]
a2_half = generate_knowledge_assignment(["half"], seed=42)[0]
check(a1_half == a2_half, "Same level+seed -> identical knowledge")

# 14. Different seeds -> different items
a1_seed1 = generate_knowledge_assignment(["half"], seed=1)[0]
a1_seed2 = generate_knowledge_assignment(["half"], seed=2)[0]
check(a1_seed1 != a1_seed2, "Different seeds -> different knowledge sets")

# 15. Mixed strategies (none, quarter, half)
mixed = generate_knowledge_assignment(["none", "quarter", "half"])
check(len(mixed[0]) == 0, "Mixed: agent 1 = none -> 0 items")
check(len(mixed[1]) == 4, "Mixed: agent 2 = quarter -> 4 items")
check(len(mixed[2]) == 8, "Mixed: agent 3 = half -> 8 items")

# 16. quarter items are a subset of half items (same seed)
half_items = set(item for item, _, _ in generate_knowledge_assignment(["half"], seed=42)[0])
quarter_items = set(item for item, _, _ in generate_knowledge_assignment(["quarter"], seed=42)[0])
check(quarter_items.issubset(half_items), "quarter items are subset of half items (same seed)")

# 17. format_knowledge_for_prompt for none
prompt_none = format_knowledge_for_prompt([])
check("no specialised" in prompt_none.lower(), "format_knowledge_for_prompt: none -> appropriate message")

# 18. format_knowledge_for_prompt for half contains item names
half_knowledge = generate_knowledge_assignment(["half"])[0]
prompt_half = format_knowledge_for_prompt(half_knowledge)
first_item_name = half_knowledge[0][0]
check(first_item_name in prompt_half, "format_knowledge_for_prompt: half contains item name")

# -- Report --
print("\n" + "-" * 40)
if errors == 0:
    print("  All checks passed!")
else:
    print("  {0} check(s) FAILED.".format(errors))
    sys.exit(1)

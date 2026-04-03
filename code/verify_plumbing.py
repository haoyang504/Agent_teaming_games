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
from knowledge_manager import (
    generate_knowledge_assignment,
    format_knowledge_for_prompt,
    knowledge_level_label,
    format_knowledge_counts,
)

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

# 10. Config 1 [0, 2, 4] — correct counts
cfg1 = generate_knowledge_assignment([0, 2, 4], seed=42)
check(len(cfg1[0]) == 0, "Config 1: Agent A -> 0 items (got {0})".format(len(cfg1[0])))
check(len(cfg1[1]) == 2, "Config 1: Agent B -> 2 items (got {0})".format(len(cfg1[1])))
check(len(cfg1[2]) == 4, "Config 1: Agent C -> 4 items (got {0})".format(len(cfg1[2])))

# 11. Config 2 [1, 2, 3] — correct counts
cfg2 = generate_knowledge_assignment([1, 2, 3], seed=42)
check(len(cfg2[0]) == 1, "Config 2: Agent A -> 1 item (got {0})".format(len(cfg2[0])))
check(len(cfg2[1]) == 2, "Config 2: Agent B -> 2 items (got {0})".format(len(cfg2[1])))
check(len(cfg2[2]) == 3, "Config 2: Agent C -> 3 items (got {0})".format(len(cfg2[2])))

# 12. Config 3 [2, 2, 2] — correct counts
cfg3 = generate_knowledge_assignment([2, 2, 2], seed=42)
for i in range(3):
    check(len(cfg3[i]) == 2, "Config 3: Agent {0} -> 2 items (got {1})".format(i, len(cfg3[i])))

# 13. Nested overlap: A ⊆ B ⊆ C
nested = generate_knowledge_assignment([1, 2, 3], "all", "nested", seed=42)
items_a = set(item for item, _, _ in nested[0])
items_b = set(item for item, _, _ in nested[1])
items_c = set(item for item, _, _ in nested[2])
check(items_a <= items_b, "Nested: A ⊆ B")
check(items_b <= items_c, "Nested: B ⊆ C")

# 14. Disjoint overlap: no overlap
disjoint = generate_knowledge_assignment([1, 2, 3], "all", "disjoint", seed=42)
items_a = set(item for item, _, _ in disjoint[0])
items_b = set(item for item, _, _ in disjoint[1])
items_c = set(item for item, _, _ in disjoint[2])
check(items_a & items_b == set(), "Disjoint: A ∩ B = ∅")
check(items_a & items_c == set(), "Disjoint: A ∩ C = ∅")
check(items_b & items_c == set(), "Disjoint: B ∩ C = ∅")

# 15. Same seed -> same items
a1 = generate_knowledge_assignment([2, 2, 2], seed=42)
a2 = generate_knowledge_assignment([2, 2, 2], seed=42)
check(a1 == a2, "Same counts+seed -> identical knowledge")

# 16. Different seeds -> different items
a1_seed1 = generate_knowledge_assignment([2, 2, 2], seed=1)
a1_seed2 = generate_knowledge_assignment([2, 2, 2], seed=2)
check(a1_seed1 != a1_seed2, "Different seeds -> different knowledge sets")

# 17. Source filtering — top
top = generate_knowledge_assignment([2, 2, 4], "top", "disjoint", seed=42)
for i, agent_know in enumerate(top):
    for item, rank, _ in agent_know:
        check(rank <= 8, "Source=top: Agent {0} item '{1}' rank {2} <= 8".format(i, item, rank))

# 18. Source filtering — bottom
bottom = generate_knowledge_assignment([2, 2, 4], "bottom", "disjoint", seed=42)
for i, agent_know in enumerate(bottom):
    for item, rank, _ in agent_know:
        check(rank >= 8, "Source=bottom: Agent {0} item '{1}' rank {2} >= 8".format(i, item, rank))

# 19. format_knowledge_for_prompt for empty
prompt_none = format_knowledge_for_prompt([])
check("no specialised" in prompt_none.lower(), "format_knowledge_for_prompt: empty -> appropriate message")

# 20. format_knowledge_for_prompt contains item names
knowledge_items = generate_knowledge_assignment([4, 4, 4], seed=42)[0]
prompt = format_knowledge_for_prompt(knowledge_items)
first_item_name = knowledge_items[0][0]
check(first_item_name in prompt, "format_knowledge_for_prompt: contains item name")

# 21. knowledge_level_label
check(knowledge_level_label(4) == "4/15 items", "knowledge_level_label(4) = '4/15 items'")
check(knowledge_level_label(0) == "0/15 items", "knowledge_level_label(0) = '0/15 items'")

# 22. format_knowledge_counts
check(
    format_knowledge_counts([0, 2, 4]) == "Agent A: 0, Agent B: 2, Agent C: 4",
    "format_knowledge_counts([0, 2, 4])"
)

print("\n=== agent.py — Setting prompts ===\n")

from agent import MoonSurvivalAgent

# 23. Setting 1: no team info, no leader in prompt
agent_s1 = MoonSurvivalAgent(
    agent_id=1,
    knowledge=[("Two 100-lb oxygen tanks", 1, "Oxygen is needed.")],
    client=None,
    model="test",
    num_agents=3,
    team_knowledge_info=None,
    leader_id=None,
)
prompt_s1 = agent_s1._system_prompt()
check("TEAM INFORMATION" not in prompt_s1, "Setting 1: no team info in prompt")
check("TEAM ROLE" not in prompt_s1, "Setting 1: no leader in prompt")

# 24. Setting 2: team info present, no leader
agent_s2 = MoonSurvivalAgent(
    agent_id=1,
    knowledge=[("Two 100-lb oxygen tanks", 1, "Oxygen is needed.")],
    client=None,
    model="test",
    num_agents=3,
    team_knowledge_info={"A": 0, "B": 2, "C": 4},
    leader_id=None,
)
prompt_s2 = agent_s2._system_prompt()
check("TEAM INFORMATION" in prompt_s2, "Setting 2: team info present")
check("TEAM ROLE" not in prompt_s2, "Setting 2: no leader in prompt")
check("Agent A" in prompt_s2 and "0" in prompt_s2, "Setting 2: Agent A count in prompt")
check("Agent C" in prompt_s2 and "4" in prompt_s2, "Setting 2: Agent C count in prompt")

# 25. Setting 3: no team info, leader present
agent_s3 = MoonSurvivalAgent(
    agent_id=2,
    knowledge=[("20 liters of water", 2, "Water is critical.")],
    client=None,
    model="test",
    num_agents=3,
    team_knowledge_info=None,
    leader_id=3,
)
prompt_s3 = agent_s3._system_prompt()
check("TEAM INFORMATION" not in prompt_s3, "Setting 3: no team info")
check("TEAM ROLE" in prompt_s3, "Setting 3: leader present")
check("Agent C" in prompt_s3 and "leader" in prompt_s3.lower(), "Setting 3: Agent C is leader")

# 26. Setting 4: both present
agent_s4 = MoonSurvivalAgent(
    agent_id=1,
    knowledge=[],
    client=None,
    model="test",
    num_agents=3,
    team_knowledge_info={"A": 0, "B": 2, "C": 4},
    leader_id=3,
)
prompt_s4 = agent_s4._system_prompt()
check("TEAM INFORMATION" in prompt_s4, "Setting 4: team info present")
check("TEAM ROLE" in prompt_s4, "Setting 4: leader present")

# 27. Team info and role sections are neutral (no instructional language)
# Extract only the TEAM INFORMATION and TEAM ROLE sections for checking
def extract_setting_sections(prompt):
    sections = ""
    for header in ["=== TEAM INFORMATION ===", "=== TEAM ROLE ==="]:
        if header in prompt:
            start = prompt.index(header)
            end = prompt.index("===", start + len(header))
            sections += prompt[start:end]
    return sections.lower()

s2_sections = extract_setting_sections(prompt_s2)
s4_sections = extract_setting_sections(prompt_s4)
for bad_phrase in ["defer to", "act based on", "follow the leader", "consider this when", "use this information"]:
    check(bad_phrase not in s2_sections, "Setting 2: no instructional phrase '{0}' in team sections".format(bad_phrase))
    check(bad_phrase not in s4_sections, "Setting 4: no instructional phrase '{0}' in team sections".format(bad_phrase))

# -- Report --
print("\n" + "-" * 40)
if errors == 0:
    print("  All checks passed!")
else:
    print("  {0} check(s) FAILED.".format(errors))
    sys.exit(1)

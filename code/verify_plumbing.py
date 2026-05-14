# -*- coding: utf-8 -*-
"""
verify_plumbing.py
==================
Sanity checks for the Moon Survival environment and knowledge manager.
Does NOT call the OpenAI API.

Run:
    python verify_plumbing.py
"""

import os
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
nested = generate_knowledge_assignment([1, 2, 3], "nested", seed=42)
items_a = set(item for item, _, _ in nested[0])
items_b = set(item for item, _, _ in nested[1])
items_c = set(item for item, _, _ in nested[2])
check(items_a <= items_b, "Nested: A ⊆ B")
check(items_b <= items_c, "Nested: B ⊆ C")

# 14. Disjoint overlap: no overlap
disjoint = generate_knowledge_assignment([1, 2, 3], "disjoint", seed=42)
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
prompt_s1 = agent_s1._static_context()
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
prompt_s2 = agent_s2._static_context()
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
prompt_s3 = agent_s3._static_context()
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
prompt_s4 = agent_s4._static_context()
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

# 28. Leader responsibility sentence: required in Settings 3/4, absent in Settings 1/2
leader_sentence = "The leader is responsible for guiding the discussion."
check(leader_sentence not in prompt_s1, "Setting 1: no leader responsibility sentence")
check(leader_sentence not in prompt_s2, "Setting 2: no leader responsibility sentence")
check(leader_sentence in prompt_s3, "Setting 3: leader responsibility sentence present")
check(leader_sentence in prompt_s4, "Setting 4: leader responsibility sentence present")

# 29. Consensus sentence (TEAM PROCESS) is in every setting, count templated
consensus_marker = "aim to reach a consensus through"
for label, p in [("S1", prompt_s1), ("S2", prompt_s2), ("S3", prompt_s3), ("S4", prompt_s4)]:
    check("TEAM PROCESS" in p, "{0}: TEAM PROCESS section present".format(label))
    check(consensus_marker in p, "{0}: consensus sentence present".format(label))

# 30. num_discussion_rounds is interpolated into the consensus sentence
agent_rounds = MoonSurvivalAgent(
    agent_id=1,
    knowledge=[],
    client=None,
    model="test",
    num_agents=3,
    num_discussion_rounds=5,
)
prompt_rounds = agent_rounds._static_context()
check("through 5 rounds of discussion" in prompt_rounds, "num_discussion_rounds=5 templated into prompt")

# ── Phase 9: per-phase system prompts ──────────────────────────────────────
print("\n=== Phase 9: per-phase system prompts ===\n")

agent_p9 = MoonSurvivalAgent(
    agent_id=2, knowledge=[],
    client=None, model="test", num_agents=3,
    team_knowledge_info={"A": 0, "B": 2, "C": 4}, leader_id=3,
)
propose = agent_p9._propose_system_prompt()
discuss = agent_p9._discuss_system_prompt()
final = agent_p9._final_select_system_prompt(k=3)

# Propose system prompt
check("Each ranking must include ALL 15 items" in propose, "Propose has rule 1 stem")
check("COMMON REASONING" in propose, "Propose has COMMON REASONING block")
check("parachute silk" in propose, "Propose has the worked example")

# Discuss system prompt
check("Refer to candidates by their IDs" in discuss, "Discuss has rule 1 stem")
check("real deliberation, not a ceremony" in discuss, "Discuss has deliberation language")
check("do not defer out of politeness" in discuss, "Discuss has anti-politeness language")
check("Do NOT output a full ranking" in discuss, "Discuss is commentary-only")

# Final-select system prompt
check("Output exactly 3 final rankings" in final, "Final has k substituted")
check("full authority over the final submission" in final, "Final has authority language")

# Propose task block: None-branch wording must be correct under F2/F3
# non-feedback iterations too (not just true iter 1). Regression check.
none_branch = agent_p9._propose_task_block(k=3, previous_results=None, results_context="ignored")
lower = none_branch.lower()
check("first iteration" not in lower, "Propose None-branch avoids 'FIRST iteration' wording")
check("No prior results exist" not in none_branch, "Propose None-branch avoids 'No prior results exist' wording")
check("No prior results are shown" in none_branch, "Propose None-branch uses neutral 'No prior results are shown' wording")

# Shared footer present in all three
footer_signature = "Your own prior turns appear with role=assistant"
for name, prompt in [("propose", propose), ("discuss", discuss), ("final_select", final)]:
    check(footer_signature in prompt, "{0} prompt has footer".format(name))

# display_name derivation
check(agent_p9.display_name == "Agent2", "display_name derived as Agent{agent_id}")

# ── Phase 9: _thread_prior_turns role tagging ──────────────────────────────
print("\n=== Phase 9: _thread_prior_turns ===\n")

turns = [
    {"agent": "Agent1", "phase": "propose", "round_num": None, "raw_output": "<A1's text>"},
    {"agent": "Agent2", "phase": "propose", "round_num": None, "raw_output": "<MY text>"},
    {"agent": "Agent3", "phase": "propose", "round_num": None, "raw_output": "<A3's text>"},
]
threaded = agent_p9._thread_prior_turns(turns)
check(
    threaded[0]["role"] == "user" and threaded[0].get("name") == "Agent1",
    "Peer turn → user with name",
)
check(
    threaded[1]["role"] == "assistant" and "name" not in threaded[1],
    "Own turn → assistant, no name",
)
check(
    threaded[2]["role"] == "user" and threaded[2].get("name") == "Agent3",
    "Other peer → user with name",
)

# ── Phase 9: Logger writes expected files ───────────────────────────────────
print("\n=== Phase 9: Logger ===\n")

import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    from logger import Logger
    log = Logger("test_exp", tmpdir)
    log.save_prompt_and_output(
        phase="propose", iteration=1, agent_name="Agent1",
        formatted_messages="[system]\nfoo\n", response_text="bar",
    )
    p = os.path.join(tmpdir, "test_exp_raw", "prompts", "iter1_propose_Agent1.txt")
    o = os.path.join(tmpdir, "test_exp_raw", "outputs", "iter1_propose_Agent1.txt")
    check(os.path.exists(p), "Logger wrote prompt file")
    check(os.path.exists(o), "Logger wrote output file")
    log.save_prompt_and_output(
        phase="discuss", iteration=2, agent_name="Agent2",
        formatted_messages="x", response_text="y", round_num=3,
    )
    p2 = os.path.join(tmpdir, "test_exp_raw", "prompts", "iter2_round3_Agent2.txt")
    check(os.path.exists(p2), "Logger uses round_num for discuss phase filenames")

# ── Phase 9: parser tolerates COMMON REASONING preamble ────────────────────
print("\n=== Phase 9: parser with COMMON REASONING preamble ===\n")

from agent import _parse_k_candidates, _extract_per_candidate_reasoning
fake_response = """COMMON REASONING:
Some pattern-level reasoning here.

CANDIDATE 1 RANKING:
Reasoning: First candidate.
1. Two 100-lb oxygen tanks
2. 20 liters of water
3. Stellar map
4. Food concentrate
5. Solar-powered FM receiver-transmitter
6. 50 feet of nylon rope
7. First aid kit with injection needles
8. Parachute silk
9. Self-inflating life raft
10. Signal flares
11. Two .45 caliber pistols
12. One case of dehydrated milk
13. Portable heating unit
14. Magnetic compass
15. Box of matches
"""
candidates = _parse_k_candidates(fake_response, k=1, label="CANDIDATE")
check(len(candidates) == 1, "Parser handles COMMON REASONING preamble")
check(len(candidates[0]) == 15, "All 15 items parsed despite preamble")

reasonings = _extract_per_candidate_reasoning(fake_response, k=1)
check(len(reasonings) == 1 and reasonings[0] == "First candidate.",
      "Per-candidate reasoning extracted")

# -- Report --
print("\n" + "-" * 40)
if errors == 0:
    print("  All checks passed!")
else:
    print("  {0} check(s) FAILED.".format(errors))
    sys.exit(1)

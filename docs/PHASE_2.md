# Phase 2 — Hierarchy: Team Knowledge Info + Leader Role

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first. Phase 1 must be complete — the new knowledge assignment system (counts/source/overlap) and `--setting` CLI parameter must be working.
> **Goal:** Wire up the `--setting` parameter (1–4) so that agents receive team knowledge info and/or leader role info in their system prompts. After this phase, all 4 experimental settings should produce different agent behavior.

---

## What exists now

The `--setting` parameter is accepted by the CLI, passed through `run_experiment()` in `experiment_runner.py`, stored in the JSON log, but has **no effect on agent behavior**. The `MoonSurvivalAgent` class in `agent.py` has no concept of team info or leader roles. Its constructor takes `agent_id`, `knowledge`, `client`, `model`, and `num_agents`.

The `_system_prompt()` method builds a system message with: agent identity, task context, 15 items list, specialized knowledge block, scoring explanation, and general guidelines. There are no sections for team information or role designation.

---

## Task 2A: Modify `agent.py` — DONE

### Constructor changes

Add two new optional parameters to `MoonSurvivalAgent.__init__()`:

```python
def __init__(
    self,
    agent_id: int,
    knowledge: List[Tuple[str, int, str]],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    num_agents: int = 3,
    team_knowledge_info: Optional[Dict[str, int]] = None,  # NEW
    leader_id: Optional[int] = None,                        # NEW
) -> None:
```

**`team_knowledge_info`** — A dictionary mapping agent labels to their knowledge counts. Example: `{"A": 0, "B": 2, "C": 4}`. When provided (Settings 2 and 4), a team information section is added to the system prompt. When `None` (Settings 1 and 3), no team info is shown.

**`leader_id`** — The agent_id of the designated leader. Example: `3` (meaning Agent C). When provided (Settings 3 and 4), a team role section is added to the system prompt. When `None` (Settings 1 and 2), no role info is shown.

Store both as instance attributes.

### System prompt changes

Modify `_system_prompt()` to conditionally add two new sections **after** the `=== YOUR SPECIALISED KNOWLEDGE ===` block and **before** the `=== SCORING ===` block.

**Team information section** (when `self.team_knowledge_info` is not None):

```
=== TEAM INFORMATION ===
Your team members have the following levels of specialised knowledge:
- Agent A knows about {n} of the 15 items.
- Agent B knows about {n} of the 15 items.
- Agent C knows about {n} of the 15 items.
```

Where `{n}` is replaced with the actual count for each agent. List ALL agents including the current one. Use "has no specialised knowledge" instead of "knows about 0 of the 15 items" when the count is 0.

**CRITICAL: This must be purely factual. No instructions like "consider this when discussing", "defer to agents with more knowledge", or "act based on this information." Just state the facts.**

**Team role section** (when `self.leader_id` is not None):

```
=== TEAM ROLE ===
Agent {label} has been designated as the team leader.
```

Where `{label}` is the letter label (A/B/C) corresponding to the leader's agent_id (1→A, 2→B, 3→C).

**CRITICAL: No explanation of why they were chosen. No behavioral instructions like "follow the leader's guidance" or "the leader should coordinate." Just the designation.**

### Mapping agent_id to label

Add a helper method or use a simple mapping: `{1: "A", 2: "B", 3: "C"}`. Use this consistently in both the team info and leader role sections.

---

## Task 2B: Modify `experiment_runner.py` — DONE

### Changes to `run_experiment()`

After generating `knowledge_assignments` and before creating agents, add logic to compute what each agent should receive based on the `setting` parameter:

```python
# ── Compute setting-dependent info ─────────────────────────────────
team_knowledge_info = None
leader_id = None

if setting in (2, 4):
    # Team knowledge info: tell each agent how many items everyone knows
    team_knowledge_info = {
        "A": counts[0],
        "B": counts[1],
        "C": counts[2],
    }

if setting in (3, 4):
    # Leader role: Agent C (agent_id=3) is always the leader
    leader_id = 3
```

Then pass these to the agent constructor:

```python
agents = [
    MoonSurvivalAgent(
        agent_id=i + 1,
        knowledge=knowledge_assignments[i],
        client=client,
        model=model,
        num_agents=3,
        team_knowledge_info=team_knowledge_info,  # NEW
        leader_id=leader_id,                       # NEW
    )
    for i in range(3)
]
```

### Logging

Add `team_knowledge_info` and `leader_id` to the `experiment_log` dict so the JSON output captures what each setting provided.

Add a verbose print line after the existing knowledge dump:

```python
if setting >= 2:
    print(f"  Team info provided: {team_knowledge_info is not None}")
    print(f"  Leader designated: Agent {leader_id} (id={leader_id})" if leader_id else "  No leader designated")
```

---

## Task 2C: Update `verify_plumbing.py` — DONE

Add checks to verify the setting logic. These should NOT call the OpenAI API — they should test prompt construction only.

Since `_system_prompt()` is a method on `MoonSurvivalAgent` which requires an OpenAI client, create a mock/dummy approach. The simplest way: instantiate `MoonSurvivalAgent` with `client=None` (it won't be used since we're only calling `_system_prompt()`), then inspect the returned string.

Add these checks:

```python
# 23. Setting 1: no team info, no leader in prompt
from agent import MoonSurvivalAgent
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

# 27. Team info is neutral (no instructional language)
for bad_phrase in ["defer", "act based", "follow", "consider this", "use this"]:
    check(bad_phrase not in prompt_s2.lower(), f"Setting 2: no instructional phrase '{bad_phrase}'")
    check(bad_phrase not in prompt_s4.lower(), f"Setting 4: no instructional phrase '{bad_phrase}'")
```

---

## Tests

### Test 1: Prompt verification

Run `python verify_plumbing.py` and confirm all checks pass, including the new Setting 1–4 checks.

### Test 2: Print prompts for manual inspection

Create a quick script or run inline:

```python
from agent import MoonSurvivalAgent

for setting, tki, lid in [
    (1, None, None),
    (2, {"A": 0, "B": 2, "C": 4}, None),
    (3, None, 3),
    (4, {"A": 0, "B": 2, "C": 4}, 3),
]:
    agent = MoonSurvivalAgent(
        agent_id=1,
        knowledge=[("Two 100-lb oxygen tanks", 1, "Oxygen is critical.")],
        client=None, model="test", num_agents=3,
        team_knowledge_info=tki, leader_id=lid,
    )
    print(f"\n{'='*60}")
    print(f"  SETTING {setting}")
    print(f"{'='*60}")
    print(agent._system_prompt())
```

Manually verify:
- Setting 1: Only knowledge block, no team info, no role
- Setting 2: Knowledge block + team info section (neutral wording), no role
- Setting 3: Knowledge block + role section (just "Agent C has been designated as the team leader."), no team info
- Setting 4: Knowledge block + team info + role section
- None of the prompts contain instructional language about how to use the info

### Test 3: End-to-end pipeline (requires API key)

Run 4 quick experiments, one per setting:

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1
python run_experiment.py --config high_div --setting 2 --k 1 --iterations 1 --discussion-rounds 1
python run_experiment.py --config high_div --setting 3 --k 1 --iterations 1 --discussion-rounds 1
python run_experiment.py --config high_div --setting 4 --k 1 --iterations 1 --discussion-rounds 1
```

Check the JSON logs to confirm `setting`, `team_knowledge_info`, and `leader_id` are recorded. You don't need to compare SAD scores — just verify the pipeline completes without errors across all 4 settings.

---

## What NOT to do in Phase 2

- Do NOT change the discussion flow, speaking order, or iteration logic — that's Phase 3.
- Do NOT implement feedback frequency/mode — that's Phase 4.
- Do NOT implement incorrect knowledge — that's Phase 5.
- Do NOT modify `knowledge_manager.py` or `moon_survival_env.py`.
- Do NOT add any behavioral instructions to the team info or role sections. The design explicitly requires neutral, factual information only.
- Do NOT make the leader the final decision-maker in Phase 3's `select_final_candidates()`. Leadership is purely informational.

---

## Implementation Notes (post-completion)

### Neutrality check refinement

The PHASE_2 spec included a neutrality test scanning for instructional phrases like "follow", "use this" in the team sections. The base knowledge prompt (`format_knowledge_for_prompt`) already contains "the following items. Use this to inform your rankings" — which triggered false positives. The test was refined to check only the extracted `=== TEAM INFORMATION ===` and `=== TEAM ROLE ===` sections, and to use more specific phrases ("defer to", "follow the leader", "use this information") to avoid matching legitimate descriptive language.

### Prompt structure

The two new sections are injected between `=== YOUR SPECIALISED KNOWLEDGE ===` and `=== SCORING ===`. When both are present (Setting 4), the order is: team information, then team role. Both sections appear only when their corresponding data is non-None.

### All 4 settings verified

- Setting 1: no team info, no role section — prompt identical to Phase 1
- Setting 2: `=== TEAM INFORMATION ===` with neutral counts, no role
- Setting 3: `=== TEAM ROLE ===` with "Agent C has been designated as the team leader.", no team info
- Setting 4: both sections present
- `verify_plumbing.py` has 10 new checks (23–27) covering all 4 settings plus neutrality

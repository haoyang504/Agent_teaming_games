# Moon Survival Experiment — Context Document

> This document provides shared context for all implementation phases. Read this first before any phase task document.

---

## Research Objective

Three LLM agents collaborate to rank 15 survival items for a 200-mile lunar trek (NASA Moon Survival task). Rankings are scored against the NASA expert answer using **Sum of Absolute Differences (SAD)**. Lower = better, 0 = perfect, 112 = worst.

The experiment tests how **information asymmetry**, **hierarchy**, and **incorrect knowledge** affect group decision-making.

---

## The Four Experimental Settings

| Setting | Team Info | Role Info | What Agents Are Told |
|---------|-----------|-----------|---------------------|
| 1 | No | No | Only their own knowledge |
| 2 | Yes | No | Own knowledge + how many items each teammate knows |
| 3 | No | Yes | Own knowledge + who the leader is (with a brief responsibility note) |
| 4 | Yes | Yes | Own knowledge + teammate counts + leader |

- **Team info** = telling each agent how many items the other agents know (just the count, not which items). Worded neutrally — no instructions like "defer to" or "act based on this."
- **Role info** = telling all agents that Agent C is the designated leader. Phase 8 adds one instructional sentence: *"The leader is responsible for guiding the discussion."* Other than that, no behavioral directives.
- **Consensus framing**: every system prompt (all 4 settings) includes a single sentence — *"The agents aim to reach a consensus through N rounds of discussion."* The number of rounds is templated from `num_discussion_rounds`.

### Structural caveat (Setting 1 leader leak)

Mechanically, **Agent C selects the final candidates in every setting**, including Setting 1 (which has no leader prompt). This was accepted by the research lead in the Phase 8 review: as long as agents are told they should reach a consensus and Agent C speaks last in the ABC order, C-as-final-selector is a natural consequence. Setting 1 is therefore "no *prompted* leader," not "no leader at all."

---

## Knowledge Assignment (2 Dimensions)

> Source variants (top/bottom of the ranking pool) were removed in Phase 8. Source is implicitly always the full 15 items.

### Dimension 1: Team Configurations (how many items each agent knows)

| Config | Agent A | Agent B | Agent C | Diversity |
|--------|---------|---------|---------|-----------|
| high_div | 0 items | 2 items | 4 items | High |
| low_div  | 2 items | 2 items | 2 items | Low (equal) |

Agent C always has the most (or equal) knowledge and is the designated leader in Settings 3 & 4. In `low_div` (equal), Agent C is still leader by convention.

### Dimension 2: Overlap Patterns (how knowledge sets relate across agents)

Agent C is always sampled first, then B and A relative to C.

| Pattern | Constraint |
|---------|------------|
| `nested` | A ⊆ B ⊆ C |
| `disjoint` | A, B, C pairwise disjoint |
| `O3` | B ⊆ C; A disjoint from **both B and C** |
| `O4` | A ⊆ C; B disjoint from **both A and C** |
| `O5` | A = B (smaller is a subset of the larger when counts differ); C disjoint from both |

Nominal sweep = 2 configs × 5 overlaps = 10 conditions. Several collapse when `count=0` for an agent — see `docs/PHASE_8.md` Task 8H for the explicit unique-conditions table (**7 unique conditions**).

---

## Feedback Settings (3 modes)

> F4 and F5 (full-history modes) were removed in Phase 8.

| Mode | Type | Description |
|------|------|-------------|
| F1 | Frequency | SAD scores from last iteration, every iteration (baseline) |
| F2 | Frequency | SAD scores from last iteration, every 2 iterations |
| F3 | Frequency | SAD scores from last iteration, every 4 iterations |

Iteration 1 never receives feedback regardless of mode.

---

## Incorrect Knowledge Phase

> Phase 8 rewrote this end-to-end. The old "append 1 incorrect item" semantics are gone.

### Fixed incorrect ranking

One canonical incorrect ranking is persisted to `code/incorrect_ranking.json` and reused across runs. It is a derangement of the 15 ground-truth ranks (no item-rank pair matches truth). SAD vs. truth = 64/112. `generate_incorrect_ranking()` loads from this file; it only falls back to derangement generation if the file is missing.

### Pattern semantics (replacement, not append)

Incorrect items **replace** correct items at the same per-agent count. The agent's total knowledge count is unchanged.

| Pattern | Behavior |
|---------|----------|
| `I1` | Agent A's items are all incorrect; B and C unchanged |
| `I2` | Agent B's items are all incorrect; A and C unchanged |
| `I3` | Agent C's items are all incorrect; A and B unchanged |
| `I4` | Each agent has half-correct + half-incorrect (incorrect side is ceiling for odd counts) |

For `high_div` `[0,2,4]`:

| Pattern | A (count=0) | B (count=2) | C (count=4) |
|---------|---|---|---|
| I1 | 0 incorrect (vacuous; equivalent to no-incorrect baseline) | 2 correct | 4 correct |
| I2 | 0 | 2 incorrect | 4 correct |
| I3 | 0 | 2 correct | 4 incorrect |
| I4 | 0 | 1 correct + 1 incorrect | 2 correct + 2 incorrect |

For `low_div` `[2,2,2]`:

| Pattern | A | B | C |
|---------|---|---|---|
| I1 | 2 incorrect | 2 correct | 2 correct |
| I2 | 2 correct | 2 incorrect | 2 correct |
| I3 | 2 correct | 2 correct | 2 incorrect |
| I4 | 1 + 1 | 1 + 1 | 1 + 1 |

**Overlap constraint:** for I4, no incorrect (item, rank) tuple may share its item OR its rank with any correct tuple held by the same agent.

**Warning condition:** with or without a prompt line telling agents some knowledge may be incorrect (`--incorrect-warning`).

---

## Codebase Structure

```
run_experiment.py              Entry point; CLI args, CONFIGS dict (2 configs)
  └── experiment_runner.py     Orchestration; 4-phase iteration loop
        ├── agent.py           LLM agent; system prompt, propose, discuss, select
        │     ├── moon_survival_env.py   Items, ground truth, scoring, parsing
        │     └── knowledge_manager.py   Assigns expert items to agents
        ├── moon_survival_env.py         Evaluation + formatting of results
        └── knowledge_manager.py         2-dimensional knowledge assignment
  └── config.py                API key + Portkey model
run_all_combinations.py        Sweeps the 7 unique conditions
verify_plumbing.py             Offline sanity checks (no API calls)
test_knowledge.py              Unit tests for knowledge assignment
metrics.py                     Standalone analysis: novelty, recombination, dominance, contribution share
incorrect_ranking.json         Persisted fixed incorrect ranking (Phase 8)
```

### Iteration Flow (4 Phases)

1. **Proposals** — Each agent proposes `k` candidate rankings (each scored individually for per-agent fluctuation data)
2. **Discussion** — Round-robin discussion in fixed order (`ABC` or `CBA`)
3. **Final Selection** — Agent C selects `k` final candidates from discussion
4. **Evaluation** — Final candidates scored against NASA ground truth (SAD)

### Key Files

- **moon_survival_env.py** — Do NOT modify. Contains items, ground truth, scoring, parsing.
- **knowledge_manager.py** — 2-dimensional knowledge distribution: `generate_knowledge_assignment(counts, overlap, seed)`. Supports `nested`, `disjoint`, `O3`, `O4`, `O5` overlap patterns. Agent C is always sampled first. Also provides `generate_incorrect_ranking()` (loads from `incorrect_ranking.json`) and `generate_mixed_knowledge_assignment(counts, incorrect_pattern, seed, incorrect_seed)` with replacement-at-same-count semantics and per-agent item/rank overlap constraints.
- **agent.py** — LLM agent class. Constructor accepts `num_discussion_rounds` (for the TEAM PROCESS prompt), `team_knowledge_info` (dict or None), `leader_id` (int or None), and `knowledge_warning` (bool). `_system_prompt()` conditionally injects team info, leader role (with responsibility sentence in Settings 3/4), unconditional TEAM PROCESS consensus sentence, and knowledge warning. `propose_candidates(k, previous_results, results_context)`, `discuss()`, and `select_final_candidates()` are the action methods.
- **experiment_runner.py** — `run_experiment(counts, overlap, setting, feedback_mode, discussion_order, ...)`. Uses Portkey AI gateway. Computes `team_knowledge_info` and `leader_id` from the `setting` parameter (Settings 2/4 → team info; Settings 3/4 → leader_id=3). Discussion speaking order is `"ABC"` (1→2→3) or `"CBA"` (3→2→1). Discussion history resets each iteration (`history_scope: "last_iteration_only"`). Feedback modes F1–F3 control when evaluation data is passed. Each proposal candidate is scored individually so per-agent fluctuation can be analyzed offline. All settings, discussion orders, and feedback state are logged in JSON output.
- **run_experiment.py** — CLI entry point. `CONFIGS` has `high_div` and `low_div`. CLI args: `--config`, `--counts`, `--overlap`, `--setting`, `--feedback-mode`, `--discussion-order`, `--incorrect-pattern`, `--incorrect-warning`, `--incorrect-seed`. Defaults: `--iterations 3` (full protocol: 20), `--discussion-rounds 3`, `--feedback-mode F1`, `--discussion-order ABC`.
- **metrics.py** — Per-iteration: novelty ratio, recombination score, dominance (pair-origin attribution), `contribution_share_pct` (% of final ranking from each agent's proposals), per-agent proposal SAD. Summary: overall versions of each. CLI: `python metrics.py <log_file> --verbose`.

---

## Implementation Phases Overview

| Phase | What | Status |
|-------|------|--------|
| 1 | Foundation — new knowledge system + setting parameter | **Done** |
| 2 | Hierarchy — team info + leader role in prompts | **Done** |
| 3 | Discussion mechanics — randomized order, scale, history | **Done** (superseded by Phase 8) |
| 4 | Feedback system — F1 through F5 | **Done** (F4/F5 removed in Phase 8) |
| 5 | Incorrect knowledge phase | **Done** (re-spec'd in Phase 8) |
| 6 | Analysis metrics — exploration + dominance | **Done** (extended in Phase 8) |
| 7 | Discussion order experiment + Portkey migration | **Done** (random removed in Phase 8) |
| 8 | Research-lead revisions — overlap redefinition, config trim, incorrect-knowledge redesign, consensus prompt, metric additions | **Done** |
| 9 | Selective port of Weikai Li's multi-agent DSE architecture — per-phase system prompts, per-call raw prompt/output logging, `role=user, name="AgentX"` discussion threading, candidate IDs (`A1-3`), COMMON REASONING block | **Done** |

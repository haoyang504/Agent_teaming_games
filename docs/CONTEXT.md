# Moon Survival Experiment — Context Document

> This document provides shared context for all implementation phases. Read this first before any phase task document.

---

## Research Objective

Three LLM agents collaborate to rank 15 survival items for a 200-mile lunar trek (NASA Moon Survival task). Rankings are scored against the NASA expert answer using **Sum of Absolute Differences (SAD)**. Lower = better, 0 = perfect, 112 = worst.

The experiment tests how **information asymmetry** and **hierarchy** affect group decision-making.

---

## The Four Experimental Settings

| Setting | Team Info | Role Info | What Agents Are Told |
|---------|-----------|-----------|---------------------|
| 1 | No | No | Only their own knowledge |
| 2 | Yes | No | Own knowledge + how many items each teammate knows |
| 3 | No | Yes | Own knowledge + who the leader is |
| 4 | Yes | Yes | Own knowledge + teammate counts + leader |

- **Team info** = telling each agent how many items the other agents know (just the count, not which items). Must be worded neutrally — no instructions like "defer to" or "act based on this."
- **Role info** = telling all agents that Agent C is the designated leader. No explanation of why. The leader does NOT become the final decision-maker — consensus must emerge through natural discussion.

---

## Knowledge Assignment (3 Dimensions)

### Dimension 1: Team Configurations (how many items each agent knows)

| Config | Agent A | Agent B | Agent C | Diversity |
|--------|---------|---------|---------|-----------|
| 1 | 0 items | 2 items | 4 items | High |
| 2 | 1 item | 2 items | 3 items | Medium |
| 3 | 2 items | 2 items | 2 items | Low (equal) |

Agent C always has the most (or equal) knowledge and is the designated leader in Settings 3 & 4. In Config 3 (equal), Agent C is still leader by convention.

### Dimension 2: Source Variants (where knowledge is drawn from)

| Variant | Source Pool |
|---------|------------|
| V1 | Random from all 15 rankings |
| V2 | Random from rankings 1–8 only (most critical) |
| V3 | Random from rankings 8–15 only (least critical) |

### Dimension 3: Overlap Patterns (how knowledge sets relate across agents)

Agent C is always sampled first, then B and A relative to C.

| Pattern | Rule | Description |
|---------|------|-------------|
| O1 | A ⊆ B ⊆ C (nested) | B is a subset of C; A is a subset of B |
| O2 | A ∩ B ∩ C = ∅ (disjoint) | B excludes C; A excludes B and C |
| O3 | B overlaps C; A disjoint from C | B **guaranteed** to share at least 1 item with C; A has no overlap with C |
| O4 | B disjoint from C; A overlaps C | B has no overlap with C; A **guaranteed** to share at least 1 item with C |

> **Implementation note:** O3/O4 overlap is guaranteed, not probabilistic. Without this guarantee, random sampling from the full pool produced zero overlap in ~46% (O3) and ~80% (O4) of seeds, making them statistically indistinguishable from disjoint.

Core sweep = 3 configs × 4 overlaps = 12 conditions. Source variants may be a secondary sweep (3 × 12 = 36 total).

---

## Feedback Settings (5 modes)

| Mode | Type | Description |
|------|------|-------------|
| F1 | Frequency | SAD scores from last iteration, every iteration (baseline) |
| F2 | Frequency | SAD scores from last iteration, every 2 iterations |
| F3 | Frequency | SAD scores from last iteration, every 4 iterations |
| F4 | Full history | SAD scores for all candidates across all prior iterations |
| F5 | Full history | SAD scores for only top 8 candidates across all prior iterations (agents told upfront) |

---

## Incorrect Knowledge Phase (runs after baseline)

A fully incorrect ranking is generated (no item-rank pair matches ground truth). Correct and incorrect knowledge are mixed and distributed:

| Pattern | Who gets incorrect knowledge |
|---------|------------------------------|
| I1 | Only Agent A |
| I2 | Only Agent B |
| I3 | Only Agent C |
| I4 | All agents get 1 incorrect item (Config 1 exception: Agent C gets 2 instead since Agent A has 0) |

Fixed to: source = V1 (all 15), overlap = O2 (disjoint). Overlap between correct and incorrect must be avoided (if item OR rank matches, exclude it). Warning condition: with or without a prompt telling agents some knowledge may be incorrect.

---

## Codebase Structure

```
run_experiment.py              Entry point; CLI args, CONFIGS dictionary (12 core conditions)
  └── experiment_runner.py     Orchestration; 4-phase iteration loop
        ├── agent.py           LLM agent; system prompt, propose, discuss, select
        │     ├── moon_survival_env.py   Items, ground truth, scoring, parsing
        │     └── knowledge_manager.py   Assigns expert items to agents
        ├── moon_survival_env.py         Evaluation + formatting of results
        └── knowledge_manager.py         3-dimensional knowledge assignment
  └── config.py                API key
run_all_combinations.py        Sweeps all 12 (or 36 with source variants) conditions
verify_plumbing.py             Offline sanity checks (no API calls)
test_knowledge.py              Unit tests for knowledge assignment
metrics.py                     Standalone analysis: novelty, recombination, dominance
```

### Iteration Flow (4 Phases)

1. **Proposals** — Each agent proposes k candidate rankings
2. **Discussion** — Round-robin discussion (order should be randomized per round)
3. **Final Selection** — Last agent selects k final candidates from discussion
4. **Evaluation** — Final candidates scored against NASA ground truth (SAD)

### Key Files

- **moon_survival_env.py** — Do NOT modify. Contains items, ground truth, scoring, parsing.
- **knowledge_manager.py** — 3-dimensional knowledge distribution: `generate_knowledge_assignment(counts, source, overlap, seed)`. Supports nested/disjoint/O3/O4 overlap patterns, all/top/bottom source pools. O3 and O4 **guarantee** at least 1 item of overlap (not left to chance). Agent C is always sampled first. Also provides `generate_incorrect_ranking(seed)` (derangement-based, no item-rank pair matches ground truth) and `generate_mixed_knowledge_assignment(counts, incorrect_pattern, seed, incorrect_seed)` for I1–I4 patterns with overlap constraint enforcement.
- **agent.py** — LLM agent class. Constructor accepts `team_knowledge_info` (dict or None), `leader_id` (int or None), and `knowledge_warning` (bool). `_system_prompt()` conditionally injects team info, leader role, and knowledge warning sections. `propose_candidates(k, previous_results, results_context)` accepts a `results_context` string that adapts the feedback framing per mode. `discuss()` and `select_final_candidates()` are the other action methods.
- **experiment_runner.py** — `run_experiment(counts, source, overlap, setting, feedback_mode, discussion_order, ...)`. Uses Portkey AI gateway (`portkey_ai.Portkey`) instead of direct OpenAI calls. Computes `team_knowledge_info` and `leader_id` from the `setting` parameter (Settings 2/4 → team info; Settings 3/4 → leader_id=3). Discussion speaking order controlled by `discussion_order`: `"random"` (shuffled per round via `discussion_seed`), `"ABC"` (fixed 1→2→3), or `"CBA"` (fixed 3→2→1). Discussion history resets each iteration (`history_scope: "last_iteration_only"`). Feedback modes F1–F5 control when and what evaluation data agents receive. All settings, discussion orders, and feedback state are logged in JSON output.
- **run_experiment.py** — CLI entry point. `CONFIGS` dictionary has all 12 core conditions (3 configs x 4 overlaps). CLI args: `--config`, `--counts`, `--source`, `--overlap`, `--setting`, `--feedback-mode`, `--discussion-order`, `--incorrect-pattern`, `--incorrect-warning`, `--incorrect-seed`. Defaults: `--iterations 3` (full protocol: 20), `--discussion-rounds 3`, `--feedback-mode F1`, `--discussion-order random`.

---

## Implementation Phases Overview

| Phase | What | Changes | Status |
|-------|------|---------|--------|
| 1 | Foundation — new knowledge system + setting parameter | 2, 1 | **Done** |
| 2 | Hierarchy — team info + leader role in prompts | 3, 4 | **Done** |
| 3 | Discussion mechanics — randomized order, scale, history | 7, 8, 11 | **Done** |
| 4 | Feedback system — F1 through F5 | 5 | **Done** |
| 5 | Incorrect knowledge phase | 6 | **Done** |
| 6 | Analysis metrics — exploration + dominance | 9, 10 | **Done** |
| 7 | Discussion order experiment + Portkey migration | — | **Done** |

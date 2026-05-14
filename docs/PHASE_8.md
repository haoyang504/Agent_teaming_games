# Phase 8 — Research Lead Revisions

> **Status: NOT STARTED**

> **Prerequisites:** Read `CONTEXT.md` first. Phases 1–7 must be complete.
> **Goal:** Apply the revisions requested by the research lead (ES) after Phase 7. This phase trims, redefines, or redesigns parts of Phases 1, 2, 3, 4, 5, 6, 7 — it does not introduce a new orthogonal dimension. After this phase the experiment is leaner (fewer overlap patterns, fewer configs, fewer feedback modes) but the semantics of the kept dimensions are tighter and the incorrect-knowledge phase is fundamentally different.

---

## Source of changes

These revisions came from ES's email of 2026-05-08 in response to a clarification request after Phase 7. Her answers were:

1. **Consensus.** Keep current mechanism (Agent C selects final). Add a sentence to the prompt: *"The agents aim to reach a consensus through N rounds of discussion."* The "consensus" label in the docs stays.
2. **Overlap O3/O4 are full-subset, not partial.** B ⊆ C (O3) and A ⊆ C (O4). The non-subset agent stays completely disjoint from the other two.
3. **Overlap O5** added: A = B, C disjoint.
4. **Configs:** drop `[1,2,3]` medium; drop `top`/`bottom` source variants. Keep `[0,2,4]` high and `[2,2,2]` low.
5. **Feedback:** keep F1–F3 only.
6. **Incorrect knowledge** redesigned end-to-end:
    - Use one fixed incorrect ranking, persisted to disk, reused across runs.
    - I1/I2/I3 = the named agent's items are **all** replaced with incorrect ones (same count, just wrong content). Agent does NOT receive the full 15-item incorrect list.
    - I4 = each agent has half-correct, half-incorrect, same total count.
7. **Metrics:** discussion/proposal raw responses (which she calls "reasoning traces") must be saved — already done, just confirm. Add per-agent SAD per iteration and per-agent contribution share %.
8. **Discussion order:** drop `random`, default to `ABC`. `CBA` deferred.

---

## What exists now

- **Overlap.** `knowledge_manager.generate_knowledge_assignment(counts, source, overlap, seed)` supports `nested` / `disjoint` / `O3` / `O4`. O3 and O4 currently guarantee **≥1 item** of overlap via `_sample_with_guaranteed_overlap()` — not a full subset.
- **Configs.** `run_experiment.CONFIGS` has 12 named configs (3 counts × 4 overlaps). `run_all_combinations.py` sweeps 12 (or 36 with source variants).
- **Source.** `_get_source_pool()` filters items by `all` / `top` / `bottom`.
- **Feedback.** `experiment_runner.run_experiment()` branches on F1–F5.
- **Discussion order.** CLI accepts `random` / `ABC` / `CBA`; default is `random`.
- **Incorrect knowledge.** `generate_mixed_knowledge_assignment()` produces a derangement-based incorrect ranking *per run* (deterministic given `incorrect_seed=99` but not persisted as an artifact). Incorrect items are **appended** to correct items (count grows). I1/I2/I3 give 1 incorrect item to the named agent; I4 gives 1 to each agent with a `[0,2,4]` special case for Agent C.
- **Consensus.** No prompt sentence. The word "consensus" appears only in docs.
- **Leader role.** Single descriptive sentence in `agent._system_prompt()`: *"Agent C has been designated as the team leader."* No statement of responsibility.
- **Metrics.** `metrics.py` computes novelty / recombination / dominance. Per-iteration team SAD is in `iteration_summary`. Per-agent proposal SAD is **not** evaluated. Dominance gives absolute pair counts per agent but not a percentage share.
- **Reasoning traces.** Every proposal and every discussion turn already logs `raw_response` (`experiment_runner.py:106-111`, `:151-157`). Nothing to add — just confirm to ES.

---

## Task 8A: Trim discussion order

**File:** `code/run_experiment.py`

- Change `--discussion-order` default from `"random"` to `"ABC"`.
- Remove `"random"` from `choices` — leave `["ABC", "CBA"]`. CBA stays callable but `ABC` is the default for all sweeps.
- Update the help text to drop the "random" wording.

**File:** `code/experiment_runner.py`

- In `run_iteration()`, keep the `discussion_order` parameter but remove the `else` branch that calls `disc_rng.shuffle()`. Change to:

```python
if discussion_order == "ABC":
    round_order = sorted(agents, key=lambda a: a.agent_id)
elif discussion_order == "CBA":
    round_order = sorted(agents, key=lambda a: a.agent_id, reverse=True)
else:
    raise ValueError(f"Unknown discussion_order: {discussion_order}")
```

- Remove the `discussion_seed` parameter from `run_iteration()` and the `disc_rng` construction — there's no shuffling left to seed. Drop the `discussion_seed=knowledge_seed + it` argument in the call site in `run_experiment()`.

**File:** `code/run_all_combinations.py`

- Hardcode `discussion_order="ABC"` in the `run_experiment()` call. (It already defaults to `random` upstream; the sweep should be explicit so old defaults don't leak.)

**Files:** `README.md`, `docs/CONTEXT.md`, `docs/PHASE_3.md`, `docs/PHASE_7.md`

- Update the discussion-order tables to show only ABC (primary) and CBA (deferred). Note that `random` was removed in Phase 8.

---

## Task 8B: Add consensus sentence + leader responsibility

**File:** `code/agent.py`

In `_system_prompt()`, add a new section between TEAM ROLE (if present) and SCORING:

```
=== TEAM PROCESS ===
The agents aim to reach a consensus through {num_discussion_rounds} rounds of discussion.
```

This requires plumbing `num_discussion_rounds` through to the agent. Two options:

- **Option A (preferred):** Add `num_discussion_rounds: int = 3` to `MoonSurvivalAgent.__init__()` and store as an attribute. Pass it from `experiment_runner.run_experiment()` when constructing agents.
- **Option B:** Pass it as a method argument every time the system prompt is built. Worse — `_system_prompt()` is called from three different places and the agent doesn't know the round count outside that context.

Use Option A.

**Leader sentence (covers #5).** In the existing `=== TEAM ROLE ===` block, change:

```
Agent {leader_label} has been designated as the team leader.
```

to:

```
Agent {leader_label} has been designated as the team leader. The leader is responsible for guiding the discussion.
```

**Update neutrality test.** `verify_plumbing.py` test 27 currently asserts the absence of phrases like `"defer to"`, `"act based on"`, `"follow the leader"`, `"consider this when"`, `"use this information"`. The new responsibility sentence won't match those exact strings, but it *is* an instructional drift. Update the test's assertion list to allow the responsibility sentence explicitly (e.g. require the *exact* string `"The leader is responsible for guiding the discussion."` is present in Settings 3/4 and absent in Settings 1/2). Add a comment that the prompt deliberately includes one instructional sentence for the leader role.

---

## Task 8C: Redefine O3, O4, add O5

**File:** `code/knowledge_manager.py`

Replace the O3 and O4 branches in `generate_knowledge_assignment()` with full-subset semantics. Add O5.

**New semantics:**

| Overlap | Constraint |
|---|---|
| `nested` | A ⊆ B ⊆ C (unchanged) |
| `disjoint` | A, B, C pairwise disjoint (unchanged) |
| `O3` | **B ⊆ C; A disjoint from both B and C** (was: B partial overlap with C, A disjoint from C only) |
| `O4` | **A ⊆ C; B disjoint from both A and C** (was: A partial overlap with C, B disjoint from C only) |
| `O5` | **A = B (same item set); C disjoint from both** — new |

**Implementation:**

```python
elif overlap == "O3":
    # B ⊆ C; A disjoint from B and C
    items_c = _sample(rng, pool_list, count_c)
    items_b = _sample(rng, items_c, count_b)  # B drawn from C's items
    c_set = set(items_c)
    non_c = [i for i in pool_list if i not in c_set]
    if count_a > len(non_c):
        raise ValueError(...)
    items_a = _sample(rng, non_c, count_a)

elif overlap == "O4":
    # A ⊆ C; B disjoint from A and C
    items_c = _sample(rng, pool_list, count_c)
    items_a = _sample(rng, items_c, count_a)  # A drawn from C's items
    c_set = set(items_c)
    non_c = [i for i in pool_list if i not in c_set]
    if count_b > len(non_c):
        raise ValueError(...)
    items_b = _sample(rng, non_c, count_b)

elif overlap == "O5":
    # A = B (smaller agent's set is a subset of the larger when counts differ);
    # C disjoint from both
    items_c = _sample(rng, pool_list, count_c)
    c_set = set(items_c)
    non_c = [i for i in pool_list if i not in c_set]
    larger = max(count_a, count_b)
    if larger > len(non_c):
        raise ValueError(...)
    larger_set = _sample(rng, non_c, larger)
    if count_a >= count_b:
        items_a = larger_set
        items_b = _sample(rng, items_a, count_b)
    else:
        items_b = larger_set
        items_a = _sample(rng, items_b, count_a)
```

For `[2,2,2]`, the smaller=larger branch makes A = B exactly. For `[0,2,4]`, A=0 ⊆ B trivially.

**Remove dead code.** `_sample_with_guaranteed_overlap()` is no longer used after the redefinition. Delete it.

**Update error messages and the `Valid:` listing** in the unknown-overlap `raise ValueError` to include `O5`.

**Update validation in the `_validate_overlap_feasibility()` paths.** Currently O3/O4 only check `count_a` or `count_b` against `non_c`. After redefinition both agents need checks against the new constraints.

---

## Task 8D: Trim configs and remove source variants

**File:** `code/run_experiment.py`

- Reduce `CONFIGS` to two named entries:
  - `"high_div"`: `{"counts": [0, 2, 4], "overlap": "nested"}` plus variants for each overlap (or keep a single named entry and rely on `--overlap` to vary).
  - `"low_div"`: `{"counts": [2, 2, 2], "overlap": "nested"}`.
  - Suggestion: drop the per-overlap named entries (`high_div_disjoint`, etc.) — too many names, just use `--counts` + `--overlap` directly.
- Remove `--source` from the CLI.
- Remove the `source` key from `CONFIGS`.

**File:** `code/knowledge_manager.py`

- Delete `_get_source_pool()`.
- Remove the `source` parameter from `generate_knowledge_assignment()`.
- Remove the `source="all"` hardcode in `generate_mixed_knowledge_assignment()` (just drop the argument since it no longer exists).

**File:** `code/experiment_runner.py`

- Remove `source` from `run_experiment()` signature, the experiment-log dict, the experiment ID format, the verbose-print line, and all call sites.
- The experiment ID becomes `c{counts}__{overlap}__s{setting}__fb{mode}__do{order}__k{k}__iter{n}__disc{d}__{ts}[__inc{pattern}[_warn]]`.

**File:** `code/run_all_combinations.py`

- Remove `SOURCES_CORE`, `SOURCES_ALL`, `--include-source-variants`.
- Replace the config × overlap × source product with the **unique-conditions list** from Task 8H below — not a naive 2 × 5 product, because some combinations collapse.

**Files:** `README.md`, `docs/CONTEXT.md`, `docs/PHASE_1.md`

- Update the knowledge-assignment table to drop source variants and `[1,2,3]`.
- Update the "core sweep = 3 × 4 = 12" line to "core sweep = 7 unique conditions" with a pointer to the collapse table in Phase 8.

---

## Task 8E: Drop feedback modes F4 and F5

**File:** `code/run_experiment.py`

- Change `--feedback-mode` choices from `["F1","F2","F3","F4","F5"]` to `["F1","F2","F3"]`.

**File:** `code/experiment_runner.py`

- Delete the `F4` and `F5` branches in the per-iteration feedback logic (`run_experiment.py:403-420`).
- Delete the `all_candidates_history` / `all_scores_history` accumulators — only F1–F3 use the previous iteration's results.
- Delete the F4/F5 `results_context` strings.
- Simplify the feedback-frequency dispatch: `{"F1": 1, "F2": 2, "F3": 4}` with no fallback default.

**Files:** `README.md`, `docs/CONTEXT.md`, `docs/PHASE_4.md`

- Strike F4 and F5 rows from the feedback-mode tables.

---

## Task 8F: Redesign incorrect knowledge

### 8F.1 Persist the fixed incorrect ranking

**File:** `code/knowledge_manager.py`

- Run `generate_incorrect_ranking(seed=99)` once and write the result to `code/incorrect_ranking.json` (committed to the repo).
- Update `generate_incorrect_ranking()` to load from that file when it exists, falling back to the derangement generator only if the file is missing. Signature stays the same — `seed` becomes a regeneration knob, not a per-call randomizer.

```python
INCORRECT_RANKING_FILE = os.path.join(os.path.dirname(__file__), "incorrect_ranking.json")

def generate_incorrect_ranking(seed: int = 99) -> List[Tuple[str, int, str]]:
    if os.path.exists(INCORRECT_RANKING_FILE):
        with open(INCORRECT_RANKING_FILE) as f:
            data = json.load(f)
        return [(d["item"], d["rank"], d["explanation"]) for d in data]
    # Fallback: regenerate (used only the first time, or when seed is bumped manually)
    ...
```

- Add a one-off script or `python -c` snippet to the docs showing how to regenerate the file if the seed is bumped.

**Verification artifact ES asked for:** generate the ranking, write `code/incorrect_ranking.json`, and print it side-by-side with the ground truth in a Markdown table to confirm it's visibly different. Add the table to `docs/PHASE_8.md` under "Verification" once generated.

### 8F.2 Replace, don't append

**File:** `code/experiment_runner.py`

The current merge at `:247-250`:

```python
knowledge_assignments = [
    correct + incorrect
    for correct, incorrect in zip(correct_assignments, incorrect_assignments)
]
```

becomes (replacement at same count):

```python
knowledge_assignments = [
    incorrect if incorrect else correct
    for correct, incorrect in zip(correct_assignments, incorrect_assignments)
]
```

Wait — this only works under the "all replaced" reading (I1/I2/I3). For I4 (half replaced) the merge is per-agent, partial.

Cleaner: do the replacement inside `generate_mixed_knowledge_assignment()` so the runner just gets a single merged list per agent.

### 8F.3 New I1–I4 semantics

**File:** `code/knowledge_manager.py`

Rewrite `generate_mixed_knowledge_assignment()` to return a single merged list per agent (not a `(correct, incorrect)` tuple). Behavior per pattern, given config counts `[a, b, c]`:

| Pattern | A's items | B's items | C's items |
|---|---|---|---|
| I1 | `a` incorrect, 0 correct | `b` correct | `c` correct |
| I2 | `a` correct | `b` incorrect, 0 correct | `c` correct |
| I3 | `a` correct | `b` correct | `c` incorrect, 0 correct |
| I4 | `a/2` correct + `a/2` incorrect | `b/2` correct + `b/2` incorrect | `c/2` correct + `c/2` incorrect |

For `[0,2,4]`:

| Pattern | A | B | C |
|---|---|---|---|
| I1 | 0 (vacuous) | 2 correct | 4 correct |
| I2 | 0 | 2 incorrect | 4 correct |
| I3 | 0 | 2 correct | 4 incorrect |
| I4 | 0 | 1 correct + 1 incorrect | 2 correct + 2 incorrect |

For `[2,2,2]`:

| Pattern | A | B | C |
|---|---|---|---|
| I1 | 2 incorrect | 2 correct | 2 correct |
| I2 | 2 correct | 2 incorrect | 2 correct |
| I3 | 2 correct | 2 correct | 2 incorrect |
| I4 | 1 + 1 | 1 + 1 | 1 + 1 |

**Edge case to flag.** I1 on `[0,2,4]` is identical to "no incorrect knowledge applied" (because A has 0 items either way). Either skip that combination in the sweep or run it explicitly as a sanity check. Recommend skip — note in `run_all_combinations.py`.

**Constraint preservation.** The existing rule that no incorrect (item, rank) tuple may have its item OR its rank match any correct (item, rank) tuple **of the same agent** still applies. With the replacement semantics that's mostly vacuous for I1/I2/I3 (the agent has no correct items left to clash with), but for I4 it's load-bearing — keep `_sample_incorrect_for_agent()`.

### 8F.4 Source/overlap pinning

Currently `generate_mixed_knowledge_assignment()` hardcodes `source="all"` and `overlap="disjoint"`. After Task 8D, `source` is gone, so just drop it. Keep `overlap="disjoint"` for the correct-knowledge portion.

---

## Task 8G: Metric additions

### 8G.1 Per-agent proposal SAD per iteration

**File:** `code/experiment_runner.py`

In `run_iteration()`, after each agent's `propose_candidates()` call, evaluate each candidate and store the SAD scores alongside the proposal log entry:

```python
candidate_scores = []
for cand in candidates:
    ranks = ranking_dict_to_list(cand)
    candidate_scores.append(evaluate_ranking(ranks))

proposal_entry = {
    "agent_id": agent.agent_id,
    "raw_response": raw,
    "candidates": [dict(c) for c in candidates],
    "candidate_scores": candidate_scores,
    "best_proposal_sad": min(candidate_scores),
    "mean_proposal_sad": round(sum(candidate_scores) / len(candidate_scores), 2),
}
```

These give per-agent, per-iteration fluctuation data without any extra LLM calls.

### 8G.2 Per-agent contribution share %

**File:** `code/metrics.py`

`compute_dominance()` already builds `pairs_per_agent` per final candidate. Add a `contribution_share` field:

```python
share = {
    agent_id: round(count / total_pairs * 100, 1)
    for agent_id, count in pairs_per_agent.items()
}
result["contribution_share_pct"] = share
```

Aggregate across iterations as well:

```python
summary["overall_contribution_share_pct"] = {
    agent_id: round(total / sum_total * 100, 1)
    for agent_id, total in overall_dominance.items()
}
```

Update `analyze_experiment()` to include the share in the returned dict and the `--verbose` CLI to print it.

### 8G.3 Reasoning traces — no code change

Confirm to ES in the response email/Slack that the raw text of proposals and discussion turns is already saved in every experiment JSON. Point her at the `proposals[*].raw_response` and `discussion[*].response` fields.

---

## Task 8H: The 7 unique conditions

The 2 × 5 product nominally gives 10 conditions, but several collapse because of degenerate cases. The sweep should run only the unique ones.

### `[0,2,4]` — 2 unique

A has 0 items, so it's trivially disjoint from everything and trivially a subset of everything.

| Nominal overlap | Reduces to | Unique? |
|---|---|---|
| nested | B ⊆ C, A empty | **YES** — call this `[0,2,4]/nested` |
| O3 (B ⊆ C, A disj B,C) | B ⊆ C, A empty (disjointness from C is vacuous when A is empty) | duplicate of nested |
| disjoint | B and C disjoint, A empty | **YES** — call this `[0,2,4]/disjoint` |
| O4 (A ⊆ C, B disj A,C) | A empty ⊆ C trivially, B disjoint from C | duplicate of disjoint |
| O5 (A = B, C disj) | A empty so A=B requires B=∅, but B has 2 items — collapses to B and C disjoint (per ES's note) | duplicate of disjoint |

### `[2,2,2]` — 5 unique

All counts equal, so subsets become equalities. Each overlap yields a structurally distinct condition because of who holds the unique-vs-shared knowledge.

| Overlap | Effective structure | Notes |
|---|---|---|
| nested | A = B = C | full information symmetry — degenerate but conceptually distinct (control) |
| disjoint | 3 pairwise disjoint pairs, 6 unique items | maximum diversity |
| O3 | B = C, A disjoint | A holds the unique knowledge |
| O4 | A = C, B disjoint | B holds the unique knowledge |
| O5 | A = B, C disjoint | C (the leader) holds the unique knowledge |

### Final sweep matrix

**7 unique knowledge conditions × 4 settings = 28 baseline runs.**

| # | Counts | Overlap | Notes |
|---|---|---|---|
| 1 | [0,2,4] | nested | (also covers O3) |
| 2 | [0,2,4] | disjoint | (also covers O4, O5) |
| 3 | [2,2,2] | nested | degenerate — all-same |
| 4 | [2,2,2] | disjoint | max diversity |
| 5 | [2,2,2] | O3 | A-unique |
| 6 | [2,2,2] | O4 | B-unique |
| 7 | [2,2,2] | O5 | C-unique |

**File:** `code/run_all_combinations.py`

- Replace the product-of-lists construction with this explicit table.
- Add a comment naming each row with the collapse rationale so future-readers don't second-guess why it isn't 10.

---

## Loose ends

### CBA + final-selector ambiguity

The final-selection wiring (`experiment_runner.py:167`) always uses `agents[-1]`, which is Agent C regardless of discussion order. ES's reasoning for keeping the current consensus mechanism (#2) was *"if Agent C speaks last (A → B → C order), then it makes sense C ends up deciding."* Under `--discussion-order CBA`, C speaks **first** and A speaks **last**, but C still selects.

This isn't a problem for the immediate Phase 8 scope (default is ABC, CBA is deferred), but before any CBA runs are launched, ES needs to answer: should the final selector follow the speaking order (last speaker decides → A under CBA), or stay fixed (C always decides regardless of order)? Flag this in the email reply.

### Setting 1 isn't a pure no-hierarchy baseline

In Setting 1 the prompt has no leader sentence, but Agent C still makes the final selection. ES accepted this ("I think it is okay to keep it as is"). Document explicitly in `docs/CONTEXT.md` so it's not forgotten: Setting 1 is "no *prompted* leader" rather than "no leader at all."

---

## Files changed in Phase 8

| File | Change |
|---|---|
| `code/knowledge_manager.py` | Redefine O3/O4, add O5, remove `_get_source_pool` and source param, remove `_sample_with_guaranteed_overlap`, persist incorrect ranking to JSON, rewrite `generate_mixed_knowledge_assignment` for replace-semantics I1–I4 |
| `code/agent.py` | Add `num_discussion_rounds` constructor param, add TEAM PROCESS section, add leader responsibility sentence |
| `code/experiment_runner.py` | Remove `source` param, remove `discussion_seed` + random shuffling, evaluate per-agent proposals, simplify feedback dispatch (F1–F3 only), update experiment ID format, pass `num_discussion_rounds` to agents |
| `code/run_experiment.py` | Drop `random` from `--discussion-order` choices, change default to ABC, drop `--source`, drop F4/F5 from `--feedback-mode` choices, simplify CONFIGS |
| `code/run_all_combinations.py` | Replace product sweep with explicit 7-row unique-conditions table, hardcode `discussion_order="ABC"` |
| `code/verify_plumbing.py` | Update test 27 neutrality check to allow the leader responsibility sentence |
| `code/metrics.py` | Add `contribution_share_pct` to `compute_dominance` per-iteration and to the aggregate summary |
| `code/incorrect_ranking.json` | **NEW** — persisted incorrect ranking artifact, regenerated only when `seed` is bumped |
| `README.md` | Trim CLI tables, drop source/F4/F5/random docs, update sweep size from 12 to 7 |
| `docs/CONTEXT.md` | Update overlap table (new O3/O4/O5 semantics), drop source variants and `[1,2,3]`, drop F4/F5, document Setting 1 leader-leak |
| `docs/PHASE_1.md` | Note Phase 8 superseded the partial-overlap O3/O4 |
| `docs/PHASE_3.md` | Note Phase 8 removed random discussion order |
| `docs/PHASE_4.md` | Note Phase 8 removed F4 and F5 |
| `docs/PHASE_5.md` | Note Phase 8 changed incorrect-knowledge semantics (replace not append, new I1–I4 distribution, fixed ranking) |
| `docs/PHASE_7.md` | Note Phase 8 dropped random and changed default to ABC |

---

## What NOT to do in Phase 8

- Do NOT change `moon_survival_env.py`.
- Do NOT add new feedback modes, settings, or overlap patterns beyond O5.
- Do NOT add a real voting/aggregation step — ES confirmed C-decides is fine.
- Do NOT regenerate the incorrect ranking on every run. Persist it once.
- Do NOT enable CBA experiments before resolving the final-selector ambiguity with ES.
- Do NOT delete the Phase 7 random-order JSON logs in `results/` — they remain valid historical data even though the parameter is removed going forward.

# Changelog — Moon Survival Agent Teaming Codebase

All changes relative to the initial commit (`99700ce`). Organized by file.

> Phase 8 (research-lead revisions) is appended at the end of this document under "Phase 8 — Research Lead Revisions." It supersedes earlier entries where noted (overlap O3/O4 semantics, source variants, feedback F4/F5, incorrect-knowledge patterns, discussion-order random).

---

## `code/knowledge_manager.py` — Modified

**Summary:** Replaced the string-based knowledge level system with a 3-dimensional knowledge assignment system (counts/source/overlap). Added incorrect knowledge generation for Phase 5.

### Changes

1. **Removed `LEVEL_COUNTS` and `VALID_LEVELS` module-level constants.** Previously defined `{"full": 15, "half": 8, "quarter": 4, "none": 0}` and a frozenset of valid level strings. No longer needed — counts are now passed as integers directly.

2. **Replaced `generate_knowledge_assignment(levels, seed)` with `generate_knowledge_assignment(counts, source, overlap, seed)`.** The old function accepted a list of string levels (e.g. `["none", "quarter", "half"]`) and sampled items from a single shuffled list where smaller levels were subsets of larger ones. The new function accepts:
   - `counts: List[int]` — exact item counts per agent `[A, B, C]`
   - `source: str` — `"all"` (ranks 1–15), `"top"` (ranks 1–8), `"bottom"` (ranks 8–15)
   - `overlap: str` — `"nested"` (A⊆B⊆C), `"disjoint"` (no overlap), `"O3"` (B overlaps C, A disjoint from C), `"O4"` (B disjoint from C, A overlaps C)
   - Agent C is always sampled first, then B, then A
   - O3/O4 guarantee at least 1 item of overlap via `_sample_with_guaranteed_overlap()`
   - Raises `ValueError` for impossible configurations (e.g. disjoint with counts exceeding pool)

3. **Simplified `format_knowledge_for_prompt()` docstring.** Removed verbose Args/Returns sections. Behavior unchanged.

4. **Replaced `knowledge_level_label(level: str)` with `knowledge_level_label(count: int)`.** Previously returned descriptive strings like `"Half knowledge (8/15 items)"`. Now returns `f"{count}/15 items"` for any integer count.

5. **Added `format_knowledge_counts(counts: List[int]) -> str`.** New helper returning e.g. `"Agent A: 0, Agent B: 2, Agent C: 4"` for logging.

6. **Added `generate_incorrect_ranking(seed) -> List[Tuple]`.** Generates a derangement of ranks 1–15 where no item-rank pair matches ground truth. Uses rejection sampling via `_derangement()`. Returns tuples with a generic placeholder explanation.

7. **Added `generate_mixed_knowledge_assignment(counts, incorrect_pattern, seed, incorrect_seed)`.** Generates correct knowledge (fixed to source=`"all"`, overlap=`"disjoint"`) then distributes incorrect items per pattern:
   - I1: only Agent A gets 1 incorrect
   - I2: only Agent B gets 1 incorrect
   - I3: only Agent C gets 1 incorrect
   - I4: all agents get 1 incorrect (exception: if Agent A has 0 items, Agent C gets 2 instead)
   - Returns `(correct_assignments, incorrect_assignments)` as separate lists

8. **Added `_derangement(rng, n) -> List[int]`.** Internal helper generating a random permutation of `[1..n]` with no fixed points.

9. **Added `_sample_incorrect_for_agent(rng, incorrect_ranking, correct_knowledge, count)`.** Filters incorrect tuples to exclude any where the item name OR rank matches the agent's correct knowledge, then samples `count` items.

10. **Added `_get_source_pool(source) -> List[int]`.** Returns item indices filtered by source pool.

11. **Added `_sample(rng, population, k) -> List[int]`.** Simple wrapper around `rng.sample()` that returns `[]` for k=0 and validates pool size.

12. **Added `_sample_with_guaranteed_overlap(rng, pool, overlap_source, k) -> List[int]`.** Samples k items from pool, guaranteeing at least 1 is from `overlap_source`. Picks 1 overlap item first, fills rest from the remaining pool.

---

## `code/agent.py` — Modified

**Summary:** Added team information, leader role, and knowledge warning sections to the system prompt. Added retry logic to both `_chat()` and `propose_candidates()`. Added `results_context` parameter to `propose_candidates()`.

### Changes

1. **Added retry-without-temperature logic to `_chat()`.** Previously called `client.chat.completions.create()` directly and let exceptions propagate. Now wraps the call in try/except: on any error, retries without the `temperature` parameter (some providers/models don't support it). If the retry also fails, raises the original error.

2. **Added `_ID_TO_LABEL` module-level constant.** Maps `{1: "A", 2: "B", 3: "C"}` for agent ID to letter label conversion.

3. **Added 3 new parameters to `MoonSurvivalAgent.__init__()`:**
   - `team_knowledge_info: Optional[Dict[str, int]]` — maps agent labels to knowledge counts (e.g. `{"A": 0, "B": 2, "C": 4}`)
   - `leader_id: Optional[int]` — agent_id of the designated leader
   - `knowledge_warning: bool` — whether to add an incorrect-knowledge warning
   All stored as instance attributes.

4. **Added knowledge warning to `_system_prompt()`.** When `self.knowledge_warning` is True, appends `"\n\nNote: Some of the knowledge you have received may be incorrect."` to the knowledge block. Purely factual, no behavioral instructions.

5. **Added `=== TEAM INFORMATION ===` section to `_system_prompt()`.** When `self.team_knowledge_info` is not None, inserts a section listing each agent's knowledge count. Uses `"has no specialised knowledge"` for count=0. Inserted between the knowledge block and scoring block.

6. **Added `=== TEAM ROLE ===` section to `_system_prompt()`.** When `self.leader_id` is not None, inserts `"Agent {label} has been designated as the team leader."` Uses `_ID_TO_LABEL` for the mapping. No behavioral instructions.

7. **Added `results_context` parameter to `propose_candidates()`.** Default `"the previous iteration"`. The context string now reads `f"Here are the results from {results_context}:"` instead of the hardcoded `"the previous iteration"`. Enables F4/F5 feedback modes to say `"all prior iterations"`.

8. **Added 3-attempt retry loop to `propose_candidates()`.** Previously called `_chat()` once and let `_parse_k_candidates()` raise on failure. Now retries up to 3 times: on `ValueError`, feeds the error message back to the model as a follow-up user message requesting correction. Matches the existing pattern in `select_final_candidates()`.

---

## `code/experiment_runner.py` — Modified

**Summary:** Replaced the string-based strategy system with the new counts/source/overlap/setting parameters. Added feedback modes F1–F5, discussion order randomization, incorrect knowledge support, and expanded logging.

### Changes

1. **Added `import random`** for discussion order shuffling.

2. **Added `generate_mixed_knowledge_assignment` and `format_knowledge_counts` to imports** from `knowledge_manager`.

3. **Added `discussion_seed` parameter to `run_iteration()`.** Creates `disc_rng = random.Random(discussion_seed)` for reproducible speaking order shuffling.

4. **Added `results_context` parameter to `run_iteration()`.** Passed through to `agent.propose_candidates()` to adapt feedback framing per mode.

5. **Added `feedback_provided` field to iteration log.** Boolean indicating whether `previous_candidates` was provided (True) or None (False).

6. **Added `discussion_orders` field to iteration log.** Dict mapping round number to list of agent_ids in speaking order.

7. **Randomized discussion speaking order.** Previously iterated `for agent in agents` (fixed 1→2→3 order). Now shuffles `round_order = list(agents); disc_rng.shuffle(round_order)` each round. Proposal order (1→2→3) and final selection (Agent 3) remain fixed.

8. **Added `speaking_order` to each discussion log entry.** Records which order agents spoke in that round.

9. **Passed `results_context` to `agent.propose_candidates()`.** Previously called with just `(k, prev_summary)`.

10. **Replaced `knowledge_strategy: List[str]` parameter in `run_experiment()` with 8 new parameters:**
    - `counts: List[int]` — replaces the old string-based levels
    - `source: str` — `"all"`, `"top"`, `"bottom"`
    - `overlap: str` — `"nested"`, `"disjoint"`, `"O3"`, `"O4"`
    - `setting: int` — experimental setting 1–4
    - `feedback_mode: str` — `"F1"` through `"F5"`
    - `incorrect_pattern: Optional[str]` — `"I1"`–`"I4"` or None
    - `incorrect_warning: bool` — whether to warn agents about incorrect knowledge
    - `incorrect_seed: int` — seed for incorrect ranking generation

11. **Added setting-dependent info computation.** Computes `team_knowledge_info` dict for Settings 2/4 and `leader_id=3` for Settings 3/4. Both passed to agent constructors.

12. **Added incorrect knowledge branching.** When `incorrect_pattern` is set, calls `generate_mixed_knowledge_assignment()` instead of `generate_knowledge_assignment()`, then merges correct + incorrect lists per agent.

13. **Passes `knowledge_warning` to agent constructors.** Set to the value of `incorrect_warning`.

14. **Updated experiment ID format.** Changed from `{strategy_label}__k{k}__iter{n}__disc{d}__{timestamp}` to `c{counts}__source__overlap__s{setting}__fb{mode}__k{k}__iter{n}__disc{d}__{timestamp}`. Appends `__inc{pattern}` and `_warn` when incorrect knowledge is active.

15. **Updated agent knowledge log.** Changed from `level`/`label` string fields to `agent_label` (A/B/C), `count` (int), `label` (e.g. "4/15 items"). Adds `incorrect_items` list when incorrect assignments exist.

16. **Updated verbose output.** Prints knowledge counts, source, overlap, setting, feedback mode, incorrect pattern. Prints team info and leader designation for Settings 2+.

17. **Updated experiment log dict.** Added fields: `counts`, `source`, `overlap`, `setting`, `feedback_mode`, `incorrect_pattern`, `incorrect_warning`, `incorrect_seed`, `team_knowledge_info`, `leader_id`, `history_scope`. Removed `knowledge_strategy`.

18. **Implemented feedback modes F1–F5 in the iterative loop.** Previously passed `prev_candidates`/`prev_scores` unconditionally. Now:
    - F1: feedback every iteration (unchanged behavior)
    - F2: feedback every 2 iterations (`it % 2 == 0`)
    - F3: feedback every 4 iterations (`it % 4 == 0`)
    - F4: accumulates all candidates/scores across iterations, passes full history
    - F5: same as F4 but filters to top 8 candidates by SAD score
    - Iteration 1 never receives feedback regardless of mode

19. **Added `results_context` string per feedback mode.** `"the previous iteration"` for F1–F3, `"all prior iterations"` for F4, `"all prior iterations (showing top 8 candidates only)"` for F5.

20. **Passes `discussion_seed=knowledge_seed + it` per iteration.** Ensures reproducible but varying discussion order across iterations.

---

## `code/run_experiment.py` — Modified

**Summary:** Replaced the string-based strategy CLI with the new multi-dimensional configuration system. Added CLI arguments for all experimental dimensions.

### Changes

1. **Replaced `STRATEGIES` dict with `CONFIGS` dict.** Previously had 4 entries mapping strategy names to lists of level strings (e.g. `"ascending": ["none", "quarter", "half"]`). Now has 12 entries covering all 3 configs x 4 overlaps (e.g. `"high_div": {"counts": [0, 2, 4], "source": "all", "overlap": "nested"}`).

2. **Added Portkey debug test at top of `main()`.** Temporary test that creates an OpenAI client pointing at Portkey's base URL, sends a simple message, and prints the response before any experiment logic runs. Used to isolate Portkey connectivity issues.

3. **Replaced `--strategy` CLI arg with `--config`.** Accepts a named config from `CONFIGS` instead of a `+`-separated level string.

4. **Added `--counts` CLI arg.** Accepts comma-separated integers (e.g. `"0,2,4"`) for fully custom runs.

5. **Added `--source` CLI arg.** Choices: `all`, `top`, `bottom`. Overrides config's source when provided.

6. **Added `--overlap` CLI arg.** Choices: `nested`, `disjoint`, `O3`, `O4`. Overrides config's overlap when provided.

7. **Added `--setting` CLI arg.** Integer 1–4, default 1. Controls team info and leader role injection.

8. **Added `--feedback-mode` CLI arg.** Choices: `F1`–`F5`, default `F1`. Controls feedback frequency and content.

9. **Added `--incorrect-pattern` CLI arg.** Choices: `I1`–`I4` or None. Controls incorrect knowledge distribution.

10. **Added `--incorrect-warning` CLI arg.** Boolean flag. Adds knowledge warning to agent prompts.

11. **Added `--incorrect-seed` CLI arg.** Default 99. Seed for incorrect ranking generation.

12. **Changed `--iterations` default.** Stays at 3 but help text now mentions "Full protocol: 20."

13. **Changed `--discussion-rounds` default from 1 to 3.** Help text updated to mention randomized order.

14. **Changed `--model` default from `"gpt-4o-mini"` to `"gpt-5-mini"`.** Reflects the team's current model.

15. **Updated config resolution logic.** Supports 3 modes: `--counts` for custom, `--config` for named presets, or defaults to `high_div`. Source and overlap can be overridden independently.

16. **Updated `run_experiment()` call.** Passes all new parameters: `counts`, `source`, `overlap`, `setting`, `feedback_mode`, `incorrect_pattern`, `incorrect_warning`, `incorrect_seed`.

---

## `code/run_all_combinations.py` — Modified

**Summary:** Replaced the 27-combination string-level sweep with a 12-condition (or 36 with source variants) configuration sweep using the new system.

### Changes

1. **Replaced `LEVELS` list and `itertools.product` approach.** Previously generated all 27 combinations of `["none", "quarter", "half"]` across 3 agents. Now uses explicit `CONFIGS` list of 3 count arrays, `OVERLAPS` list of 4 patterns, and `SOURCES_CORE`/`SOURCES_ALL` lists.

2. **Removed `--skip-all-none` CLI arg.** No longer applicable.

3. **Added `--setting` CLI arg.** Integer 1–4, default 1.

4. **Added `--include-source-variants` CLI arg.** When set, also sweeps `source=top` and `source=bottom` (36 total instead of 12).

5. **Updated `run_experiment()` call.** Changed from `knowledge_strategy=strategy` to `counts=counts, source=source, overlap=overlap, setting=args.setting`.

6. **Updated results tracking.** Changed key from `"strategy"` to `"config"` in summary dicts. Updated table formatting and CSV column names.

7. **Updated CSV filename.** Includes setting number: `sweep_s{setting}_k{k}_...`.

8. **Updated model default from `"gpt-4o-mini"` to `"gpt-5-mini"`.**

---

## `code/verify_plumbing.py` — Modified

**Summary:** Rewrote knowledge manager checks for the new API and added system prompt verification for all 4 experimental settings.

### Changes

1. **Added imports** for `knowledge_level_label`, `format_knowledge_counts`, and `MoonSurvivalAgent`.

2. **Replaced tests 10–16 (old knowledge level tests) with new tests 10–16.** Previously tested `generate_knowledge_assignment(["half", "half", "half"])` etc. Now tests:
   - Config 1 `[0, 2, 4]` correct counts
   - Config 2 `[1, 2, 3]` correct counts
   - Config 3 `[2, 2, 2]` correct counts
   - Nested overlap: A ⊆ B ⊆ C
   - Disjoint overlap: no pairwise overlap
   - Same seed reproducibility
   - Different seed produces different results

3. **Added tests 17–18: source pool filtering.** Verifies `source="top"` produces only ranks ≤ 8 and `source="bottom"` produces only ranks ≥ 8.

4. **Replaced tests 17–18 (old format/label tests) with updated tests 19–22.** Tests `format_knowledge_for_prompt` (empty and non-empty), `knowledge_level_label(4)` returns `"4/15 items"`, `format_knowledge_counts([0, 2, 4])` returns expected string.

5. **Added tests 23–27: Setting prompt verification.** Creates `MoonSurvivalAgent` instances with `client=None` for each of the 4 settings and inspects `_system_prompt()`:
   - Setting 1: no `TEAM INFORMATION`, no `TEAM ROLE`
   - Setting 2: `TEAM INFORMATION` present with correct counts, no `TEAM ROLE`
   - Setting 3: `TEAM ROLE` present with "Agent C" as leader, no `TEAM INFORMATION`
   - Setting 4: both sections present
   - Neutrality check: extracts only the team/role sections and verifies absence of instructional phrases (`"defer to"`, `"act based on"`, `"follow the leader"`, `"consider this when"`, `"use this information"`)

---

## `code/metrics.py` — New File

**Summary:** Standalone analysis module that reads experiment JSON logs and computes exploration (novelty, recombination) and agent dominance metrics. No dependencies on `agent.py` or `experiment_runner.py` — only standard library.

### Functions

1. **`extract_pairs(candidate: Dict[str, int]) -> Set[Tuple[str, int]]`** — Extracts the set of (item, rank) pairs from a candidate ranking dict. These are the atomic units for all metrics.

2. **`compute_novelty(experiment_log) -> List[Dict]`** — For each iteration's final candidate(s), counts how many item-rank pairs have never appeared in any prior iteration's final output. Maintains a running `seen_pairs` set. Iteration 1 always has novelty_ratio = 1.0. Returns dicts with `iteration`, `candidate_index`, `total_pairs`, `novel_pairs`, `novelty_ratio`.

3. **`compute_recombination(experiment_log) -> List[Dict]`** — For each final candidate, attributes each item-rank pair to the agent who first proposed it (lowest agent_id wins ties). Unattributed pairs go to the selecting agent. Returns `source_agents`, `pairs_per_agent`, `max_agent_contribution`, `recombination_score` (1 - max_contribution/total), `is_recombination` (2+ source agents).

4. **`compute_dominance(experiment_log) -> List[Dict]`** — Builds a pair-origin mapping from proposals (first proposer wins, iterated in agent_id order). For each final candidate, counts pairs per agent and identifies most/least dominant. All 3 agents always appear in the dominance dict (even with 0 contributions). Also counts unattributed pairs (invented by the final selector).

5. **`analyze_experiment(experiment_log) -> Dict`** — Runs all three metrics and produces a combined result with per-iteration data plus a summary: `mean_novelty_ratio`, `mean_recombination_score`, `overall_dominance` (total pairs per agent across all iterations), `most_dominant_agent`.

6. **`analyze_batch(results_dir) -> List[Dict]`** — Processes all `.json` files in a directory (skipping `_summary.json` and `_analysis.json` files). Returns list of analysis results.

7. **CLI (`__main__` block)** — `python metrics.py <log_file> [--output path] [--verbose]`. Loads a JSON log, runs `analyze_experiment()`, optionally prints detailed results and/or saves analysis JSON.

---

## `code/test_knowledge.py` — New File

**Summary:** Unit test script for the knowledge assignment system. Tests all 12 core configurations and source pool filtering.

### Contents

1. **Core configuration tests.** Iterates all 12 combinations of 3 count configs × 4 overlap patterns with `source="all"`. For each:
   - Asserts correct item counts per agent
   - Asserts overlap constraints: nested (A⊆B⊆C), disjoint (no pairwise overlap), O3 (A∩C = ∅), O4 (B∩C = ∅)

2. **Source filtering tests.** Tests `source="top"` (all ranks ≤ 8) and `source="bottom"` (all ranks ≥ 8) with disjoint overlap.

---

## `code/run_overnight.sh` — New File

**Summary:** Bash script for running the full experimental sweep overnight. Executes 12 conditions (3 configs × 4 settings) at production scale, then generates summary tables and metrics.

### Contents

1. **Experiment loop.** Runs `run_experiment.py` for each of the 3 knowledge configs (`[0,2,4]`, `[1,2,3]`, `[2,2,2]`) × 4 settings, with `--overlap disjoint --k 3 --iterations 20 --discussion-rounds 3 --feedback-mode F1`. Logs output to `results/overnight_log.txt` via `tee`.

2. **Results summary table.** Inline Python script that reads all matching JSON logs and prints a table of config, setting, best SAD, final-iteration best SAD, and per-iteration trend.

3. **Metrics summary.** Calls `metrics.analyze_batch()` on the results directory and prints novelty, recombination, and dominance summaries for all overnight runs. Saves to `results/overnight_metrics.txt`.

---

# Phase 8 — Research Lead Revisions

Phase 8 implements ES's email of 2026-05-08 in response to a clarification request after Phase 7. The changes trim, redefine, or replace pieces of Phases 1–7 rather than adding a new orthogonal dimension. After Phase 8 the experiment is leaner (fewer overlap patterns are *redundant*, fewer feedback modes, fewer configs) but the kept dimensions are tighter and the incorrect-knowledge phase is fundamentally different. See `docs/PHASE_8.md` for the spec.

## `code/knowledge_manager.py` — Modified

**Summary:** Redefined O3 and O4 from partial-overlap to full-subset; added O5 (A=B, C disjoint); removed source-pool plumbing; rewrote incorrect-knowledge assignment for replacement-at-same-count semantics; switched incorrect ranking from per-run derangement to a persisted JSON artifact.

### Changes

1. **Module docstring updated.** "3-dimensional" → "2-dimensional"; explicit note that source pool variants are gone.

2. **`generate_knowledge_assignment(counts, overlap, seed)` — removed `source` parameter.** The pool is always all 15 items. Callers updated everywhere.

3. **`O3` redefined.** Was: B has ≥1 item guaranteed in common with C; A disjoint from C only. Now: **B ⊆ C; A disjoint from both B and C.** A is sampled from `non_c`, which is disjoint from C; because B ⊆ C, A is also disjoint from B. Raises `ValueError` if `count_b > count_c`.

4. **`O4` redefined.** Was: A has ≥1 item guaranteed in common with C; B disjoint from C only. Now: **A ⊆ C; B disjoint from both A and C.** Symmetric to O3. Raises `ValueError` if `count_a > count_c`.

5. **`O5` added.** A = B; C disjoint from both. When `count_a == count_b` the two sets are literally equal; when they differ, the smaller set is a subset of the larger (the larger set is sampled from `non_c`, then the smaller is sampled from the larger). Raises `ValueError` if `max(count_a, count_b) > len(non_c)`.

6. **`_sample_with_guaranteed_overlap()` deleted.** Was used only by the old partial-overlap O3/O4 paths; not needed under full-subset semantics.

7. **`_get_source_pool()` deleted.** Source variants are removed. The pool list is now constructed inline as `list(range(len(ITEMS)))`.

8. **`INCORRECT_RANKING_FILE` module constant added.** Absolute path to `code/incorrect_ranking.json`. Used by `generate_incorrect_ranking()`.

9. **`generate_incorrect_ranking(seed)` rewritten to prefer the persisted artifact.** Loads from `incorrect_ranking.json` when present and returns the same tuples every time. Falls back to derangement-based generation seeded by `seed` only if the file is missing. The `seed` argument is now a fallback knob, not a per-call randomizer.

10. **`generate_mixed_knowledge_assignment(counts, incorrect_pattern, seed, incorrect_seed)` rewritten.** Replaces the old append-one-incorrect-item semantics with replacement at the same per-agent count:
    - **I1**: `incorrect_counts = [counts[0], 0, 0]` — Agent A's items are all incorrect.
    - **I2**: `incorrect_counts = [0, counts[1], 0]` — Agent B's items are all incorrect.
    - **I3**: `incorrect_counts = [0, 0, counts[2]]` — Agent C's items are all incorrect.
    - **I4**: `incorrect_counts = [c - (c // 2) for c in counts]` — each agent has `floor(c/2)` correct + `ceil(c/2)` incorrect.
    For each agent, a random `n_incorrect` of the originally-correct items are dropped and replaced via `_sample_incorrect_for_agent()` (which still enforces the no-item/no-rank-clash constraint). Returns `(final_correct, final_incorrect)` lists per agent; the runner merges them.

11. **`generate_mixed_knowledge_assignment` no longer hardcodes `source`.** It now calls `generate_knowledge_assignment(counts, overlap="disjoint", seed=seed)` without the source argument.

## `code/agent.py` — Modified

**Summary:** Added `num_discussion_rounds` constructor parameter and an unconditional `=== TEAM PROCESS ===` section to the system prompt announcing the consensus goal. Added a "responsible for guiding the discussion" sentence to the leader role.

### Changes

1. **Constructor signature.** Added `num_discussion_rounds: int = 3` parameter (between `num_agents` and `team_knowledge_info`). Stored as `self.num_discussion_rounds`.

2. **TEAM ROLE block expanded.** The single sentence `"Agent {label} has been designated as the team leader."` now ends with `" The leader is responsible for guiding the discussion."` This is a deliberate, single-sentence drift from the pre-Phase-8 "purely factual" leader description. Only renders in Settings 3/4.

3. **TEAM PROCESS block added.** Renders in every setting (not gated on `leader_id` or `team_knowledge_info`). Body: `"The agents aim to reach a consensus through {self.num_discussion_rounds} rounds of discussion."` Inserted between TEAM ROLE (if present) and SCORING.

## `code/experiment_runner.py` — Modified

**Summary:** Removed the `random` discussion-order branch and its RNG plumbing. Removed F4/F5 feedback branches and the cross-iteration history accumulator. Removed the `source` parameter. Now scores every agent's proposals individually so per-agent SAD fluctuation is captured. Threads `num_discussion_rounds` through to agent constructors.

### Changes

1. **`import random` removed.** No discussion-order shuffling left.

2. **`run_iteration()` signature.** Removed `discussion_seed`. `discussion_order` default changed from `"random"` to `"ABC"`. `disc_rng = random.Random(...)` line deleted.

3. **Discussion-order dispatch tightened.** The `else` branch (which used to shuffle via `disc_rng`) now raises `ValueError` for any value other than `"ABC"` or `"CBA"`.

4. **`run_experiment()` signature.** Removed `source`. Default `discussion_order` is now `"ABC"`. The call to `run_iteration` no longer passes `discussion_seed`.

5. **Per-agent proposal scoring (Phase 8 8G).** Inside the proposal loop, every candidate dict produced by an agent is converted via `ranking_dict_to_list()` and evaluated with `evaluate_ranking()`. Each proposal log entry now carries:
    - `candidate_scores: List[int]` — SAD per candidate
    - `best_proposal_sad: int`
    - `mean_proposal_sad: float` (2 decimals)
   Verbose output adds a per-proposal `proposal SADs: [...]` line.

6. **Knowledge assignment plumbing.** `generate_knowledge_assignment(counts, overlap=overlap, seed=knowledge_seed)` — no `source` argument. The merge for incorrect-knowledge runs is unchanged at the call site (`correct + incorrect`); under Phase 8 semantics, `correct` is the *kept* correct items (shrunk inside `generate_mixed_knowledge_assignment`) and `incorrect` is the replacements, so `correct + incorrect` totals exactly `counts[i]` per agent.

7. **`num_discussion_rounds` passed to agent constructors.** Threaded into `MoonSurvivalAgent(...)` so the TEAM PROCESS sentence in the prompt reflects the actual round count.

8. **Experiment ID format.** Removed the `__{source}__` segment. Now: `c{counts}__{overlap}__s{setting}__fb{mode}__do{order}__k{k}__iter{n}__disc{d}__{ts}[__inc{pattern}[_warn]]`.

9. **Experiment log dict.** Removed the `source` field. All other fields unchanged.

10. **Verbose print.** Dropped the `Source: ...` line from the experiment header.

11. **Feedback dispatch simplified.** F4 and F5 branches deleted along with the `all_candidates_history` / `all_scores_history` accumulators. The loop reduces to:
    - F1/F2/F3: pass `prev_candidates` only when `it > 1 and it % frequency == 0`.
    - Frequency dispatch dict is now `{"F1": 1, "F2": 2, "F3": 4}` with no default fallback (an unknown mode would `KeyError` immediately).
    `results_context` is now always `"the previous iteration"`.

## `code/run_experiment.py` — Modified

**Summary:** CLI surface narrowed to the Phase 8 dimensions. `CONFIGS` reduced from 12 to 2 named entries. Source variants and F4/F5 removed. Default discussion order flipped to ABC.

### Changes

1. **`CONFIGS` reduced.** From 12 named entries (3 counts × 4 overlaps) to 2: `high_div` (`[0,2,4]`) and `low_div` (`[2,2,2]`). Overlap is no longer baked into the named config — pass `--overlap` explicitly.

2. **`--source` argument removed.**

3. **`--overlap` default changed.** From `None` (look up from config) to `"nested"`. Choices extended to `["nested", "disjoint", "O3", "O4", "O5"]`.

4. **`--feedback-mode` choices.** Trimmed to `["F1", "F2", "F3"]`. Help text updated to drop F4/F5.

5. **`--discussion-order` default and choices.** Default `"random"` → `"ABC"`. Choices `["random","ABC","CBA"]` → `["ABC","CBA"]`. Help text rewritten.

6. **`--discussion-rounds` help text.** "randomized order" → "configured order."

7. **Config resolution.** Simplified: `--counts` overrides everything; `--config` looks up counts only; default falls back to `high_div`. `--overlap` is now always the CLI value (no per-config override).

8. **`run_experiment()` call.** Removed `source=...` argument.

## `code/run_all_combinations.py` — Modified

**Summary:** Replaced the 12-condition product sweep with the explicit Phase 8 unique-conditions table (7 entries). Removed source-variant CLI. Hardcoded `discussion_order="ABC"`.

### Changes

1. **Module docstring rewritten.** New count is 7 unique conditions; collapse rationale documented row-by-row in `UNIQUE_CONDITIONS`.

2. **`CONFIGS`, `OVERLAPS`, `SOURCES_CORE`, `SOURCES_ALL` lists deleted.** Replaced by `UNIQUE_CONDITIONS: List[Tuple[List[int], str, str]]` — `(counts, overlap, note)` — explicitly enumerating the 7 unique knowledge conditions. Each row carries a per-row comment describing why it isn't a duplicate of another row.

3. **`--include-source-variants` argument removed.**

4. **`combos` construction removed.** The sweep iterates `UNIQUE_CONDITIONS` directly.

5. **Dry-run output.** Each row now prints `counts=..., overlap=...    # note` instead of `counts=..., source=..., overlap=...`.

6. **`run_experiment()` call.** Removed `source=...`. Added explicit `discussion_order="ABC"` so the sweep doesn't accidentally rely on the upstream default.

## `code/metrics.py` — Modified

**Summary:** Added contribution-share percentages (per iteration and overall) to dominance analysis. Surfaced per-iteration team SAD and per-agent proposal SAD as analysis fields. Updated CLI verbose output.

### Changes

1. **`compute_dominance()` — per-iteration `contribution_share_pct` added.** For each final candidate, divides each agent's pair count by `total_pairs` and stores rounded percentages in `contribution_share_pct: Dict[int, float]`. Also adds `unattributed_share_pct: float` for pairs not traceable to any proposal. Sums of the three agent shares plus `unattributed_share_pct` total 100 (modulo rounding).

2. **`analyze_experiment()` — `overall_contribution_share_pct` added.** Aggregates `dominance` + `unattributed` across all iterations into agent-share percentages of the grand total. `overall_unattributed_share_pct` is the matching denominator slice.

3. **`analyze_experiment()` — `iteration_final_sad` and `iteration_proposal_sad` added.** Per-iteration tables. `iteration_proposal_sad` is populated only if proposal entries carry `candidate_scores` (i.e. logs generated by the Phase 8 runner); pre-Phase-8 logs simply produce an empty list.

4. **CLI `--verbose` output extended.** New sections: per-iteration dominance now prints `share%` and `unattrib%`; a new "Iteration final SAD" section lists `best/mean/scores` per iteration; a new "Per-agent proposal SAD" section lists `best/mean` per agent per iteration; the summary block now prints `Overall contribution share %`.

## `code/verify_plumbing.py` — Modified

**Summary:** Updated for the Phase 8 prompt structure (TEAM PROCESS, leader responsibility sentence). Removed source-variant tests. Updated `generate_knowledge_assignment` calls to the new signature.

### Changes

1. **`generate_knowledge_assignment` call sites updated.** All `(counts, "all", overlap, seed=...)` calls changed to `(counts, overlap, seed=...)` or `(counts, seed=...)` where the overlap defaults to `nested`.

2. **Tests 17 and 18 (source filtering) removed.** Source variants are gone.

3. **Test 28 added.** Asserts the leader-responsibility sentence (`"The leader is responsible for guiding the discussion."`) is present in Settings 3/4 prompts and absent in Settings 1/2.

4. **Test 29 added.** Asserts the TEAM PROCESS section and the consensus marker (`"aim to reach a consensus through"`) are present in all 4 settings.

5. **Test 30 added.** Asserts `num_discussion_rounds` is interpolated into the prompt (constructs a `MoonSurvivalAgent` with `num_discussion_rounds=5` and checks `"through 5 rounds of discussion"` is present).

## `code/test_knowledge.py` — Modified

**Summary:** Updated for the Phase 8 overlap semantics and signature.

### Changes

1. **`configs` list rebuilt.** 12 entries (3 counts × 4 overlaps) → 10 entries (2 configs × 5 overlaps). Removes `[1,2,3]` entirely; adds `O5` per config.

2. **`generate_knowledge_assignment` calls updated.** New 2-arg signature `(counts, overlap, seed=42)`.

3. **O3 assertion strengthened.** Was: `items_a & items_c == set()`. Now: `items_b <= items_c` AND `items_a & items_b == set()` AND `items_a & items_c == set()`.

4. **O4 assertion strengthened.** Was: `items_b & items_c == set()`. Now: `items_a <= items_c` AND `items_a & items_b == set()` AND `items_b & items_c == set()`.

5. **O5 assertion added.** Smaller set ⊆ larger; both disjoint from C.

6. **Source-filtering test block removed.** The old `for source in ["all", "top", "bottom"]` loop is gone; `generate_knowledge_assignment` no longer accepts a `source` argument.

7. **O5 [2,2,2] equality sanity check added.** Asserts `items_a == items_b` for the equal-count case.

## `code/incorrect_ranking.json` — New File

**Summary:** Persisted fixed incorrect ranking. One derangement of the 15 ground-truth ranks, generated once and committed so every incorrect-knowledge run uses the same wrong-answer baseline.

### Properties

- **Derangement**: every item has a rank different from its NASA expert rank — 0 item-rank pairs match ground truth.
- **SAD vs ground truth**: 64 / 112 (max).
- **Explanation string**: `"Based on preliminary analysis, this item has been assessed at this priority level."` — same neutral framing the agent would see for correct items.

### Structure

Top-level keys: `_meta` (metadata including `fixed_points`, `sad_vs_ground_truth`, `max_sad`, `derangement`, `explanation`, `source_seed`) and `ranking` (list of 15 `{item, rank}` objects). The explanation is shared (lives in `_meta`) rather than repeated per item to keep the file readable.

### Regeneration

To regenerate with a different seed: delete the file and re-run `generate_incorrect_ranking(seed=...)`; the function falls back to derangement-based generation when the file is absent. The first run after deletion will regenerate the JSON on demand. (Currently this regeneration writes nothing back to disk — re-creating the artifact is a manual step using the snippet in `docs/PHASE_8.md` Task 8F.)

---

# Phase 9 — Selective Port of Weikai Li's Multi-Agent DSE Architecture

Phase 9 imports four targeted elements of Weikai Li's UCLA multi-agent HLS pragma DSE architecture (see `docs/PHASE_9.md` for the full design discussion): (a) per-call raw-prompt + raw-output logging to disk, (b) per-phase system prompts replacing the single shared system message, (c) discussion-history threading using `role=user, name="AgentX"` so peers appear as named users and the agent's own prior turns appear as `assistant`, and (d) candidate IDs (`A1-3` style) referenceable in discussion. Phase 8 schema, F1–F3 feedback dispatch, ABC/CBA discussion order, construct-by-leader final selection, the text-format `CANDIDATE <n> RANKING:` parsing path, and metric outputs are all preserved unchanged.

## `code/util.py` — New File

**Summary:** Shared helper for rendering an OpenAI-format message list as a top-to-bottom transcript for logging.

### Function

1. **`format_messages_for_log(messages: List[Dict[str, str]]) -> str`** — Iterates the message list, tags each block as `[system]` / `[user]` / `[assistant]`, and uses `[user name="X"]` for user messages with a `name` field. Content is dumped verbatim under each tag with a blank line between blocks. Mirrors the format used in the prompt-dump files.

## `code/logger.py` — New File

**Summary:** `Logger` class that writes one prompt file and one output file per LLM call.

### Class

1. **`Logger(experiment_id, output_dir)`** — Creates `<output_dir>/<experiment_id>_raw/prompts/` and `<output_dir>/<experiment_id>_raw/outputs/`. Both directories are created at construction time via `os.makedirs(..., exist_ok=True)`.

2. **`save_prompt_and_output(phase, iteration, agent_name, formatted_messages, response_text, round_num=None)`** — Writes two parallel UTF-8 text files. The filename is `iter{iteration}_{phase}_{agent_name}.txt`, with the special case that `round_num` (when provided) replaces `phase` in the filename — so discussion turns become `iter2_round1_Agent2.txt` rather than `iter2_discuss_Agent2.txt`. The `agent_name` is treated opaquely; in practice it's the agent's `display_name` (e.g., `"Agent1"`).

## `code/agent.py` — Modified

**Summary:** Split the single `_system_prompt()` method into a user-role static context block plus three per-phase system prompts. Added structured-turn threading (`role=user, name="AgentX"` for peers, `role=assistant` for own prior turns), per-call logging via a wrapper around `_chat`, and a `display_name` derivation. `propose_candidates` now returns a 3-tuple `(candidates, reasonings, raw)` so the orchestrator can preserve per-candidate reasoning in the CANDIDATES section.

### Changes

1. **Removed `_system_prompt()`.** All callers redirected: team-info / leader / consensus content moved to `_static_context()`; identity / goal / output-format content moved into the three per-phase system prompts; `=== SCORING ===` and `=== GENERAL GUIDELINES ===` blocks deleted (their content is now folded into the per-phase rules).

2. **Constructor signature.** Added trailing `logger: Optional[Any] = None` parameter and assigned `self.logger`. Also derived `self.display_name = f"Agent{agent_id}"`. Existing parameters and defaults preserved.

3. **Module-level `_FOOTER_WITH_PEERS` constant added.** Verbatim from `docs/PHASE_9.md` "Locked prompts § Shared footer". Appended to all three per-phase system prompts so every system message documents the role conventions consistently.

4. **`_static_context()` added.** Returns the user-role orchestrator block: TASK CONTEXT (crash-landing description), ITEMS TO RANK, YOUR SPECIALISED KNOWLEDGE (with optional "may be incorrect" warning appended via `knowledge_warning`), optional TEAM INFORMATION (Settings 2/4), optional TEAM ROLE with the Phase 8 "responsible for guiding the discussion" sentence (Settings 3/4), and unconditional TEAM PROCESS with the templated consensus sentence. Does NOT include identity, scoring, or guidelines — those moved into the per-phase system prompts.

5. **`_propose_system_prompt()` added.** Verbatim from `docs/PHASE_9.md` "Locked prompts § Propose system prompt" with `{agent_id}` / `{num_agents}` interpolated, followed by `_FOOTER_WITH_PEERS`. Includes CRITICAL RULES (#1 all-15-items, #2 per-candidate reasoning, #3 COMMON REASONING preamble with the parachute-silk / dehydrated-milk worked example) and the output format spec.

6. **`_discuss_system_prompt()` added.** Verbatim from `docs/PHASE_9.md` "Locked prompts § Discuss system prompt" with `{agent_id}` interpolated, footer appended. Includes CRITICAL RULES (#1 reference candidates by ID, #2 real deliberation / disagreement / no-politeness, #3 use SPECIALISED KNOWLEDGE) and commentary-only output instruction.

7. **`_final_select_system_prompt(k)` added.** Verbatim from `docs/PHASE_9.md` "Locked prompts § Final-select system prompt" with `{agent_id}` and `{k}` interpolated, footer appended. Includes CRITICAL RULES (#1 exactly k rankings, #2 minimize SAD, #3 full authority — pool-or-combine-or-construct, #4 prefer SPECIALISED KNOWLEDGE on disagreements) and FINAL CANDIDATE output format.

8. **`_propose_task_block(k, previous_results, results_context)` added.** Renders the per-call user-role addendum: `=== PREVIOUS ITERATION RESULTS ===` (filled with the feedback when `previous_results` is non-None, neutral "No prior results are shown this iteration" wording when None) followed by `=== YOUR TASK ===` asking for k candidates per the system-prompt format. The None-branch wording is deliberately mode-neutral so it's correct under both true iteration 1 and F2/F3 mid-experiment iterations where feedback is suppressed.

9. **`_discuss_task_block(iteration, round_num)` added.** Renders `=== YOUR TASK ===\nComment on the candidates. This is discussion round {round_num}/{self.num_discussion_rounds} of iteration {iteration}.`

10. **`_final_select_task_block(k, iteration)` added.** Renders `=== YOUR TASK ===` noting the discussion has concluded and that the agent may reference candidate IDs in its justification.

11. **`_thread_prior_turns(turns)` added.** Converts a list of structured turn records (`{agent, phase, round_num, raw_output}`) into OpenAI message dicts. Records whose `agent` equals `self.display_name` render as `{"role": "assistant", "content": raw_output}` (no `name` key). All others render as `{"role": "user", "name": agent, "content": raw_output}`.

12. **`_call_and_log(messages, phase, iteration, round_num=None)` added.** Calls `_chat(self.client, self.model, messages).strip()`, then if `self.logger` is set, dumps the formatted message transcript and raw response via `self.logger.save_prompt_and_output(...)`. Returns the response text. Every LLM call in the class now routes through this wrapper, so retry attempts overwrite the same prompt/output files — the final on-disk pair reflects the final attempt.

13. **`propose_candidates()` rewritten.** New return type: `Tuple[List[Dict[str,int]], List[str], str]` = `(candidates, reasonings, raw)`. Messages assembled as `[{system: _propose_system_prompt()}, {user: _static_context() + _propose_task_block(...)}]`. The 3-attempt parse-retry loop from Phase 8 is preserved; each attempt routes through `self._call_and_log(messages, phase="propose", iteration=iteration)`. After successful parse, per-candidate reasonings are extracted via the new module-level helper `_extract_per_candidate_reasoning`. The previous default `iteration=0` is added as a trailing positional/keyword argument so old call sites that don't pass it still work but produce filenames with `iter0_*`.

14. **`discuss()` rewritten.** New signature: `(candidates_block, current_iter_propose_outputs, discussion_history, iteration, round_num)`. Messages: `[{system: _discuss_system_prompt()}, {user: _static_context() + candidates_block + _discuss_task_block(...)}]` followed by `_thread_prior_turns(current_iter_propose_outputs)` then `_thread_prior_turns(discussion_history)` so the agent sees propose outputs first, then prior discussion turns in chronological order. No retry loop — commentary doesn't fail parsing. Returns raw response. Routes through `_call_and_log` with `phase="discuss"` and the round number.

15. **`select_final_candidates()` rewritten.** New signature: `(k, candidates_block, current_iter_propose_outputs, discussion_history, iteration)`. Message assembly mirrors `discuss()` but uses `_final_select_system_prompt(k)` and `_final_select_task_block(k, iteration)`. The 3-attempt parse-retry loop from Phase 8 is preserved with the existing `FINAL CANDIDATE` label. Routes through `_call_and_log` with `phase="final_select"` (no `round_num`).

16. **Module-level `_extract_per_candidate_reasoning(text, k, label)` helper added.** Uses the same regex split as `_parse_k_candidates`; within each candidate block, captures the text between `Reasoning:` (case-insensitive) and the first numbered ranking line (`\n1.` / `\n1)`) via `Reasoning\s*:\s*(.*?)(?=\n\s*1\s*[\.\)])`. Missing reasonings become empty strings. Returns a list parallel to the candidates list.

17. **Unused imports removed.** `import json` and `format_results_summary` are no longer used in this module after the rewrite (the runner still uses `format_results_summary` to build feedback strings before passing them in).

18. **`_chat()` and `_parse_k_candidates()` preserved unchanged.** The regex split in `_parse_k_candidates` already tolerates a `COMMON REASONING:` preamble — it lands in `parts[0]` and is discarded.

## `code/experiment_runner.py` — Modified

**Summary:** Switched the in-memory `discussion_history` from OpenAI-format message dicts to structured turn records, maintained a parallel `current_iter_propose_outputs` list, added a `_build_candidates_block` helper that enumerates candidates with `A{agent_id}-{idx+1}` IDs, and wired a `Logger` instance into every agent constructor. All Phase 8 features preserved unchanged (per-agent proposal SAD scoring, F1/F2/F3 dispatch, ABC/CBA order, incorrect-knowledge handling, JSON schema, CSV summary, verbose prints).

### Changes

1. **Imports updated.** Added `from logger import Logger`. No removals.

2. **`_build_candidates_block(all_proposals)` module-level helper added.** Renders the `=== CANDIDATES ({N}) ===` section: each entry headed by `[A{id}-{idx+1}] (by Agent{id}):`, followed by the 15-item ranking sorted by rank (`  {rank}. {item}` per line), then an optional `  Reasoning: {text}` line when reasoning is non-empty.

3. **`run_iteration()` Phase 1.** `agent.propose_candidates(...)` call updated to unpack the new 3-tuple `(candidates, reasonings, raw)` and to pass `iteration=iteration` so the Logger filenames carry the iteration. Two new in-memory collections are populated alongside the existing `log["proposals"]`: `current_iter_propose_outputs` (one turn record per agent with `phase="propose"`, `round_num=None`) and `all_candidates_records` (one record per candidate, with the `A{agent_id}-{idx+1}` ID and the extracted reasoning). After Phase 1 completes, `candidates_block = _build_candidates_block(all_candidates_records)` is constructed once and reused for both discussion and final-selection.

4. **`run_iteration()` Phase 2.** `discussion_history` is now `List[Dict[str, Any]]` of structured turn records instead of OpenAI message dicts. Each `agent.discuss(...)` call receives `candidates_block`, `current_iter_propose_outputs`, the in-memory `discussion_history`, `iteration`, and `round_num`. After each call, two parallel writes happen: the existing JSON-serializable entry is appended to `iter_log["discussion"]` (`{agent_id, discussion_round, speaking_order, response}` — schema unchanged) and a structured turn record is appended to the in-memory `discussion_history` (`{agent, phase, round_num, raw_output}`).

5. **`run_iteration()` Phase 3.** `last_agent.select_final_candidates(...)` call updated to pass `candidates_block`, `current_iter_propose_outputs`, and `discussion_history` (the in-memory structured form). Return tuple and `log["final_selection"]` shape are unchanged.

6. **`run_experiment()` Logger instantiation.** After `experiment_id` is constructed (and `output_dir` ensured via `os.makedirs(..., exist_ok=True)`), a single `Logger(experiment_id, output_dir)` is created and passed to each `MoonSurvivalAgent(..., logger=logger)`. The Logger creates `<output_dir>/<experiment_id>_raw/prompts/` and `<output_dir>/<experiment_id>_raw/outputs/` at construction time. The directory ordering means `_raw/` exists before any LLM call.

7. **Preserved unchanged.** Setting → team_knowledge_info / leader_id derivation, F1/F2/F3 feedback dispatch and `feedback_provided` flag, per-agent proposal SAD scoring (`candidate_scores` / `best_proposal_sad` / `mean_proposal_sad`), ABC/CBA discussion order selection and `discussion_orders` log field, incorrect-knowledge handling, experiment ID format, trajectory JSON top-level schema, CSV summary writer, all verbose prints.

## `code/verify_plumbing.py` — Modified

**Summary:** Updated existing Phase 8 prompt assertions to call `_static_context()` instead of the removed `_system_prompt()`. Added ~20 new Phase 9 checks covering per-phase prompt content, footer presence, threading behavior, Logger filename behavior, the COMMON REASONING parser preamble, and the regression check on `_propose_task_block` None-branch wording.

### Changes

1. **`os` import added.** Needed by the Logger filename-existence checks.

2. **Existing `_system_prompt()` call sites redirected.** All five Phase 8 prompt-content assertions (Settings 1–4 prompts plus the `num_discussion_rounds=5` template check) now call `_static_context()` — the team-info / leader / consensus content lives there in Phase 9. Existing assertion text is preserved.

3. **Per-phase system prompt content checks added** (9 assertions). Propose has "Each ranking must include ALL 15 items" / "COMMON REASONING" / "parachute silk"; discuss has "Refer to candidates by their IDs" / "real deliberation, not a ceremony" / "do not defer out of politeness" / "Do NOT output a full ranking"; final has "Output exactly 3 final rankings" / "full authority over the final submission".

4. **Footer presence check added.** Confirms `"Your own prior turns appear with role=assistant"` is in all three per-phase system prompts.

5. **`display_name` check added.** Confirms `MoonSurvivalAgent(agent_id=2, ...).display_name == "Agent2"`.

6. **`_propose_task_block` None-branch regression check added** (3 assertions). Calls `agent_p9._propose_task_block(k=3, previous_results=None, results_context="ignored")` and asserts the output (a) does NOT contain "first iteration" (case-insensitive — would be factually wrong under F2/F3 non-feedback iterations), (b) does NOT contain "No prior results exist" (also wrong — results may exist but be suppressed by the feedback schedule), (c) DOES contain "No prior results are shown" (the mode-neutral phrasing that's correct in both true iter 1 and F2/F3 suppressed-feedback iterations).

7. **`_thread_prior_turns` tagging check added** (3 assertions). Confirms peer turns produce `role=user` with `name` set; own turns produce `role=assistant` with no `name` key; another peer agent's turn produces `role=user` with `name`.

8. **Logger filename behavior check added** (3 assertions). Constructs a temporary `Logger`, writes a propose call and a discuss call (with `round_num=3`), and confirms `iter1_propose_Agent1.txt` and `iter2_round3_Agent2.txt` exist on disk in both `prompts/` and `outputs/`.

9. **Parser COMMON REASONING preamble checks added** (3 assertions). Feeds a fake response with a `COMMON REASONING:` paragraph before the first `CANDIDATE 1 RANKING:` header to `_parse_k_candidates`; confirms 1 candidate is returned with all 15 items, and that `_extract_per_candidate_reasoning` pulls out the per-candidate `Reasoning:` text.

## `code/run_experiment.py` — No changes

CLI surface is unchanged. The new Logger and Phase 9 plumbing live entirely below the `run_experiment()` boundary.

## `code/metrics.py` — No changes

Trajectory JSON schema is preserved (Phase 8 top-level keys, per-iteration shape, proposal entries with `candidate_scores`, and the discussion entry schema all unchanged). `analyze_experiment()` runs cleanly on Phase 9 logs without modification.

## `code/knowledge_manager.py` — No changes

## `code/moon_survival_env.py` — No changes (protected)

## Output layout

Each experiment now produces, alongside the existing flat JSON and CSV:

```
results/<experiment_id>_raw/
  prompts/
    iter{N}_propose_Agent{K}.txt
    iter{N}_round{R}_Agent{K}.txt
    iter{N}_final_select_Agent{leader}.txt
  outputs/
    (same filenames, matched output content)
```

For a run with `num_iterations=I`, `num_discussion_rounds=R`, and 3 agents with leader=Agent3, the file count per directory is `I × (3 + 3R + 1)`. The smoke test (`I=2, R=2`) produces exactly 20 files per directory: 6 propose + 12 discuss + 2 final_select.


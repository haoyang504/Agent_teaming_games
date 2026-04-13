# Changelog — Moon Survival Agent Teaming Codebase

All changes relative to the initial commit (`99700ce`). Organized by file.

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

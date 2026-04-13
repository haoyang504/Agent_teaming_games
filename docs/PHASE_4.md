# Phase 4 — Feedback System: Modes F1 through F5

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first. Phases 1–3 must be complete.
> **Goal:** Implement 5 feedback modes that control when and what evaluation feedback agents receive between iterations. After this phase, experiments can be run with any of the 5 feedback settings.

---

## What exists now

In `experiment_runner.py`, the iterative loop tracks `prev_candidates` and `prev_scores` from the most recent iteration and passes them to the next iteration's `run_iteration()`. Inside `run_iteration()`, these are formatted via `format_results_summary()` into `prev_summary`, which is passed to each agent's `propose_candidates(previous_results=prev_summary)`.

In `agent.py`, `propose_candidates()` wraps `previous_results` in a context string: "Here are the results from the previous iteration: ..." and tells the agent to study them.

Current behavior is equivalent to **F1** (feedback every iteration, last iteration only).

---

## The 5 Feedback Modes

| Mode | Type | What agents receive |
|------|------|-------------------|
| F1 | Frequency | SAD scores for last iteration's candidates, every iteration (current behavior) |
| F2 | Frequency | SAD scores for last iteration's candidates, every 2 iterations |
| F3 | Frequency | SAD scores for last iteration's candidates, every 4 iterations |
| F4 | Full history | SAD scores for ALL candidates across ALL prior iterations |
| F5 | Full history | SAD scores for only the TOP 8 candidates (by SAD) across all prior iterations; agents told upfront |

---

## Task 4A: Add `--feedback-mode` CLI argument — DONE

### Changes to `run_experiment.py`

Add a new CLI argument:

```python
parser.add_argument(
    "--feedback-mode",
    type=str,
    default="F1",
    choices=["F1", "F2", "F3", "F4", "F5"],
    help=(
        "Feedback mode (default: F1). "
        "F1=every iter, F2=every 2, F3=every 4, "
        "F4=full history all candidates, F5=full history top 8 only."
    ),
)
```

Pass `args.feedback_mode` to `run_experiment()`.

---

## Task 4B: Implement feedback logic in `experiment_runner.py` — DONE

### Changes to `run_experiment()` signature

Add `feedback_mode` parameter:

```python
def run_experiment(
    counts: List[int],
    source: str = "all",
    overlap: str = "nested",
    setting: int = 1,
    feedback_mode: str = "F1",   # NEW
    k: int = 3,
    ...
```

### Changes to the iterative loop

Replace the current simple `prev_candidates`/`prev_scores` tracking with feedback-mode-aware logic:

```python
# ── Iterative loop ───────────────────────────────────────────────────────
prev_candidates: Optional[List[Dict[str, int]]] = None
prev_scores: Optional[List[int]] = None

# For F4/F5: accumulate all candidates and scores across iterations
all_candidates_history: List[Dict[str, int]] = []
all_scores_history: List[int] = []

# Determine feedback frequency for F1-F3
feedback_frequency = {"F1": 1, "F2": 2, "F3": 4}.get(feedback_mode, 1)

for it in range(1, num_iterations + 1):
    # ── Determine what feedback to pass this iteration ──
    if feedback_mode in ("F1", "F2", "F3"):
        # Frequency-based: pass last iteration's results on feedback iterations only
        if it == 1:
            iter_candidates = None
            iter_scores = None
        elif (it - 1) % feedback_frequency == 0:
            # This is a feedback iteration — pass last iteration's results
            iter_candidates = prev_candidates
            iter_scores = prev_scores
        else:
            # Non-feedback iteration — agents get no evaluation data
            iter_candidates = None
            iter_scores = None

    elif feedback_mode == "F4":
        # Full history: pass ALL accumulated candidates and scores
        if all_candidates_history:
            iter_candidates = list(all_candidates_history)
            iter_scores = list(all_scores_history)
        else:
            iter_candidates = None
            iter_scores = None

    elif feedback_mode == "F5":
        # Full history, top 8 only: filter to best 8 candidates by SAD
        if all_candidates_history:
            paired = list(zip(all_scores_history, all_candidates_history))
            paired.sort(key=lambda x: x[0])  # sort by SAD ascending (best first)
            top_8 = paired[:8]
            iter_scores = [s for s, _ in top_8]
            iter_candidates = [c for _, c in top_8]
        else:
            iter_candidates = None
            iter_scores = None

    final_candidates, final_scores, iter_log = run_iteration(
        agents=agents,
        k=k,
        iteration=it,
        previous_candidates=iter_candidates,
        previous_scores=iter_scores,
        num_discussion_rounds=num_discussion_rounds,
        discussion_seed=knowledge_seed + it,
        verbose=verbose,
    )
    experiment_log["iterations"].append(iter_log)

    # Update tracking
    prev_candidates = final_candidates
    prev_scores = final_scores

    # Accumulate for F4/F5
    all_candidates_history.extend(final_candidates)
    all_scores_history.extend(final_scores)
```

### Feedback iteration logging

Add to each iteration's log whether feedback was provided:

In `run_iteration()`, add a `feedback_provided` field to the log. The simplest way: check if `previous_candidates` is None or not.

```python
log: Dict[str, Any] = {
    "iteration": iteration,
    "feedback_provided": previous_candidates is not None,  # NEW
    "proposals": [],
    ...
}
```

---

## Task 4C: Update `format_results_summary()` context in `agent.py` — DONE

### The problem

Currently, `propose_candidates()` always says "Here are the results from the previous iteration." For F4/F5, this is misleading — the results span all prior iterations, not just the last one.

### The fix

Add an optional `results_context` parameter to `propose_candidates()`:

```python
def propose_candidates(
    self,
    k: int,
    previous_results: Optional[str] = None,
    results_context: str = "the previous iteration",  # NEW
) -> Tuple[List[Dict[str, int]], str]:
```

Update the context string to use it:

```python
if previous_results:
    context = (
        f"Here are the results from {results_context}:\n\n"
        f"{previous_results}\n\n"
        "Study these results carefully. You are NOT limited to building "
        "on any single previous candidate — you may combine insights "
        "across all of them."
    )
```

### Calling from `run_iteration()`

Add `results_context` parameter to `run_iteration()`:

```python
def run_iteration(
    agents: List[MoonSurvivalAgent],
    k: int,
    iteration: int,
    previous_candidates: Optional[List[Dict[str, int]]],
    previous_scores: Optional[List[int]],
    num_discussion_rounds: int = 1,
    discussion_seed: Optional[int] = None,
    results_context: str = "the previous iteration",  # NEW
    verbose: bool = True,
) -> Tuple[List[Dict[str, int]], List[int], Dict[str, Any]]:
```

Then pass it through when calling `propose_candidates()`:

```python
candidates, raw = agent.propose_candidates(k, prev_summary, results_context=results_context)
```

### Calling from `run_experiment()`

Determine the context string based on feedback mode:

```python
# Determine results context label
if feedback_mode in ("F1", "F2", "F3"):
    results_context = "the previous iteration"
elif feedback_mode == "F4":
    results_context = "all prior iterations"
elif feedback_mode == "F5":
    results_context = "all prior iterations (showing top 8 candidates only)"
```

Pass to `run_iteration()`:

```python
final_candidates, final_scores, iter_log = run_iteration(
    ...
    results_context=results_context,
    ...
)
```

---

## Task 4D: Logging and experiment ID — DONE

### Experiment ID

Add feedback mode to the experiment ID string in `run_experiment()`:

```python
experiment_id = (
    f"c{counts_label}__{source}__{overlap}__s{setting}__fb{feedback_mode}"
    f"__k{k}__iter{num_iterations}__disc{num_discussion_rounds}__{timestamp}"
)
```

### Experiment log

Add `feedback_mode` to the experiment log dict:

```python
experiment_log: Dict[str, Any] = {
    "experiment_id": experiment_id,
    "counts": counts,
    "source": source,
    "overlap": overlap,
    "setting": setting,
    "feedback_mode": feedback_mode,  # NEW
    ...
}
```

### Verbose output

Add feedback mode to the experiment header:

```python
if verbose:
    print(f"  Feedback mode: {feedback_mode}")
```

---

## Tests

### Test 1: F1 baseline unchanged

Run and verify current behavior is preserved:

```bash
python run_experiment.py --config high_div --setting 1 --feedback-mode F1 --k 1 --iterations 3 --discussion-rounds 1
```

Check JSON log: `feedback_provided` should be `false` for iteration 1, `true` for iterations 2 and 3. `feedback_mode` should be `"F1"`.

### Test 2: F2 frequency

```bash
python run_experiment.py --config high_div --setting 1 --feedback-mode F2 --k 1 --iterations 5 --discussion-rounds 1
```

Check JSON log: `feedback_provided` should follow the pattern `false, true, false, true, false` (iterations 1–5). Feedback appears on iterations where `(it - 1) % 2 == 0`, i.e., iterations 2 and 4.

### Test 3: F3 frequency

```bash
python run_experiment.py --config high_div --setting 1 --feedback-mode F3 --k 1 --iterations 5 --discussion-rounds 1
```

Check JSON log: `feedback_provided` should be `false, false, false, true, false` (iterations 1–5). Only iteration 4 gets feedback.

### Test 4: F4 full history

```bash
python run_experiment.py --config high_div --setting 1 --feedback-mode F4 --k 3 --iterations 3 --discussion-rounds 1
```

Check JSON log: iteration 1 has no feedback. Iteration 2's proposal prompt should reference results from iteration 1 (3 candidates). Iteration 3's proposal prompt should reference results from iterations 1 AND 2 (6 candidates total). Verify by checking `raw_response` in proposals — agents should reference more candidates in later iterations.

### Test 5: F5 top-8 filtering

```bash
python run_experiment.py --config high_div --setting 1 --feedback-mode F5 --k 3 --iterations 4 --discussion-rounds 1
```

After 3 iterations with k=3, there are 9 accumulated candidates. Iteration 4 should only see 8 of them (the 8 with lowest SAD). Check the proposal prompt in the JSON — it should say "top 8 candidates only" in the context and show exactly 8 candidates.

### Test 6: Verify results_context text

For F1/F2/F3 runs, check that proposal prompts say "results from the previous iteration."
For F4 runs, check that proposal prompts say "results from all prior iterations."
For F5 runs, check that proposal prompts say "results from all prior iterations (showing top 8 candidates only)."

---

## What NOT to do in Phase 4

- Do NOT modify `moon_survival_env.py` — `format_results_summary()` stays unchanged. The summary formatting function works the same regardless of whether it's formatting 3 candidates or 30.
- Do NOT modify `knowledge_manager.py`.
- Do NOT change discussion mechanics (speaking order, history scope) — that's done in Phase 3.
- Do NOT implement incorrect knowledge — that's Phase 5.
- Keep the `agent.py` changes minimal — only the `results_context` parameter addition to `propose_candidates()`. No system prompt changes for F5 (the context is communicated through the results text itself).

---

## Implementation Notes (post-completion)

### Feedback frequency formula bug

The spec's suggested code used `(it - 1) % feedback_frequency == 0`, which gives the wrong pattern. For F2 (every 2 iterations), this would give feedback on iterations 1, 3, 5 instead of 2, 4, 6. The correct formula is `it % feedback_frequency == 0`:
- F1: feedback on iterations 2, 3, 4, 5, ... (every iteration after the first)
- F2: feedback on iterations 2, 4, 6, ... (every 2nd iteration)
- F3: feedback on iterations 4, 8, 12, ... (every 4th iteration)

Iteration 1 never gets feedback (no prior data exists).

### Files changed

- **`run_experiment.py`**: Added `--feedback-mode` CLI arg with choices F1–F5, default F1. Passed through to `run_experiment()`.
- **`experiment_runner.py`**: Added `feedback_mode` parameter. Iterative loop now determines feedback per mode. Added `feedback_provided` boolean to each iteration log. Added `feedback_mode` to experiment log and experiment ID. Added `results_context` passthrough.
- **`agent.py`**: Added `results_context` parameter to `propose_candidates()`. Context string now says "results from {results_context}" instead of hardcoded "results from the previous iteration".

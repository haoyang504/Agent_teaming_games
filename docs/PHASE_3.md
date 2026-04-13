# Phase 3 — Discussion Mechanics: Randomized Order, Scale, History Control

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first. Phases 1 and 2 must be complete.
> **Goal:** Randomize agent speaking order each discussion round, update default iteration/round counts to match the experimental protocol, and verify that communication history is correctly scoped. All changes are in `experiment_runner.py` and `run_experiment.py`.

---

## What exists now

In `experiment_runner.py`, the discussion phase (Phase 2 of `run_iteration()`) iterates over agents in the fixed order they were created: Agent 1 → Agent 2 → Agent 3, every round. The `discussion_history` list accumulates within a single iteration (proposals + all discussion turns) and is built fresh each iteration — it does not carry over from prior iterations. Feedback from prior iterations is passed via `previous_candidates` / `previous_scores` which get formatted into `prev_summary` and given to agents during their proposal phase.

Current defaults in `run_experiment.py`: `--iterations 3`, `--discussion-rounds 1`.

The experimental design specifies: speaking order randomized each round, 3 rounds per iteration with up to 20 iterations, and agents should see only the previous iteration's communication (not full history).

---

## Task 3A: Randomize Speaking Order Per Discussion Round — DONE

### What to change in `experiment_runner.py`

In `run_iteration()`, add a `discussion_seed` parameter (or derive randomness from a passed-in RNG) so that speaking order is shuffled at the start of each discussion round but remains reproducible.

**Option 1 (simpler):** Accept an optional `rng` parameter (a `random.Random` instance) in `run_iteration()`. If not provided, create one with a default seed.

**Option 2 (recommended):** Accept a `discussion_seed` parameter in `run_iteration()`, create a local `random.Random(discussion_seed)` inside the function, and use it to shuffle.

In the `run_experiment()` function, derive per-iteration discussion seeds from the main `knowledge_seed` so results are reproducible:

```python
discussion_seed = knowledge_seed + it  # different per iteration, reproducible
```

### Changes to the discussion loop

Currently:
```python
for disc_round in range(1, num_discussion_rounds + 1):
    for agent in agents:
        # agent discusses
```

Change to:
```python
import random

for disc_round in range(1, num_discussion_rounds + 1):
    # Shuffle speaking order for this round
    round_order = list(agents)
    disc_rng.shuffle(round_order)
    
    if verbose:
        order_str = " → ".join(str(a.agent_id) for a in round_order)
        print(f"\n  -- Discussion Round {disc_round}/{num_discussion_rounds} (order: {order_str}) --")
    
    for agent in round_order:
        # agent discusses (same logic as before)
```

### Logging

Add the speaking order to the iteration log so it's recorded in the JSON output:

```python
discussion_entry = {
    "agent_id": agent.agent_id,
    "discussion_round": disc_round,
    "speaking_order": [a.agent_id for a in round_order],  # NEW
    "response": response,
}
```

Also add a top-level `"discussion_orders"` field to the iteration log:
```python
log["discussion_orders"] = {}  # will be filled as {round_num: [agent_ids]}
```

Then inside the round loop:
```python
log["discussion_orders"][disc_round] = [a.agent_id for a in round_order]
```

### Important: Proposal order stays fixed

Only the discussion phase gets randomized order. Proposals (Phase 1) should stay in the fixed 1→2→3 order. Final selection (Phase 3) is always done by the last agent (Agent 3 / Agent C), regardless of discussion order.

---

## Task 3B: Update Default Iteration and Round Counts — DONE

### What to change in `run_experiment.py`

Update the CLI argument defaults:

```python
parser.add_argument(
    "--iterations",
    type=int,
    default=3,  # Keep at 3 for now — full protocol is 20 but expensive
    help="Number of iterative rounds (default: 3). Full protocol: 20.",
)
parser.add_argument(
    "--discussion-rounds",
    type=int,
    default=3,  # Changed from 1 to 3
    help=(
        "Number of discussion rounds per iteration (default: 3). "
        "Each round = all 3 agents speak once in randomized order."
    ),
)
```

The experimental design says 3 rounds × 20 iterations. Setting `--discussion-rounds` default to 3 matches the protocol. Keep `--iterations` at 3 as the default for development speed — 20 iterations is expensive and should be used for production runs only. Document 20 as the full protocol value in the help text.

---

## Task 3C: Verify Communication History Scope — DONE (no code changes needed)

### What to verify

The experimental design says agents should receive "just last iteration communication (+Feedbacks from all prior iterations)." Check the current behavior:

1. **Discussion history resets each iteration** — YES, this is already correct. `discussion_history` is built fresh inside `run_iteration()` starting with proposals from the current iteration. It does not carry over from prior iterations.

2. **Feedback from prior iterations is passed** — YES, this is already correct. `prev_summary` (formatted from `previous_candidates` and `previous_scores`) is passed to `propose_candidates()` at the start of each iteration.

So the current behavior already matches the design. **No code changes needed for Task 3C** — just verify and document.

### What to add to the log

Add a field to the experiment log confirming the history scope:

```python
experiment_log["history_scope"] = "last_iteration_only"
```

This makes it explicit in the JSON output that agents only see the current iteration's discussion, not the full history.

---

## Task 3D: Pass `discussion_seed` through the pipeline — DONE

### Changes to `run_experiment()`

The `run_experiment()` function needs to pass a seed to `run_iteration()` for reproducible shuffling:

```python
for it in range(1, num_iterations + 1):
    final_candidates, final_scores, iter_log = run_iteration(
        agents=agents,
        k=k,
        iteration=it,
        previous_candidates=prev_candidates,
        previous_scores=prev_scores,
        num_discussion_rounds=num_discussion_rounds,
        discussion_seed=knowledge_seed + it,  # NEW — reproducible per iteration
        verbose=verbose,
    )
```

### Changes to `run_iteration()` signature

```python
def run_iteration(
    agents: List[MoonSurvivalAgent],
    k: int,
    iteration: int,
    previous_candidates: Optional[List[Dict[str, int]]],
    previous_scores: Optional[List[int]],
    num_discussion_rounds: int = 1,
    discussion_seed: Optional[int] = None,  # NEW
    verbose: bool = True,
) -> Tuple[List[Dict[str, int]], List[int], Dict[str, Any]]:
```

Inside the function, create the RNG:
```python
disc_rng = random.Random(discussion_seed) if discussion_seed is not None else random.Random()
```

---

## Tests

### Test 1: Speaking order is randomized and logged

Run a short experiment and inspect the JSON log:

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 2 --discussion-rounds 3
```

In the JSON log, check:
- `discussion_orders` exists in each iteration log
- Each round within an iteration has a potentially different order
- Orders vary between iterations
- All 3 agents appear in every round (no agent is skipped)

### Test 2: Reproducibility

Run the same experiment twice with the same seed:

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 2 --discussion-rounds 3 --seed 42
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 2 --discussion-rounds 3 --seed 42
```

The `discussion_orders` in both JSON logs should be identical (same shuffled orders for each round in each iteration). Note: LLM responses will differ due to API non-determinism, but the speaking orders must match.

### Test 3: Default discussion rounds

Run without specifying `--discussion-rounds`:

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1
```

Verify the output shows 3 discussion rounds (the new default), not 1.

### Test 4: Proposal order unchanged

In the JSON log, check that proposals are always in order: agent_id 1, then 2, then 3. Only discussion order should be randomized.

### Test 5: History scope verification

Run a 2-iteration experiment and inspect the JSON log. In iteration 2's proposals, the agents should reference `prev_summary` (SAD scores from iteration 1) but should NOT reference specific discussion points from iteration 1's discussion phase. This confirms discussion history resets between iterations.

---

## What NOT to do in Phase 3

- Do NOT modify `agent.py` — no prompt changes needed.
- Do NOT modify `knowledge_manager.py` or `moon_survival_env.py`.
- Do NOT implement feedback frequency/mode logic — that's Phase 4.
- Do NOT change which agent does `select_final_candidates()` — it stays as Agent 3 (the last agent).
- Do NOT randomize proposal order — only discussion order is randomized.
- Do NOT carry discussion history across iterations — the current reset-per-iteration behavior is correct.

---

## Implementation Notes (post-completion)

### Discussion order randomization

Each discussion round shuffles agent speaking order using `disc_rng.shuffle()` where `disc_rng = random.Random(knowledge_seed + iteration)`. Orders vary across rounds within an iteration and across iterations, but are fully reproducible with the same seed.

Example with seed=42:
- Iteration 1: Round 1 → 3,2,1 | Round 2 → 2,1,3 | Round 3 → 1,3,2
- Iteration 2: Round 1 → 3,1,2 | Round 2 → 3,2,1 | Round 3 → 3,2,1

### JSON log additions

- `discussion_orders`: dict mapping round number → list of agent_ids in speaking order
- `speaking_order`: added to each discussion entry for per-entry traceability
- `history_scope`: `"last_iteration_only"` — documents that discussion history resets per iteration

### Task 3C verification

Confirmed: `discussion_history` is built fresh inside `run_iteration()` starting with the current iteration's proposals. Prior iteration feedback is passed only via `prev_summary` in the proposal phase. No code changes needed — behavior already matched the design.

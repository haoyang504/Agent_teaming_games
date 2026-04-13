# Phase 5 — Incorrect Knowledge Phase

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first. Phases 1–4 must be complete.
> **Goal:** Implement the incorrect knowledge system — generate fully incorrect rankings, distribute a mix of correct and incorrect knowledge to agents, enforce overlap constraints, and add a warning condition to the prompt. This phase runs *after* baseline experiments, not as a modification to them.

---

## What exists now

`knowledge_manager.py` generates correct knowledge: each agent receives tuples of `(item_name, ground_truth_rank, explanation)` drawn from the NASA expert ranking. There is no concept of incorrect knowledge.

`agent.py`'s `_system_prompt()` presents all knowledge as trustworthy: "You have specialised knowledge (from NASA training)..." There is no mechanism to warn agents that some knowledge may be wrong.

`run_experiment.py` and `experiment_runner.py` have no parameters for incorrect knowledge distribution or warning conditions.

---

## Design Recap (from ES)

1. Generate a **fully incorrect ranking** — same 15 items, randomly assigned ranks where **no item-rank pair matches** the correct solution.
2. Mix correct and incorrect knowledge per agent in 4 distribution patterns (I1–I4).
3. Fix knowledge assignment to: source = `"all"` (V1), overlap = `"disjoint"` (O2).
4. **Overlap constraint:** If either the item OR the rank matches between a correct and incorrect knowledge tuple for the same agent, exclude the incorrect tuple and resample.
5. **Warning condition:** With or without a prompt line telling agents some knowledge may be incorrect.
6. **Config 1 special case:** In {0,2,4}, Agent A has 0 items and can't receive incorrect knowledge. Agent C receives 2 incorrect items instead.
7. Config 2 ({1,2,3}) needs no special handling — all agents have ≥1 item.

---

## Task 5A: Add incorrect ranking generator to `knowledge_manager.py` — DONE

### New function: `generate_incorrect_ranking()`

```python
def generate_incorrect_ranking(seed: int = 99) -> List[Tuple[str, int, str]]:
    """Generate a fully incorrect ranking where no item-rank pair matches ground truth.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        List of 15 (item_name, incorrect_rank, incorrect_explanation) tuples.
        The incorrect_explanation is a generic placeholder.
    """
```

**Algorithm:**
1. Start with the list of 15 items and their ground truth ranks (1–15).
2. Create a derangement of the ranks — a permutation where no element appears in its original position. This guarantees no item-rank pair matches.
3. For the explanation, use a generic string like: `"Based on preliminary analysis, this item has been assessed at this priority level."` — the explanation should sound plausible but not give away that it's wrong.

**Derangement algorithm:**
```python
def _derangement(rng: random.Random, n: int) -> List[int]:
    """Generate a random derangement of [1..n] (no fixed points)."""
    while True:
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        if all(perm[i] != i + 1 for i in range(n)):
            return perm
```

**Return format:** Same tuple structure as correct knowledge — `(item_name, rank, explanation)` — so it can be mixed seamlessly with correct knowledge.

---

## Task 5B: Add incorrect knowledge distribution to `knowledge_manager.py` — DONE

### New function: `generate_mixed_knowledge_assignment()`

```python
def generate_mixed_knowledge_assignment(
    counts: List[int],
    incorrect_pattern: str,  # "I1", "I2", "I3", "I4"
    seed: int = 42,
    incorrect_seed: int = 99,
) -> Tuple[List[List[Tuple[str, int, str]]], List[List[Tuple[str, int, str]]]]:
    """Generate correct + incorrect knowledge for each agent.

    Fixed to: source="all", overlap="disjoint".

    Args:
        counts:            [A, B, C] item counts.
        incorrect_pattern: Distribution pattern for incorrect knowledge.
            "I1" = only Agent A gets 1 incorrect item
            "I2" = only Agent B gets 1 incorrect item
            "I3" = only Agent C gets 1 incorrect item
            "I4" = all agents get 1 incorrect item
                   (Config 1 exception: Agent C gets 2 instead of A getting 1)
        seed:              Seed for correct knowledge assignment.
        incorrect_seed:    Seed for incorrect ranking generation.

    Returns:
        Tuple of:
          - correct_assignments: List of 3 lists of correct (item, rank, explanation) tuples.
          - incorrect_assignments: List of 3 lists of incorrect (item, rank, explanation) tuples.
        The caller should merge these lists when building the agent's knowledge.
    """
```

**Logic:**

1. Generate correct assignments using `generate_knowledge_assignment(counts, source="all", overlap="disjoint", seed=seed)`.
2. Generate the incorrect ranking using `generate_incorrect_ranking(seed=incorrect_seed)`.
3. Determine how many incorrect items each agent gets based on `incorrect_pattern`:

```python
if incorrect_pattern == "I1":
    incorrect_counts = [1, 0, 0]  # Only Agent A
elif incorrect_pattern == "I2":
    incorrect_counts = [0, 1, 0]  # Only Agent B
elif incorrect_pattern == "I3":
    incorrect_counts = [0, 0, 1]  # Only Agent C
elif incorrect_pattern == "I4":
    # All get 1, but Config 1 exception: if Agent A has 0 correct items,
    # they can't receive incorrect items — give Agent C 2 instead
    if counts[0] == 0:
        incorrect_counts = [0, 1, 2]
    else:
        incorrect_counts = [1, 1, 1]
```

4. For each agent, sample their incorrect items from the incorrect ranking, enforcing the overlap constraint.

### Overlap constraint enforcement

For each agent, an incorrect tuple `(item, incorrect_rank, explanation)` is invalid if:
- The **item** appears in the agent's correct knowledge (same item, different rank — still overlapping by item), OR
- The **incorrect_rank** matches any rank in the agent's correct knowledge (different item, same rank — overlapping by rank)

```python
def _sample_incorrect_for_agent(
    rng: random.Random,
    incorrect_ranking: List[Tuple[str, int, str]],
    correct_knowledge: List[Tuple[str, int, str]],
    count: int,
) -> List[Tuple[str, int, str]]:
    """Sample incorrect items for one agent, avoiding overlap with their correct knowledge."""
    if count == 0:
        return []
    
    correct_items = {item for item, _, _ in correct_knowledge}
    correct_ranks = {rank for _, rank, _ in correct_knowledge}
    
    eligible = [
        (item, rank, expl) for item, rank, expl in incorrect_ranking
        if item not in correct_items and rank not in correct_ranks
    ]
    
    if len(eligible) < count:
        raise ValueError(
            f"Cannot find {count} non-overlapping incorrect items. "
            f"Only {len(eligible)} eligible after filtering."
        )
    
    return rng.sample(eligible, count)
```

### Return value

Return both correct and incorrect assignments separately. The caller (`experiment_runner.py`) will merge them into each agent's knowledge list before passing to the agent constructor. This keeps the knowledge manager's responsibility clean — it generates, the runner assembles.

---

## Task 5C: Add warning condition to `agent.py` — DONE

### Constructor change

Add one new optional parameter:

```python
def __init__(
    self,
    agent_id: int,
    knowledge: List[Tuple[str, int, str]],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    num_agents: int = 3,
    team_knowledge_info: Optional[Dict[str, int]] = None,
    leader_id: Optional[int] = None,
    knowledge_warning: bool = False,  # NEW
) -> None:
```

Store as `self.knowledge_warning`.

### System prompt change

In `_system_prompt()`, if `self.knowledge_warning` is True, add a line at the end of the knowledge block (after `format_knowledge_for_prompt()`):

```python
knowledge_block = format_knowledge_for_prompt(self.knowledge)
if self.knowledge_warning:
    knowledge_block += (
        "\n\nNote: Some of the knowledge you have received may be incorrect."
    )
```

**CRITICAL:** Keep it neutral and brief. No additional instructions like "be extra careful" or "verify with teammates." Just the factual statement.

---

## Task 5D: Wire up in `experiment_runner.py` and `run_experiment.py` — DONE

### New CLI arguments in `run_experiment.py`

```python
parser.add_argument(
    "--incorrect-pattern",
    type=str,
    default=None,
    choices=["I1", "I2", "I3", "I4"],
    help=(
        "Incorrect knowledge distribution pattern. "
        "I1=Agent A only, I2=Agent B only, I3=Agent C only, "
        "I4=all agents (Config 1: Agent C gets 2 instead of A). "
        "If not set, no incorrect knowledge is used."
    ),
)
parser.add_argument(
    "--incorrect-warning",
    action="store_true",
    default=False,
    help="Add a warning to agents that some knowledge may be incorrect.",
)
parser.add_argument(
    "--incorrect-seed",
    type=int,
    default=99,
    help="Seed for incorrect ranking generation (default: 99).",
)
```

### Changes to `run_experiment()` in `experiment_runner.py`

Add parameters: `incorrect_pattern`, `incorrect_warning`, `incorrect_seed`.

```python
def run_experiment(
    counts: List[int],
    source: str = "all",
    overlap: str = "nested",
    setting: int = 1,
    feedback_mode: str = "F1",
    incorrect_pattern: Optional[str] = None,   # NEW
    incorrect_warning: bool = False,            # NEW
    incorrect_seed: int = 99,                   # NEW
    k: int = 3,
    ...
```

**When `incorrect_pattern` is not None**, replace the normal knowledge assignment flow:

```python
if incorrect_pattern:
    from knowledge_manager import generate_mixed_knowledge_assignment
    
    correct_assignments, incorrect_assignments = generate_mixed_knowledge_assignment(
        counts=counts,
        incorrect_pattern=incorrect_pattern,
        seed=knowledge_seed,
        incorrect_seed=incorrect_seed,
    )
    # Merge correct + incorrect for each agent
    knowledge_assignments = [
        correct + incorrect
        for correct, incorrect in zip(correct_assignments, incorrect_assignments)
    ]
    # Note: source and overlap are fixed to "all" and "disjoint" inside
    # generate_mixed_knowledge_assignment, regardless of CLI args
else:
    knowledge_assignments = generate_knowledge_assignment(
        counts, source=source, overlap=overlap, seed=knowledge_seed
    )
```

Pass `incorrect_warning` to agent constructors:

```python
agents = [
    MoonSurvivalAgent(
        agent_id=i + 1,
        knowledge=knowledge_assignments[i],
        client=client,
        model=model,
        num_agents=3,
        team_knowledge_info=team_knowledge_info,
        leader_id=leader_id,
        knowledge_warning=incorrect_warning,  # NEW
    )
    for i in range(3)
]
```

### Logging

Add to experiment log:

```python
experiment_log["incorrect_pattern"] = incorrect_pattern
experiment_log["incorrect_warning"] = incorrect_warning
experiment_log["incorrect_seed"] = incorrect_seed
```

Add incorrect knowledge details to `agent_knowledge_log`:

```python
if incorrect_pattern:
    for i, incorrect in enumerate(incorrect_assignments):
        agent_knowledge_log[i]["incorrect_items"] = [
            {"item": item, "incorrect_rank": rank, "explanation": expl}
            for item, rank, expl in incorrect
        ]
```

Add to experiment ID:

```python
if incorrect_pattern:
    experiment_id += f"__inc{incorrect_pattern}"
    if incorrect_warning:
        experiment_id += "_warn"
```

---

## Tests

### Test 1: Incorrect ranking generator

Create a test script or add to `verify_plumbing.py`:

```python
from knowledge_manager import generate_incorrect_ranking
from moon_survival_env import GROUND_TRUTH_RANKS, ITEMS

incorrect = generate_incorrect_ranking(seed=99)

# All 15 items present
assert len(incorrect) == 15, f"Expected 15 items, got {len(incorrect)}"

# No item-rank pair matches ground truth
for (item, inc_rank, _), gt_rank in zip(incorrect, GROUND_TRUTH_RANKS):
    assert inc_rank != gt_rank, f"Item '{item}' has incorrect rank {inc_rank} == ground truth {gt_rank}"

# All ranks 1-15 present (it's a permutation)
inc_ranks = sorted(r for _, r, _ in incorrect)
assert inc_ranks == list(range(1, 16)), "Incorrect ranks are not a valid permutation"

# Reproducible
incorrect2 = generate_incorrect_ranking(seed=99)
assert incorrect == incorrect2, "Same seed should produce same incorrect ranking"

# Different seed = different ranking
incorrect3 = generate_incorrect_ranking(seed=100)
assert incorrect != incorrect3, "Different seeds should produce different rankings"

print("PASS: incorrect ranking generator")
```

### Test 2: Overlap constraint enforcement

```python
from knowledge_manager import generate_mixed_knowledge_assignment

# Config 1 with I4 — Agent A has 0 items, so Agent C gets 2 incorrect
correct, incorrect = generate_mixed_knowledge_assignment([0, 2, 4], "I4", seed=42)

assert len(incorrect[0]) == 0, "Config 1 I4: Agent A should get 0 incorrect (has 0 correct)"
assert len(incorrect[1]) == 1, "Config 1 I4: Agent B should get 1 incorrect"
assert len(incorrect[2]) == 2, "Config 1 I4: Agent C should get 2 incorrect"

# Verify no overlap: for each agent, incorrect items don't share item name or rank with correct
for i in range(3):
    correct_items = {item for item, _, _ in correct[i]}
    correct_ranks = {rank for _, rank, _ in correct[i]}
    for item, rank, _ in incorrect[i]:
        assert item not in correct_items, f"Agent {i}: incorrect item '{item}' overlaps with correct items"
        assert rank not in correct_ranks, f"Agent {i}: incorrect rank {rank} overlaps with correct ranks"

print("PASS: overlap constraint enforcement")
```

### Test 3: Config 2 I4 — no special case needed

```python
correct, incorrect = generate_mixed_knowledge_assignment([1, 2, 3], "I4", seed=42)

assert len(incorrect[0]) == 1, "Config 2 I4: Agent A should get 1 incorrect"
assert len(incorrect[1]) == 1, "Config 2 I4: Agent B should get 1 incorrect"
assert len(incorrect[2]) == 1, "Config 2 I4: Agent C should get 1 incorrect"

print("PASS: Config 2 I4 no special case")
```

### Test 4: Warning in prompt

```python
from agent import MoonSurvivalAgent

# Without warning
agent_no_warn = MoonSurvivalAgent(
    agent_id=1, knowledge=[("Two 100-lb oxygen tanks", 1, "Oxygen.")],
    client=None, model="test", num_agents=3, knowledge_warning=False,
)
prompt_no_warn = agent_no_warn._system_prompt()
assert "may be incorrect" not in prompt_no_warn, "No warning when knowledge_warning=False"

# With warning
agent_warn = MoonSurvivalAgent(
    agent_id=1, knowledge=[("Two 100-lb oxygen tanks", 1, "Oxygen.")],
    client=None, model="test", num_agents=3, knowledge_warning=True,
)
prompt_warn = agent_warn._system_prompt()
assert "may be incorrect" in prompt_warn, "Warning present when knowledge_warning=True"

print("PASS: warning condition")
```

### Test 5: End-to-end pipeline

```bash
# With incorrect knowledge and warning
python run_experiment.py --config high_div --setting 1 --incorrect-pattern I3 --incorrect-warning --k 1 --iterations 1 --discussion-rounds 1

# Without warning
python run_experiment.py --config high_div --setting 1 --incorrect-pattern I4 --k 1 --iterations 1 --discussion-rounds 1
```

Check JSON logs:
- `incorrect_pattern` and `incorrect_warning` fields present
- Agent knowledge log includes `incorrect_items` entries
- Agents with incorrect knowledge have mixed correct + incorrect items in their knowledge
- The correct ranking items show real NASA explanations; incorrect items show the generic placeholder

### Test 6: All 4 patterns × 2 warnings = 8 conditions (dry run)

```bash
for pattern in I1 I2 I3 I4; do
  for warn in "" "--incorrect-warning"; do
    echo "=== $pattern $warn ==="
    python run_experiment.py --config high_div --setting 1 --incorrect-pattern $pattern $warn --k 1 --iterations 1 --discussion-rounds 1
  done
done
```

Verify all 8 complete without errors.

---

## What NOT to do in Phase 5

- Do NOT modify `moon_survival_env.py`.
- Do NOT change the existing `generate_knowledge_assignment()` function — the new `generate_mixed_knowledge_assignment()` is a separate function that calls it internally.
- Do NOT add behavioral instructions to the warning. Just "Some of the knowledge you have received may be incorrect." — nothing more.
- Do NOT change how incorrect knowledge is presented in the prompt. It should look identical to correct knowledge (same format: item name, rank, explanation). The agent should not be able to distinguish correct from incorrect knowledge by formatting alone.
- Do NOT override `source` or `overlap` CLI args when using incorrect patterns — the function internally fixes to "all" and "disjoint" regardless.
- Do NOT modify discussion mechanics, feedback modes, or hierarchy settings — those are independent dimensions.

---

## Implementation Notes (post-completion)

### Derangement algorithm
Uses rejection sampling: shuffle ranks, check no fixed points, retry if any match. For n=15 this converges quickly (~63% chance per attempt).

### Overlap constraint
For each agent, incorrect tuples are filtered out if either the item name or the rank matches any tuple in that agent's correct knowledge. This is stricter than just item matching — it prevents rank collisions too.

### Config 1 I4 special case
When `counts[0] == 0` (Agent A has no items), the incorrect counts shift from `[1,1,1]` to `[0,1,2]` — Agent C gets 2 incorrect items instead. Verified across all configs and patterns (4 patterns × 3 configs = 12 conditions, all passing).

### Files changed
- **`knowledge_manager.py`**: Added `generate_incorrect_ranking()`, `generate_mixed_knowledge_assignment()`, `_derangement()`, `_sample_incorrect_for_agent()`.
- **`agent.py`**: Added `knowledge_warning` parameter. Appends neutral warning to knowledge block when True.
- **`experiment_runner.py`**: Added `incorrect_pattern`, `incorrect_warning`, `incorrect_seed` parameters. Branches knowledge assignment to use mixed when pattern is set. Logs all incorrect knowledge details.
- **`run_experiment.py`**: Added `--incorrect-pattern`, `--incorrect-warning`, `--incorrect-seed` CLI args.

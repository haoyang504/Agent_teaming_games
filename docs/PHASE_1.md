# Phase 1 — Foundation: Knowledge Assignment System + Setting Parameter

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first for full experimental design context.
> **Goal:** Replace the existing knowledge assignment system and add the `--setting` CLI parameter. After this phase, the pipeline should run end-to-end with the new knowledge configurations.

---

## Task 1A: Rework `knowledge_manager.py` — DONE

### What existed before Phase 1

The old system used string-based levels (`"none"`, `"quarter"`, `"half"`, `"full"`) mapping to fixed item counts (0, 4, 8, 15). All agents drew from the front of the same shuffled list, so a "quarter" agent's items were always a subset of a "half" agent's items. There was no source pool selection and no overlap control.

### What to build

Replace `generate_knowledge_assignment()` with a new function that accepts three parameters:

```python
def generate_knowledge_assignment(
    counts: List[int],          # e.g., [0, 2, 4] for Config 1
    source: str = "all",        # "all" | "top" | "bottom"
    overlap: str = "nested",    # "nested" | "disjoint" | "O3" | "O4"
    seed: int = 42,
) -> List[List[Tuple[str, int, str]]]:
```

**Parameter details:**

**`counts`** — A list of 3 integers specifying how many items each agent knows. The list is ordered [Agent A, Agent B, Agent C]. Valid configurations:
- `[0, 2, 4]` — Config 1 (high diversity)
- `[1, 2, 3]` — Config 2 (medium diversity)
- `[2, 2, 2]` — Config 3 (low/equal diversity)

**`source`** — Which portion of the 15-item ranking to draw knowledge from:
- `"all"` — items ranked 1–15 (full pool)
- `"top"` — items ranked 1–8 only
- `"bottom"` — items ranked 8–15 only

Implementation: filter `ITEMS` / `GROUND_TRUTH_RANKS` / `EXPLANATIONS` to only include items within the source range, then sample from this filtered pool.

**`overlap`** — How knowledge sets relate across agents. **Agent C is always sampled first** (index 2 in the counts list), then B, then A:

- **`"nested"` (O1):** Sample C's items from the source pool. B's items are a random subset of C's items. A's items are a random subset of B's items. Result: A ⊆ B ⊆ C.
- **`"disjoint"` (O2):** Sample C's items from the source pool. Sample B's items from the remaining pool (excluding C's items). Sample A's items from what's left (excluding B's and C's items). Result: no overlap between any agents.
- **`"O3"`:** Sample C's items. Sample B's items allowing overlap with C (sample from full source pool). Sample A's items only from items NOT in C's set.
- **`"O4"`:** Sample C's items. Sample B's items only from items NOT in C's set. Sample A's items allowing overlap with C (sample from full source pool).

**Return value:** Same as current — a list of 3 lists, each containing `(item_name, ground_truth_rank, explanation)` tuples. Agent at index 0 = Agent A, index 1 = Agent B, index 2 = Agent C.

### Additional requirements

- Keep `format_knowledge_for_prompt()` and `knowledge_level_label()` working. Update `knowledge_level_label()` to accept an integer count instead of a string level, e.g., `knowledge_level_label(4)` returns `"4/15 items"`.
- Add a `format_knowledge_counts()` helper that takes the counts list and returns a human-readable summary for logging, e.g., `"Agent A: 0, Agent B: 2, Agent C: 4"`.
- Use the `seed` parameter for all randomness (via `random.Random(seed)`) to ensure reproducibility.
- Raise `ValueError` if a disjoint configuration is impossible (e.g., requesting [5, 5, 5] from a source pool of only 8 items with disjoint overlap).

---

## Task 1B: Update `run_experiment.py` — DONE

### What existed before Phase 1

The file defined a `STRATEGIES` dictionary with string-based strategy names mapping to lists of level strings (e.g., `"ascending": ["none", "quarter", "half"]`). The `--strategy` CLI argument accepted a `+`-separated string of 3 levels.

### What to build

Replace the `STRATEGIES` dictionary with a `CONFIGS` dictionary that uses the new three-dimensional system:

```python
CONFIGS = {
    "high_div": {
        "counts": [0, 2, 4],
        "source": "all",
        "overlap": "nested",
    },
    "med_div": {
        "counts": [1, 2, 3],
        "source": "all",
        "overlap": "nested",
    },
    "low_div": {
        "counts": [2, 2, 2],
        "source": "all",
        "overlap": "nested",
    },
    # Add more as needed for the full 12-condition sweep
}
```

Update CLI arguments:
- Replace `--strategy` with `--config` (accepts a config name from `CONFIGS`, or a custom spec)
- Add `--source` (default `"all"`, values: `all`, `top`, `bottom`) to override the config's source
- Add `--overlap` (default `"nested"`, values: `nested`, `disjoint`, `O3`, `O4`) to override the config's overlap
- Add `--counts` (e.g., `"0,2,4"`) for fully custom runs
- Add `--setting` (integer 1–4, default 1) — this doesn't do anything yet in Phase 1 except get passed through and logged. It will be wired up in Phase 2.

Update the call to `run_experiment()` in `experiment_runner.py` to pass the new parameters instead of `knowledge_strategy`.

---

## Task 1C: Update `experiment_runner.py` — DONE

### What was changed

Update `run_experiment()` to:
1. Accept the new parameters (`counts`, `source`, `overlap`, `setting`) instead of `knowledge_strategy`.
2. Call the new `generate_knowledge_assignment(counts, source, overlap, seed)`.
3. Pass `setting` through to agent construction (but don't act on it yet — that's Phase 2).
4. Update the experiment ID, logging, and JSON output to reflect the new parameter names.
5. Update `agent_knowledge_log` to use integer counts instead of string levels.

The `MoonSurvivalAgent` constructor call should still work — each agent still receives its list of knowledge tuples. The agent doesn't need to know about the configuration system, only about its own knowledge.

---

## Tests

After implementing, run these verification steps:

### Test 1: Knowledge assignment unit test

Create a file `test_knowledge.py`:

```python
from knowledge_manager import generate_knowledge_assignment

# Test all 12 core configurations
configs = [
    ([0, 2, 4], "all", "nested"),
    ([0, 2, 4], "all", "disjoint"),
    ([0, 2, 4], "all", "O3"),
    ([0, 2, 4], "all", "O4"),
    ([1, 2, 3], "all", "nested"),
    ([1, 2, 3], "all", "disjoint"),
    ([1, 2, 3], "all", "O3"),
    ([1, 2, 3], "all", "O4"),
    ([2, 2, 2], "all", "nested"),
    ([2, 2, 2], "all", "disjoint"),
    ([2, 2, 2], "all", "O3"),
    ([2, 2, 2], "all", "O4"),
]

for counts, source, overlap in configs:
    assignments = generate_knowledge_assignment(counts, source, overlap, seed=42)
    
    # Check counts
    for i, (count, assignment) in enumerate(zip(counts, assignments)):
        assert len(assignment) == count, f"FAIL {counts}/{overlap}: Agent {i} expected {count} items, got {len(assignment)}"
    
    items_a = {item for item, _, _ in assignments[0]}
    items_b = {item for item, _, _ in assignments[1]}
    items_c = {item for item, _, _ in assignments[2]}
    
    # Check overlap patterns
    if overlap == "nested":
        assert items_a <= items_b, f"FAIL nested: A not subset of B"
        assert items_b <= items_c, f"FAIL nested: B not subset of C"
    elif overlap == "disjoint":
        assert items_a & items_b == set(), f"FAIL disjoint: A and B overlap"
        assert items_a & items_c == set(), f"FAIL disjoint: A and C overlap"
        assert items_b & items_c == set(), f"FAIL disjoint: B and C overlap"
    elif overlap == "O3":
        assert items_a & items_c == set(), f"FAIL O3: A and C should not overlap"
    elif overlap == "O4":
        assert items_b & items_c == set(), f"FAIL O4: B and C should not overlap"
    
    print(f"PASS: counts={counts}, source={source}, overlap={overlap}")
    for i, label in enumerate(["A", "B", "C"]):
        item_names = [item for item, _, _ in assignments[i]]
        print(f"  Agent {label}: {item_names}")
    print()

# Test source filtering
for source in ["all", "top", "bottom"]:
    assignments = generate_knowledge_assignment([2, 2, 4], "all" if source == "all" else source, "disjoint", seed=42)
    for i, assignment in enumerate(assignments):
        for item, rank, _ in assignment:
            if source == "top":
                assert rank <= 8, f"FAIL source=top: item '{item}' has rank {rank}"
            elif source == "bottom":
                assert rank >= 8, f"FAIL source=bottom: item '{item}' has rank {rank}"
    print(f"PASS: source={source} filtering correct")

print("\nAll tests passed!")
```

Run: `python test_knowledge.py`

### Test 2: End-to-end pipeline test

Run a single experiment to verify the full pipeline still works:

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1
```

Expected: completes without errors, produces a JSON log in `results/`, the log contains the correct knowledge assignments matching Config 1 with nested overlap.

### Test 3: CLI parameter test

Verify custom parameters work:

```bash
python run_experiment.py --counts 1,2,3 --source top --overlap disjoint --setting 1 --k 1 --iterations 1
```

Expected: completes without errors. In the JSON log, all knowledge items should have ranks 1–8 (source=top), and no items should overlap between agents (overlap=disjoint).

---

## What NOT to do in Phase 1

- Do NOT modify `agent.py`'s `_system_prompt()` — that's Phase 2.
- Do NOT implement team knowledge info or leader role injection — that's Phase 2.
- Do NOT change discussion order randomization — that's Phase 3.
- Do NOT implement feedback frequency/mode logic — that's Phase 4.
- Do NOT implement incorrect knowledge — that's Phase 5.
- Do NOT modify `moon_survival_env.py` at all.
- The `--setting` parameter should be accepted, logged, and passed through, but should have no effect on behavior in Phase 1.

---

## Implementation Notes (post-completion)

### O3/O4 guaranteed overlap

The original spec said O3 = "B overlaps C" and O4 = "A overlaps C". The initial implementation sampled from the full pool, which only *allowed* overlap by chance. Testing showed:

- O3 with Config 1 `[0,2,4]`: B had **zero** overlap with C in 46/100 seeds
- O4 with Config 2 `[1,2,3]`: A had zero overlap with C in 80/100 seeds

This made O3/O4 statistically indistinguishable from disjoint in many runs. Fix: `_sample_with_guaranteed_overlap()` picks 1 item from the overlap source first, then fills the rest from the full pool. Now 0/100 seeds have empty overlap.

### Config 3 + nested = identical knowledge

Config 3 `[2,2,2]` with nested overlap produces A = B = C (all agents get the same 2 items). This is mathematically forced: A ⊆ B ⊆ C with equal counts means all sets are identical. This is the intended "Low (equal) diversity" behavior per CONTEXT.md.

### Additional files updated

- **`verify_plumbing.py`** — Rewritten with 36 checks covering the new API (counts, overlaps, source filtering, helpers, reproducibility). No API calls.
- **`run_all_combinations.py`** — Rewritten to sweep the 12 core conditions (3 configs x 4 overlaps). Added `--include-source-variants` for 36-condition sweep, `--setting` parameter.
- **`test_knowledge.py`** — New file with unit tests for all 12 core configurations + source filtering.

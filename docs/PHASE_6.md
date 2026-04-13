# Phase 6 — Analysis Metrics: Exploration + Agent Dominance

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first. Phases 1–5 must be complete. You need saved JSON experiment logs in the `results/` directory to test against.
> **Goal:** Create a standalone `metrics.py` module that reads experiment JSON logs and computes two sets of metrics: exploration (novelty + recombination) and agent dominance. This module does NOT modify any existing files — it's pure analysis code.

---

## What exists now

Experiment JSON logs contain everything needed for analysis:
- `iterations[].proposals[].candidates[]` — each agent's proposed rankings per iteration (list of {item_name: rank} dicts)
- `iterations[].final_selection.candidates[]` — the final selected rankings per iteration
- `iterations[].final_selection.agent_id` — which agent made the final selection
- `iterations[].proposals[].agent_id` — which agent proposed each candidate

There is no analysis code. SAD scores are computed during the experiment but no deeper analysis exists.

---

## Key Concepts

### Item-Rank Pair

An item-rank pair is a tuple `(item_name, rank)` — e.g., `("Two 100-lb oxygen tanks", 1)`. A candidate ranking of 15 items contains exactly 15 item-rank pairs. These are the atomic units for both metrics.

### Extracting pairs from a candidate dict

A candidate is stored as `{item_name: rank}`. To extract pairs:

```python
def extract_pairs(candidate: Dict[str, int]) -> Set[Tuple[str, int]]:
    """Extract the set of (item, rank) pairs from a candidate ranking."""
    return {(item, rank) for item, rank in candidate.items()}
```

---

## Task 6A: Create `metrics.py` with exploration metrics — DONE

### File location

Create `code/metrics.py` as a new standalone module.

### Function 1: `compute_novelty()`

```python
def compute_novelty(experiment_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute novelty for each iteration's final ranking(s).

    Novelty = number of item-rank pairs in this iteration's final output
    that have NEVER appeared in any prior iteration's final output.

    Args:
        experiment_log: Parsed JSON experiment log.

    Returns:
        List of dicts, one per iteration:
        {
            "iteration": int,
            "candidate_index": int,
            "total_pairs": 15,
            "novel_pairs": int,       # pairs not seen in any prior iteration
            "novelty_ratio": float,   # novel_pairs / total_pairs
        }
    """
```

**Algorithm:**
1. Maintain a running set `seen_pairs` that accumulates all item-rank pairs from final outputs across iterations.
2. For each iteration, for each final candidate:
   - Extract its 15 item-rank pairs
   - Count how many are NOT in `seen_pairs` → that's the novelty count
   - Add all pairs from this iteration's final candidates to `seen_pairs`
3. Iteration 1 should have novelty_ratio = 1.0 (everything is new since there's no prior history).

### Function 2: `compute_recombination()`

```python
def compute_recombination(experiment_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute recombination for each iteration's final ranking(s).

    Recombination measures whether the final ranking draws item-rank pairs
    from multiple agents' proposals rather than from a single agent.

    Args:
        experiment_log: Parsed JSON experiment log.

    Returns:
        List of dicts, one per iteration, one entry per final candidate:
        {
            "iteration": int,
            "candidate_index": int,
            "source_agents": List[int],       # agents whose proposals contributed pairs
            "pairs_per_agent": Dict[int, int], # agent_id -> count of pairs from that agent
            "max_agent_contribution": int,     # most pairs from any single agent
            "recombination_score": float,      # 1 - (max_contribution / 15)
            "is_recombination": bool,          # True if pairs come from 2+ agents
        }
    """
```

**Algorithm:**
1. For each iteration, collect all proposals from all agents. Each proposal is a candidate dict. An agent may have proposed k candidates — collect all item-rank pairs from all of their candidates into a per-agent set.
2. For each final candidate:
   - Extract its 15 item-rank pairs
   - For each pair, check which agent(s) proposed it. If multiple agents proposed the same pair, attribute it to the agent who proposed it first (lowest agent_id).
   - If a pair wasn't proposed by any agent (the final selector created it), attribute it to the selecting agent.
   - Count how many pairs came from each agent.
3. `recombination_score`: 1 - (max single-agent contribution / 15). Score of 0 = entire ranking from one agent. Score near 1 = evenly spread across agents.
4. `is_recombination`: True if pairs come from 2+ distinct agents.

---

## Task 6B: Add agent dominance metrics — DONE

### Function 3: `compute_dominance()`

```python
def compute_dominance(experiment_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute agent dominance for each iteration.

    Dominance traces each item-rank pair in the final output back to the
    agent who FIRST proposed that exact pair.

    Args:
        experiment_log: Parsed JSON experiment log.

    Returns:
        List of dicts, one per iteration, one entry per final candidate:
        {
            "iteration": int,
            "candidate_index": int,
            "dominance": Dict[int, int],  # agent_id -> number of pairs they contributed
            "most_dominant": int,          # agent_id with most contributed pairs
            "least_dominant": int,         # agent_id with fewest contributed pairs
            "unattributed": int,           # pairs not found in any agent's proposals
        }
    """
```

**Algorithm:**
1. For each iteration, build a mapping from item-rank pairs to the agent who first proposed them:
   - Iterate through proposals in agent_id order (1, 2, 3)
   - For each agent, for each of their k candidates, extract all item-rank pairs
   - For each pair, if it's not already in the mapping, assign it to this agent
2. For each final candidate:
   - Extract its 15 item-rank pairs
   - Look up each pair in the mapping → attribute to the proposing agent
   - Pairs not found in any proposal are "unattributed" (the final selector invented them)
3. Count pairs per agent. The agent with the most = most dominant. The agent with the fewest = least dominant.

**Important:** The `most_dominant` and `least_dominant` fields should only consider agents that are in the experiment (all 3 agents). If an agent contributed 0 pairs, they're still the least dominant — don't skip them.

---

## Task 6C: Add summary and CLI interface — DONE

### Function 4: `analyze_experiment()`

```python
def analyze_experiment(experiment_log: Dict[str, Any]) -> Dict[str, Any]:
    """Run all metrics on an experiment log and return a combined summary.

    Returns:
        {
            "experiment_id": str,
            "novelty": [...],        # from compute_novelty
            "recombination": [...],  # from compute_recombination
            "dominance": [...],      # from compute_dominance
            "summary": {
                "mean_novelty_ratio": float,
                "mean_recombination_score": float,
                "overall_dominance": Dict[int, int],  # total pairs attributed per agent across all iterations
                "most_dominant_agent": int,
            }
        }
    """
```

### CLI interface

Add a `if __name__ == "__main__"` block so `metrics.py` can be run directly:

```python
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze Moon Survival experiment logs.")
    parser.add_argument("log_file", help="Path to experiment JSON log file.")
    parser.add_argument("--output", help="Path to save analysis JSON (optional).")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results.")
    args = parser.parse_args()

    with open(args.log_file) as f:
        experiment_log = json.load(f)

    results = analyze_experiment(experiment_log)

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"  ANALYSIS: {results['experiment_id']}")
        print(f"{'='*60}")

        print(f"\n  --- Novelty ---")
        for entry in results["novelty"]:
            print(f"  Iter {entry['iteration']} Cand {entry['candidate_index']}: "
                  f"{entry['novel_pairs']}/15 novel (ratio={entry['novelty_ratio']:.2f})")

        print(f"\n  --- Recombination ---")
        for entry in results["recombination"]:
            agents = entry["source_agents"]
            print(f"  Iter {entry['iteration']} Cand {entry['candidate_index']}: "
                  f"from {len(agents)} agents, score={entry['recombination_score']:.2f}")

        print(f"\n  --- Dominance ---")
        for entry in results["dominance"]:
            print(f"  Iter {entry['iteration']} Cand {entry['candidate_index']}: "
                  f"{entry['dominance']}, most={entry['most_dominant']}, "
                  f"least={entry['least_dominant']}, unattributed={entry['unattributed']}")

        print(f"\n  --- Summary ---")
        s = results["summary"]
        print(f"  Mean novelty ratio: {s['mean_novelty_ratio']:.2f}")
        print(f"  Mean recombination score: {s['mean_recombination_score']:.2f}")
        print(f"  Overall dominance: {s['overall_dominance']}")
        print(f"  Most dominant agent: {s['most_dominant_agent']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Analysis saved: {args.output}")
```

### Batch analysis helper

Add a function that processes all JSON files in a directory:

```python
def analyze_batch(results_dir: str) -> List[Dict[str, Any]]:
    """Analyze all experiment logs in a directory.

    Args:
        results_dir: Path to directory containing experiment JSON files.

    Returns:
        List of analysis results, one per experiment.
    """
    import glob
    analyses = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        # Skip summary files and analysis files
        if path.endswith("_summary.json") or path.endswith("_analysis.json"):
            continue
        with open(path) as f:
            log = json.load(f)
        # Skip if it's not an experiment log (no iterations key)
        if "iterations" not in log:
            continue
        analyses.append(analyze_experiment(log))
    return analyses
```

---

## Tests

### Test 1: Novelty on existing logs

Run against one of the experiment logs from the previous batch:

```bash
cd code
python metrics.py ../results/c0-2-4__all__nested__s1__k3__iter3__disc1__*.json --verbose
```

Verify:
- Iteration 1 has high novelty (close to 1.0 for at least the first candidate)
- Later iterations should show declining novelty if the team is converging
- All entries have `total_pairs` = 15

### Test 2: Recombination

In the same output, check recombination entries:
- `source_agents` should contain agent IDs that actually proposed the pairs
- `is_recombination` should be True when pairs come from 2+ agents
- `recombination_score` should be between 0 and 1

### Test 3: Dominance

Check dominance entries:
- `dominance` dict should have keys for all 3 agents (1, 2, 3)
- Sum of dominance values + unattributed should equal 15 for each candidate
- `most_dominant` and `least_dominant` should be valid agent IDs

### Test 4: Synthetic verification

Create a quick script that constructs a known experiment log and verifies exact metric values:

```python
from metrics import compute_novelty, compute_dominance

# Create a minimal synthetic log
synthetic = {
    "experiment_id": "test",
    "iterations": [
        {
            "iteration": 1,
            "proposals": [
                {
                    "agent_id": 1,
                    "candidates": [
                        {"A": 1, "B": 2, "C": 3}  # 3 items for simplicity
                    ]
                },
                {
                    "agent_id": 2,
                    "candidates": [
                        {"A": 2, "B": 1, "C": 3}
                    ]
                },
                {
                    "agent_id": 3,
                    "candidates": [
                        {"A": 1, "B": 3, "C": 2}
                    ]
                },
            ],
            "final_selection": {
                "agent_id": 3,
                "candidates": [
                    {"A": 1, "B": 2, "C": 3}  # matches Agent 1's proposal exactly
                ]
            },
            "final_scores": [0],
        }
    ]
}

novelty = compute_novelty(synthetic)
assert novelty[0]["novel_pairs"] == 3, "Iteration 1: all 3 pairs should be novel"
assert novelty[0]["novelty_ratio"] == 1.0, "Iteration 1: novelty ratio should be 1.0"

dominance = compute_dominance(synthetic)
# Final ranking matches Agent 1 exactly: A=1, B=2, C=3
# Agent 1 proposed (A,1), (B,2), (C,3) — all 3 match
assert dominance[0]["dominance"][1] == 3, "Agent 1 should dominate with all 3 pairs"
assert dominance[0]["most_dominant"] == 1, "Agent 1 should be most dominant"

print("PASS: synthetic verification")
```

### Test 5: Batch analysis

```bash
python -c "
from metrics import analyze_batch
results = analyze_batch('../results')
print(f'Analyzed {len(results)} experiments')
for r in results[:3]:
    s = r['summary']
    print(f\"  {r['experiment_id'][:40]}... novelty={s['mean_novelty_ratio']:.2f} recombo={s['mean_recombination_score']:.2f} dominant=Agent {s['most_dominant_agent']}\")
"
```

Verify it processes multiple logs without errors and produces reasonable summaries.

---

## What NOT to do in Phase 6

- Do NOT modify any existing files (`agent.py`, `experiment_runner.py`, `knowledge_manager.py`, `run_experiment.py`, `moon_survival_env.py`).
- Do NOT re-run experiments — this phase only analyzes existing JSON logs.
- Do NOT import from `agent.py` or `experiment_runner.py` — `metrics.py` should only depend on standard library (`json`, `os`, `glob`, `argparse`) and optionally `moon_survival_env.py` for constants.
- Do NOT modify the experiment JSON format — read it as-is.
- Keep metric calculations deterministic — no randomness needed.
- The module should handle experiments with any value of k (1, 2, 3, etc.) and any number of iterations gracefully.

---

## Implementation Notes (post-completion)

### Verified against real data
- Batch analysis processed all 26 experiment logs in `results/` without errors
- Novelty declines over iterations as expected (1.0 → ~0.13–0.53)
- Recombination scores range 0–0.67, reflecting varying levels of cross-agent integration
- Dominance sum + unattributed = total_pairs (15) for all candidates

### Synthetic tests
- Single-iteration: all pairs novel (ratio=1.0), single-agent dominance correctly detected
- Two-iteration with identical output: second iteration has 0 novel pairs
- Mixed-source recombination: 3-agent attribution with score = 0.67

### No existing files modified
`metrics.py` is fully standalone — depends only on standard library (`json`, `os`, `glob`, `argparse`). No imports from `agent.py`, `experiment_runner.py`, or other project modules.

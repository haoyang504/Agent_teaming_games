# Moon Survival Agent Teaming Experiment

Three LLM agents collaborate to rank 15 survival items for a 200-mile lunar trek (NASA Moon Survival task). Rankings are scored against the NASA expert answer using Sum of Absolute Differences (SAD). The experiment tests how **information asymmetry**, **hierarchy**, and **incorrect knowledge** affect group decision-making.

## Setup

Install dependencies:

```bash
pip install portkey-ai
```

The codebase uses [Portkey](https://portkey.ai/) as the LLM gateway. API credentials are configured in `code/config.py`.

## Running Experiments

All scripts are in the `code/` directory.

### Single Experiment

```bash
cd code

# Basic run with a named config
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 3

# Custom knowledge counts and overlap
python run_experiment.py --counts 0,2,4 --overlap disjoint --setting 2

# Quick sanity check (1 iteration, 1 candidate, 1 discussion round)
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1

# Test discussion order effect (ABC primary, CBA secondary)
python run_experiment.py --config high_div --setting 1 --discussion-order ABC
python run_experiment.py --config high_div --setting 1 --discussion-order CBA

# Incorrect-knowledge run (Agent C's items all replaced with wrong ones)
python run_experiment.py --config high_div --setting 4 --incorrect-pattern I3 --incorrect-warning
```

### Full Configuration Sweep

```bash
cd code

# 7 unique conditions (post-Phase-8 collapse from 2 configs × 5 overlaps)
python run_all_combinations.py --dry-run   # preview what will run
python run_all_combinations.py             # execute
```

### Analysis

```bash
cd code

# Analyze a single experiment log
python metrics.py ../results/<experiment>.json --verbose

# Batch analysis of all logs
python -c "from metrics import analyze_batch; [print(r['summary']) for r in analyze_batch('../results')]"
```

## CLI Arguments

### `run_experiment.py`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `high_div` | Named config: `high_div` (`[0,2,4]`) or `low_div` (`[2,2,2]`) |
| `--counts` | `str` | — | Custom item counts for [A,B,C], e.g. `"0,2,4"` |
| `--overlap` | `str` | `nested` | Overlap pattern: `nested`, `disjoint`, `O3`, `O4`, `O5` |
| `--setting` | `int` | `1` | Experimental setting 1–4 (controls team info + leader role) |
| `--feedback-mode` | `str` | `F1` | Feedback mode: `F1`, `F2`, or `F3` |
| `--discussion-order` | `str` | `ABC` | Speaking order: `ABC` (1→2→3) or `CBA` (3→2→1) |
| `--k` | `int` | `3` | Candidate rankings per iteration |
| `--iterations` | `int` | `3` | Number of iterations (full protocol: 20) |
| `--discussion-rounds` | `int` | `3` | Discussion rounds per iteration |
| `--model` | `str` | `gpt-5-mini` | Model name (routed via Portkey) |
| `--seed` | `int` | `42` | Random seed for knowledge assignment |
| `--incorrect-pattern` | `str` | — | Incorrect knowledge: `I1`, `I2`, `I3`, `I4` |
| `--incorrect-warning` | flag | `false` | Warn agents that some knowledge may be incorrect |
| `--incorrect-seed` | `int` | `99` | Fallback seed when `incorrect_ranking.json` is missing |
| `--output-dir` | `str` | `../results` | Output directory for JSON logs and CSV summaries |

## Experimental Design

### Knowledge Configurations

| Config | Agent A | Agent B | Agent C | Diversity |
|--------|---------|---------|---------|-----------|
| high_div | 0 items | 2 items | 4 items | High |
| low_div | 2 items | 2 items | 2 items | Low (equal) |

### Overlap Patterns

Agent C is always sampled first; B and A relative to C.

| Pattern | Constraint |
|---------|------------|
| `nested` | A ⊆ B ⊆ C |
| `disjoint` | A, B, C pairwise disjoint |
| `O3` | B ⊆ C; A disjoint from both B and C |
| `O4` | A ⊆ C; B disjoint from both A and C |
| `O5` | A = B (smaller subset of larger when counts differ); C disjoint from both |

The 2 × 5 product is 10 conditions, but several collapse (when A has 0 items, several overlaps reduce to the same structure). The sweep runs **7 unique conditions** — see `docs/PHASE_8.md` Task 8H for the collapse table.

### Settings (Information Conditions)

| Setting | Team Info | Role Info |
|---------|-----------|-----------|
| 1 | No | No |
| 2 | Yes | No |
| 3 | No | Yes |
| 4 | Yes | Yes |

- **Team info** = each agent told how many items the other agents know (neutral wording)
- **Role info** = all agents told Agent C is the designated leader, *and* "The leader is responsible for guiding the discussion."
- **Consensus framing**: every setting's system prompt includes `"The agents aim to reach a consensus through N rounds of discussion."` Mechanically, Agent C selects the final candidates after discussion in all settings — see `docs/CONTEXT.md` for the structural caveat.

### Discussion Order

| Order | Behavior |
|-------|----------|
| `ABC` (default) | Fixed: Agent A → B → C every round |
| `CBA` | Fixed: Agent C → B → A every round |

### Feedback Modes

| Mode | Description |
|------|-------------|
| F1 | SAD scores from last iteration, every iteration |
| F2 | SAD scores from last iteration, every 2 iterations |
| F3 | SAD scores from last iteration, every 4 iterations |

### Incorrect Knowledge Patterns

Incorrect items **replace** correct items at the same per-agent count (the total never grows). The incorrect ranking itself is fixed and persisted to `code/incorrect_ranking.json`.

| Pattern | Behavior |
|---------|----------|
| `I1` | Agent A's items are all replaced with incorrect ones |
| `I2` | Agent B's items are all replaced |
| `I3` | Agent C's items are all replaced |
| `I4` | Each agent has half-correct + half-incorrect (rounded up on incorrect side) |

For `[0,2,4]`/I1, A has 0 items so the run is structurally equivalent to no-incorrect baseline.

## Outputs

Experiment results are saved to `results/` as JSON logs and CSV summaries. Each experiment ID encodes the full configuration, e.g.:

```
c0-2-4__nested__s1__fbF1__doABC__k3__iter20__disc3__20260512_180944
```

Use `code/metrics.py` to compute novelty, recombination, dominance, contribution share %, and per-agent proposal SAD from the JSON logs.

## Codebase Structure

```
code/
  run_experiment.py           CLI entry point, named configs
  experiment_runner.py        Orchestration, iteration loop, feedback logic
  agent.py                    LLM agent class, system prompt construction
  knowledge_manager.py        Knowledge assignment (counts/overlap)
  moon_survival_env.py        Items, ground truth, scoring, parsing (DO NOT MODIFY)
  config.py                   API keys and model configuration
  config.example.py           Template for config.py (safe to commit)
  metrics.py                  Post-hoc analysis (novelty, recombination, dominance, contribution share)
  run_all_combinations.py     Phase-8 unique-conditions sweep (7 runs)
  aggregate_iterations.py     Utility for aggregating results across iterations
  verify_plumbing.py          Offline sanity checks (no API calls)
  test_knowledge.py           Knowledge assignment unit tests
  incorrect_ranking.json      Persisted fixed incorrect ranking (Phase 8)
```

See `docs/CONTEXT.md` for the experimental design, `docs/PHASE_1.md`–`docs/PHASE_8.md` for implementation details, and `docs/CHANGELOG.md` for a file-by-file list of every code change.

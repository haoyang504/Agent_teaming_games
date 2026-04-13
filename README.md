# Moon Survival Agent Teaming Experiment

Three LLM agents collaborate to rank 15 survival items for a 200-mile lunar trek (NASA Moon Survival task). Rankings are scored against the NASA expert answer using Sum of Absolute Differences (SAD). The experiment tests how **information asymmetry**, **hierarchy**, and **discussion order** affect group decision-making.

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

# Custom knowledge counts
python run_experiment.py --counts 1,2,3 --source top --overlap disjoint --setting 2

# Quick sanity check (1 iteration, 1 candidate, 1 discussion round)
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1

# Test discussion order effect
python run_experiment.py --config high_div --setting 1 --discussion-order ABC
python run_experiment.py --config high_div --setting 1 --discussion-order CBA
python run_experiment.py --config high_div --setting 1 --discussion-order random
```

### Full Configuration Sweep

```bash
cd code

# 12 core conditions (3 configs × 4 overlaps)
python run_all_combinations.py --dry-run   # preview what will run
python run_all_combinations.py             # execute

# Include source variants (36 conditions)
python run_all_combinations.py --include-source-variants
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
| `--config` | `str` | `high_div` | Named config (e.g. `high_div`, `med_div_disjoint`, `low_div_O3`) |
| `--counts` | `str` | — | Custom item counts for [A,B,C], e.g. `"0,2,4"` |
| `--source` | `str` | `all` | Source pool: `all`, `top` (ranks 1–8), `bottom` (ranks 8–15) |
| `--overlap` | `str` | `nested` | Overlap pattern: `nested`, `disjoint`, `O3`, `O4` |
| `--setting` | `int` | `1` | Experimental setting 1–4 (controls team info + leader role) |
| `--feedback-mode` | `str` | `F1` | Feedback mode: `F1`–`F5` |
| `--discussion-order` | `str` | `random` | Speaking order: `random`, `ABC` (1→2→3), `CBA` (3→2→1) |
| `--k` | `int` | `3` | Candidate rankings per iteration |
| `--iterations` | `int` | `3` | Number of iterations (full protocol: 20) |
| `--discussion-rounds` | `int` | `3` | Discussion rounds per iteration |
| `--model` | `str` | `gpt-5-mini` | Model name (routed via Portkey) |
| `--seed` | `int` | `42` | Random seed for knowledge assignment |
| `--incorrect-pattern` | `str` | — | Incorrect knowledge: `I1`, `I2`, `I3`, `I4` |
| `--incorrect-warning` | flag | `false` | Warn agents that some knowledge may be incorrect |
| `--output-dir` | `str` | `../results` | Output directory for JSON logs and CSV summaries |

## Experimental Design

### Knowledge Configurations

| Config | Agent A | Agent B | Agent C | Diversity |
|--------|---------|---------|---------|-----------|
| 1 | 0 items | 2 items | 4 items | High |
| 2 | 1 item | 2 items | 3 items | Medium |
| 3 | 2 items | 2 items | 2 items | Low (equal) |

### Settings (Information Conditions)

| Setting | Team Info | Role Info |
|---------|-----------|-----------|
| 1 | No | No |
| 2 | Yes | No |
| 3 | No | Yes |
| 4 | Yes | Yes |

- **Team info** = each agent told how many items the other agents know (neutral wording)
- **Role info** = all agents told Agent C is the designated leader (no behavioral instructions)

### Discussion Order

| Order | Behavior |
|-------|----------|
| `random` | Shuffled each round (reproducible via seed) |
| `ABC` | Fixed: Agent A → B → C every round |
| `CBA` | Fixed: Agent C → B → A every round |

### Feedback Modes

| Mode | Description |
|------|-------------|
| F1 | SAD scores from last iteration, every iteration |
| F2 | SAD scores from last iteration, every 2 iterations |
| F3 | SAD scores from last iteration, every 4 iterations |
| F4 | Full history of all candidates across all prior iterations |
| F5 | Top 8 candidates (by SAD) across all prior iterations |

## Outputs

Experiment results are saved to `results/` as JSON logs and CSV summaries. Each experiment ID encodes the full configuration, e.g.:

```
c0-2-4__all__nested__s1__fbF1__doABC__k3__iter20__disc3__20260413_100029
```

Use `code/metrics.py` to compute novelty, recombination, and dominance metrics from the JSON logs.

## Codebase Structure

```
code/
  run_experiment.py           CLI entry point, named configs
  experiment_runner.py        Orchestration, iteration loop, feedback logic
  agent.py                    LLM agent class, system prompt construction
  knowledge_manager.py        Knowledge assignment (counts/source/overlap)
  moon_survival_env.py        Items, ground truth, scoring, parsing (DO NOT MODIFY)
  config.py                   API keys and model configuration
  config.example.py           Template for config.py (safe to commit)
  metrics.py                  Post-hoc analysis (novelty, recombination, dominance)
  run_all_combinations.py     Full configuration sweep
  aggregate_iterations.py     Utility for aggregating results across iterations
  verify_plumbing.py          Offline sanity checks (no API calls)
  test_knowledge.py           Knowledge assignment unit tests
```

See `docs/CONTEXT.md` for the full experimental design, `docs/PHASE_1.md`–`docs/PHASE_7.md` for implementation details, and `docs/CHANGELOG.md` for a file-by-file list of every code change.

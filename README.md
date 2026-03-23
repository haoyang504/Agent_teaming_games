# Moon Survival Agent Teaming Experiment

This repository contains the codebase for running LLM agent teaming experiments on a "Moon Survival" task. The system allows you to simulate collaboration among multiple AI agents, each initialized with varying levels of knowledge ("strategies"), to collectively solve complex problems.

## Getting Started

Before running any experiments, ensure you have set your OpenAI API key as an environment variable, as the agents rely on it to generate responses:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Code

All execution scripts are located within the `code/` directory. There are two primary ways to run the experiments depending on your needs.

### 1. Specific or Predefined Strategies (`code/run_experiment.py`)

This script is ideal for running a single strategy or a few predefined ones for quick testing and small-scale experiments.

* **Run all predefined strategies:**
  By default (no flags), the script will run 4 predefined strategies: `ascending` (none, quarter, half), `descending` (half, half, none), `all_half`, and `all_quarter`.
  ```bash
  cd code
  python run_experiment.py
  ```

* **Run a specific custom strategy:**
  Use the `--strategy` flag by providing a `+` separated list of the 3 knowledge levels (e.g., `none`, `quarter`, `half`, `full`).
  ```bash
  cd code
  python run_experiment.py --strategy none+quarter+half --k 3 --iterations 3
  ```

* **Quick Sanity Check (1 iteration, 1 candidate):**
  Useful for verifying that the plumbing and API keys are working correctly.
  ```bash
  cd code
  python run_experiment.py --strategy none+quarter+half --k 1 --iterations 1
  ```

### 2. Full Strategy Sweep (`code/run_all_combinations.py`)

This script runs an exhaustive sweep of all 27 possible knowledge combinations for a team of 3 agents (where each agent can have `none`, `quarter`, or `half` knowledge). It generates a comprehensive summary CSV file and individual JSON logs in the `results/` folder.

* **Run the full 27 combinatorics sweep:**
  ```bash
  cd code
  python run_all_combinations.py
  ```

* **Skip the zero-knowledge baseline:**
  Ignores the `[none, none, none]` trial to save time and API costs.
  ```bash
  cd code
  python run_all_combinations.py --skip-all-none
  ```

* **Dry Run Verification:**
  To see exactly what combinations will execute without actually making LLM calls:
  ```bash
  cd code
  python run_all_combinations.py --dry-run
  ```

## Customization Options

Both `run_experiment.py` and `run_all_combinations.py` accept several arguments to customize the experiment constraints:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--k` | `int` | `3` | Number of candidate rankings to be generated per iteration. |
| `--iterations` | `int` | `3` | Total number of macro-iterations or large-scale rounds. |
| `--discussion-rounds` | `int` | `1` | Discussion rounds per iteration where agents speak sequentially. |
| `--model` | `str` | `"gpt-4o-mini"` | The OpenAI model utilized by the agents. |
| `--seed` | `int` | `42` | Random seed for assigning internal knowledge combinations. |
| `--output-dir` | `str` | `../results` | Target directory where the logs and CSVs will be outputted. |

## Outputs

By default, logs reporting the simulation results (including optimal paths and tracking of the Best SAD metric) are saved inside the `results/` folder. For custom paths, use the `--output-dir` argument.

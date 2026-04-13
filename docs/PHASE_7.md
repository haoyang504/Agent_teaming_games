# Phase 7 — Discussion Order Experiment + Portkey Migration

> **Status: COMPLETE**

> **Prerequisites:** Read `CONTEXT.md` first. Phases 1–6 must be complete.
> **Goal:** Add a configurable discussion speaking order parameter (ABC / CBA / random) to enable experiments comparing how speaking sequence affects group decision-making. Separately, migrate the LLM gateway from direct OpenAI API calls to the Portkey AI gateway.

---

## What exists now

In `experiment_runner.py`, the discussion phase shuffles agent speaking order randomly each round using `disc_rng.shuffle()` (added in Phase 3). There is no way to force a fixed order. The experiment always uses random ordering, which means we cannot isolate whether speaking first or last confers an advantage.

The LLM client is initialized as `OpenAI(api_key=config.OPENAI_API_KEY)` pointing directly at OpenAI's API.

---

## Task 7A: Add `discussion_order` parameter

### What to build

Add a `discussion_order` parameter to both `run_iteration()` and `run_experiment()` in `experiment_runner.py`. It accepts three values:

- **`"random"`** — current behavior. Shuffle speaking order each round using `disc_rng`.
- **`"ABC"`** — fixed order: Agent 1 → Agent 2 → Agent 3, every round.
- **`"CBA"`** — fixed order: Agent 3 → Agent 2 → Agent 1, every round.

### Changes to `experiment_runner.py`

**`run_iteration()` signature:**
```python
def run_iteration(
    ...
    discussion_order: str = "random",
    ...
)
```

**Discussion loop logic:**
```python
for disc_round in range(1, num_discussion_rounds + 1):
    if discussion_order == "ABC":
        round_order = sorted(agents, key=lambda a: a.agent_id)
    elif discussion_order == "CBA":
        round_order = sorted(agents, key=lambda a: a.agent_id, reverse=True)
    else:
        round_order = list(agents)
        disc_rng.shuffle(round_order)
```

**`run_experiment()` signature:**
```python
def run_experiment(
    ...
    discussion_order: str = "random",
    ...
)
```

Passed through to `run_iteration()` on each iteration call.

### Logging

- Added `discussion_order` to the experiment log dict
- Added `do{discussion_order}` to the experiment ID string (e.g. `doABC`, `doCBA`, `dorandom`)

### Important: What stays fixed

- Proposal order is always 1 → 2 → 3 (not affected)
- Final selection is always Agent 3 / Agent C (not affected)
- Only the discussion phase speaking order changes

---

## Task 7B: Add `--discussion-order` CLI argument

### Changes to `run_experiment.py`

```python
parser.add_argument(
    "--discussion-order",
    type=str,
    default="random",
    choices=["random", "ABC", "CBA"],
    help=(
        "Discussion speaking order (default: random). "
        "ABC=fixed 1→2→3, CBA=fixed 3→2→1, random=shuffled each round."
    ),
)
```

Passed through to `run_experiment()` as `discussion_order=args.discussion_order`.

---

## Task 7C: Migrate to Portkey AI gateway

### Why

The team uses Portkey as an LLM gateway for API key management and routing. Direct OpenAI calls have been replaced with Portkey SDK calls, which share the same `chat.completions.create()` interface.

### Changes to `config.py`

```python
# Legacy OpenAI direct access (commented out)
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-proj-...")

# Portkey gateway
PORTKEY_API_KEY = "gzCOSjXNGvo4U5hl4R1RXtKMIEsT"
PORTKEY_MODEL = "@info-yd4956-ope-01316b/gpt-5-mini"
```

### Changes to `experiment_runner.py`

- Import changed from `from openai import OpenAI` to `from portkey_ai import Portkey`
- Client initialization changed from `OpenAI(api_key=config.OPENAI_API_KEY)` to `Portkey(api_key=config.PORTKEY_API_KEY)`
- Model passed to agents is `config.PORTKEY_MODEL` instead of the CLI `--model` value

### Changes to `agent.py`

- Import updated: tries `from portkey_ai import Portkey as LLMClient`, falls back to `from openai import OpenAI as LLMClient`
- Type hints on `client` parameter made generic (removed `OpenAI` type annotation) since both SDKs share the same interface
- `_chat()` retry logic preserved — on any error, retries without `temperature` parameter (some Portkey-routed models don't support it)

### Dependency

Requires `pip install portkey-ai`.

---

## Tests

### Test 1: ABC order

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1 --discussion-order ABC
```

Verify: experiment ID contains `doABC`, discussion round shows order `1 → 2 → 3`.

### Test 2: CBA order

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1 --discussion-order CBA
```

Verify: experiment ID contains `doCBA`, discussion round shows order `3 → 2 → 1`.

### Test 3: Random order (default)

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 3
```

Verify: experiment ID contains `dorandom`, discussion rounds show varying orders.

### Test 4: Multi-round fixed order stays fixed

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 3 --discussion-order CBA
```

Verify: all 3 discussion rounds show the same order `3 → 2 → 1`.

### Test 5: Portkey connectivity

```bash
python run_experiment.py --config high_div --setting 1 --k 1 --iterations 1 --discussion-rounds 1
```

Verify: completes without API errors, uses `gpt-5-mini` via Portkey.

---

## What NOT to do in Phase 7

- Do NOT change proposal order — always 1 → 2 → 3.
- Do NOT change which agent does final selection — always Agent 3.
- Do NOT modify `knowledge_manager.py` or `moon_survival_env.py`.
- Do NOT add discussion order to the agent's system prompt — agents should not know their speaking order.

---

## Implementation Notes (post-completion)

### Verified working

- ABC order confirmed: experiment ID `doABC`, single discussion round shows `1 → 2 → 3`
- Portkey API confirmed working with model `@info-yd4956-ope-01316b/gpt-5-mini`
- Retry logic triggered and recovered successfully during testing (Agent 3 proposal parse failed attempt 1, succeeded attempt 2)

### Files changed

- **`experiment_runner.py`**: Added `discussion_order` parameter to `run_iteration()` and `run_experiment()`. Branching logic in discussion loop. Added to experiment log and experiment ID. Migrated from `OpenAI` to `Portkey` client.
- **`run_experiment.py`**: Added `--discussion-order` CLI arg with choices `random`/`ABC`/`CBA`.
- **`config.py`**: Added `PORTKEY_MODEL`, commented out `OPENAI_API_KEY`.
- **`agent.py`**: Updated import to prefer `portkey_ai`, generic client type hints, retry-without-temperature logic.

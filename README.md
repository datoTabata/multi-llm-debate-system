# Multi-LLM Collaborative Debate System

Three LLMs solve a problem independently, peer-review each other's solutions,
refine their answers based on the critiques, and a fourth LLM judges the refined
answers and returns the best one. The goal is to beat a single LLM working alone
by combining diverse perspectives and adversarial review.

On the 29-problem "Easy Problems That LLMs Get Wrong" benchmark
(arXiv 2405.19616), the debate system scores **0.69** vs a single-LLM baseline of
**0.57**.

## Workflow

For each problem, the pipeline runs five stages (`src/debate/pipeline.py`):

1. **Role preference** — each of the four models is told its own identity, the
   named peer roster, and today's date, and rates its confidence as Solver vs
   Judge.
2. **Role assignment** — the model with the highest judge confidence becomes the
   Judge; the other three become Solvers (plain logic, no API call).
3. **Independent solutions** — the three Solvers answer the problem
   independently.
4. **Peer review** — each Solver reviews the other two solutions (reviews are
   anonymized to avoid brand bias).
5. **Refinement** — each Solver revises its answer using the reviews it received.
6. **Judgment** — the Judge picks the best refined answer.

Everything is async: independent calls within a stage fan out with
`asyncio.gather`, and problems run concurrently (capped by
`max_concurrent_problems`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                       # installs the debate package + deps
echo "OPEN_ROUTER_KEY=sk-..." > .env   # your OpenRouter API key (gitignored)
```

The project talks to every model through [OpenRouter](https://openrouter.ai); no
per-provider keys are needed.

## Configuration

All runtime settings live in `config/models.toml` — no environment variables:

- `[participants.model_a..d]` — which OpenRouter model fills each of the four
  slots (default mix: gpt-4o-mini, claude-haiku-4.5, gemini-2.5-flash, grok-4.3).
- `[defaults]` — `temperature` and `max_tokens`, overridable per slot.
- `[grader]` — the model used to grade free-answer problems (default
  gemini-2.5-flash at temperature 0.0).
- `[run]` — which dataset and which problems to run, concurrency cap, and output
  file. Either list explicit `problem_ids`, or set `problem_limit` for a random
  (optionally `seed`ed) sample.

The config is validated on load (`AppConfig` in `src/debate/models.py`), so a bad
key or type fails fast with a clear error.

## Run

```bash
python scripts/run_demo.py       # runs the debate, writes outputs/<output_file>
python scripts/plot_metrics.py   # bar chart of the latest run's metrics
```

`run_demo.py` writes a self-describing JSON to `outputs/`: a `config` block
(resolved models/temperatures/max_tokens, chosen problem ids, seed), the full
per-problem `results`, per-problem `evaluations`, and aggregate `metrics`.

## Datasets

- `data/problems.json` — 25 challenging problems across math/logic, physics,
  logic puzzles, and game theory (the assignment set).
- `data/llm_hard_problems.json` — the 29-question "Easy Problems That LLMs Get
  Wrong" benchmark (overfitting traps: the six-horse race, a Monty Hall variant,
  river crossing, etc.).

Each problem carries an `answer_type` that decides how an answer is graded:

- `number` — numeric compare (units stripped, fractions handled, rel-tol 1e-3).
- `multiple_choice` — normalized string match.
- `free_answer` — graded by the `[grader]` LLM for semantic equivalence.

Grading logic lives in `Problem.check()` (`src/debate/problem.py`).

## Metrics

`run_demo.py` reports, and `plot_metrics.py` charts:

- **accuracy** — debate-system accuracy (the judge's chosen answer).
- **single_llm_accuracy** — baseline: each Solver's independent, pre-debate
  answer, averaged (no extra API calls).
- **consensus_rate** — how often all three Solvers agree after refinement.
- **judge_accuracy_on_disagreements** — how often the Judge picks a correct
  answer when the Solvers disagree.
- **improvement_rate** — how often refinement turned a wrong answer right.

## Project structure

```text
config/models.toml        models + run settings + grader (single config source)
data/                     problems.json, llm_hard_problems.json
src/debate/
  models.py               OpenRouterClient (async), typed config, grader factory
  problem.py              Problem model + check() (type-aware grading)
  pipeline.py             the five debate stages
  schemas.py              Pydantic schemas (structured outputs + result/metrics)
  prompts.py              prompt builders
  evaluation.py           metrics
scripts/
  run_demo.py             async end-to-end runner
  plot_metrics.py         metrics plot
outputs/                  generated results + plots (gitignored)
pyproject.toml            package metadata, dependencies, ruff config
```

## Development

```bash
ruff check src scripts     # lint
ruff format src scripts    # format
```

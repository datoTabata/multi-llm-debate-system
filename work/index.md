# Project Index — Multi-LLM Collaborative Debate System

**Last updated:** 2026-07-03
**Due date:** 2026-07-03
**Staleness check:** If this date trails recent commits or the code clearly
diverged, trust the code and update this file.

## What this project is
Three LLMs solve a problem independently, peer-review each other, refine their
answers, and a fourth LLM judges and returns the best answer. Goal: beat single-
LLM and majority-vote baselines on a 25-problem dataset. Full spec in
[`../Assignment.md`](../Assignment.md). User-facing usage in
[`../README.md`](../README.md).

## Current status by phase
| Phase | Item | Status |
|-------|------|--------|
| 1 | 25-problem dataset (`data/problems.json`) | ✅ done |
| 2 | 5-stage debate pipeline (`src/debate/pipeline.py`) — **async** | ✅ done |
| 2 | Model provider — **OpenRouter only**, config-driven | ✅ done |
| 2 | Concurrency — async stages + concurrent problems | ✅ done |
| 3 | System metrics (accuracy, consensus, judge-acc, improvement) | ✅ done |
| 3 | Type-aware answer checking (number / MC / free-answer + LLM grader) | ✅ done |
| 3 | Single-LLM baseline (solo solver accuracy) | ✅ done |
| 3 | Majority-vote baseline | ❌ missing |
| 3 | Comparison plot (system vs single-LLM) | ✅ done |

## How it runs (no env vars)
Everything is driven by **`config/models.toml`**:
- `[participants.model_a..d]` — which model fills each slot (current mix:
  gpt-4o-mini, claude-haiku-4.5, gemini-2.5-flash, grok-4.3).
- `[defaults]` — `temperature`, `max_tokens` (now 40000; per-slot overridable).
- `[run]` — which problems to run:
  - `dataset` — which file under `data/` to load (e.g. `problems.json` or
    `llm_hard_problems.json`).
  - `problem_ids` — if non-empty, run exactly those (in order); `problem_limit`
    ignored.
  - `problem_limit` + `seed` — otherwise run a RANDOM sample of that many
    (seeded = reproducible; remove `seed` for fresh randomness each run).
  - `max_concurrent_problems` — 0 = all problems at once; positive = throttle.
  - `output_file`.
- API key lives in `.env` as `OPEN_ROUTER_KEY` (gitignored).

The run output (`outputs/<output_file>`) includes a top-level `config` block
(participants' model/temperature/max_tokens, chosen problem_ids, seed), so every
result is self-describing.

## Concurrency (asyncio)
The pipeline is async. Within a problem, each stage fans out its independent
model calls with `asyncio.gather` (4 role-prefs, 3 solves, 6 reviews, 3 refines);
stages stay sequential because each depends on the previous. Across problems,
`run_all_debates` runs every problem concurrently (capped by
`max_concurrent_problems`). Client async methods: `OpenRouterClient.agenerate` /
`agenerate_structured` (SDK `chat.send_async`) — the client is async-only. The
grader is a sync callable that wraps `agenerate_structured` in `asyncio.run`
(only safe outside a running loop; grading happens after the debate loop ends)
and memoizes results by (question, predicted, correct).

```bash
pip install -e .                  # once: install the debate package + deps (pyproject.toml)
python scripts/run_demo.py        # runs the debate, writes outputs/<output_file>
python scripts/plot_metrics.py    # plots metrics for the latest run (reads output_file)
ruff check src scripts && ruff format src scripts   # lint + format
```
Deps/tooling live in `pyproject.toml` (`requirements.txt` is gone). The current
dev env is the `llm_homeworks` conda env, where the package is installed
editable.

## Datasets
- `data/problems.json` — the 25-problem assignment dataset.
- `data/llm_hard_problems.json` — the full 29-question "Easy Problems That LLMs
  Get Wrong" benchmark (arXiv 2405.19616), `hard_001..030` minus the deferred
  Q8 (Bible sentence). Overfitting traps: horses, Monty-Hall-variant, river
  crossing, Sally's sisters, LOLLAPALOOZA, etc.

**Latest result (29 hard problems):** debate **0.69** vs single-LLM baseline
**0.57** (+12 pts); judge resolves disagreements at 0.65. See
`outputs/hard_result.json` / `outputs/hard_result_plot.png`.

**Strong-vs-weak experiment (2026-07-03):** 1 strong model
(gemini-3-flash-preview) + 3 weak (gpt-4o-mini, gemma-3-27b-it,
llama-3.1-8b-instruct) on 5 problems incl. new `hard_031` (car-wash trap).
Debate 0.20 vs single-LLM 0.00. The strong model self-selected as judge every
time; when all three weak solvers converge on the same wrong answer (e.g. all
said "walk" on the car-wash problem) the judge cannot recover — it can only
pick among solver answers. See `outputs/gemini3_strong_vs_weak.json`.

## How answers are checked
Each problem has an `answer_type`:
- `number` — numeric compare (units stripped, fractions handled, rel-tol 1e-3).
- `multiple_choice` — normalized string match.
- `free_answer` — graded by an LLM (`[grader]` slot, default gemini-2.5-flash)
  for semantic equivalence.

All comparison logic lives in **`Problem.check(predicted, grader)`**
(`src/debate/problem.py`) — one model, no inheritance, branches on
`answer_type`. The grader is injected as a callable; if absent, free-answer
falls back to a string match.

## File map
- `config/models.toml` — models + run settings + `[grader]` (single source of config).
- `src/debate/models.py` — `OpenRouterClient` (async-only), typed config
  (`AppConfig`/`RunConfig`/`SlotConfig`), `load_config`, grader factory.
- `src/debate/problem.py` — `Problem` model + `check()` (type-aware grading).
- `src/debate/pipeline.py` — the 5 debate stages (all take/return Pydantic).
- `src/debate/schemas.py` — Pydantic schemas: structured outputs +
  `DebateResult`, `EvaluationRecord`, `RunMetrics`.
- `src/debate/prompts.py` — prompt builders (incl. `build_grader_prompt`).
- `src/debate/evaluation.py` — metrics (delegate to `Problem.check`).
- `data/llm_hard_problems.json` — LLM-hard / human-easy problem set.
- `scripts/run_demo.py` — async end-to-end runner (concurrent problems).
- `scripts/plot_metrics.py` — plot generator (reads config `output_file`).

## Known issues / TODO
- **Majority-vote baseline** still missing (single-LLM is done via
  `calculate_single_llm_accuracy`). Computable from existing independent
  solutions; no new debate calls needed.
- **Brittle number grading** — e.g. `hard_029` model answered "0.1-0.2 m (or ~0
  m theoretically)"; `_to_number` grabbed 0.1 and marked it wrong. Hedged ranges
  on number-type problems can misgrade.
- ~~Answer matching is brittle~~ — fixed: now type-aware via `Problem.check`
  (number / multiple_choice / free_answer + LLM grader). See [`../notes.md`](../notes.md).
- **Peer reviews are unstructured** — `generate_peer_reviews` uses `.generate()`
  (raw text) while every other stage is structured.
- ~~Dead `OpenRouterClient.name` field + sync `generate`/`generate_structured`~~
  — removed; the client is async-only now and the grader wraps
  `agenerate_structured` in `asyncio.run`.
- ~~JSON-string passing smell~~ — fixed: pipeline stages now pass/return
  Pydantic objects (`DebateResult` in `schemas.py`); serialization happens once
  in `run_demo.py` via `model_dump()`.

## Recent changes
- **2026-07-03** — Strong-vs-weak run + robustness fixes: added `hard_031`
  (car-wash trap) to `data/llm_hard_problems.json` (now 30 problems);
  `OpenRouterClient` gained a 300s `request_timeout` (`asyncio.wait_for` — the
  SDK has none, and CLOSE_WAIT connections hung a run forever) and retries now
  also cover `TimeoutError` + `httpx.HTTPError` (`agenerate` retries too);
  grader prompt hardened (reference answer is ground truth — the grader
  previously re-solved trap problems itself and misgraded correct answers).
  Model notes: `ministral-8b` ignores JSON schemas; `gpt-4.1-nano` intermittently
  routes to endpoints requiring strict schemas (`additionalProperties: false`)
  which pydantic's `model_json_schema()` doesn't emit — both unusable as-is.
- **2026-07-03** — Code review fixes: `GraderConfig` restores grader defaults
  (temp 0.0 / gemini model) when `[grader]` is partial; grader results are
  memoized (11 → 3 LLM calls per free-answer problem across metrics);
  `judge_selected_correct_answer` reuses `evaluate_result`; plotter uses new
  `load_run_config` (no full-config validation just to read `output_file`);
  `OpenRouterClient` fields are required ([defaults] is the single source of
  truth); `*.egg-info/` gitignored.
- **2026-07-03** — Tooling: added `pyproject.toml` (project metadata + deps +
  ruff config; replaces `requirements.txt`, drops unused `pandas`) and `ruff`
  (lint + format, all clean). Package installs editable (`pip install -e .`),
  so the `sys.path.insert("src")` hacks in both scripts are gone. Ruff caught
  and fixed: unused imports, `zip()` without `strict=`, `Optional[X]` →
  `X | None`, duplicated Field description (now `FINAL_ANSWER_DESCRIPTION`).
- **2026-07-03** — Pydantic everywhere: `load_config` now returns a validated
  `AppConfig` (`RunConfig`/`DefaultsConfig`/`SlotConfig` in `models.py`);
  problems are `Problem` objects from load onward (pipeline + prompts take
  `Problem`, not dicts); `evaluate_result` returns `EvaluationRecord` and run
  metrics are a `RunMetrics` model (both in `schemas.py`). Also removed dead
  code: sync `generate`/`generate_structured` (client is async-only; the
  grader wraps `agenerate_structured` in `asyncio.run`) and the unused
  `OpenRouterClient.name` field.
- **2026-07-03** — Typed pipeline: every stage now takes/returns Pydantic
  schema objects instead of JSON strings; new `DebateResult` model wraps a full
  debate. `evaluation.py` reads typed attributes (dropped
  `parse_json_response`/`extract_*` helpers); `run_demo.py` serializes once at
  the output boundary. Also documented `collect_role_preferences` and renamed
  its misleading `model_name` param to `slot_name`.
- **2026-06-26** — Single-LLM baseline (`calculate_single_llm_accuracy`): grades
  each solver's independent pre-debate answer; reported next to system accuracy
  and plotted. On the 29 hard problems: 0.69 (debate) vs 0.57 (single LLM).
- **2026-06-26** — Completed the LLM-hard dataset to the full 29 (added Q11–Q30
  from arXiv 2405.19616). Run robustness: structured calls retry 3x on
  malformed/truncated responses; per-problem failures are isolated (one bad
  problem no longer kills the batch); progress log prints `Solved N/total`.
- **2026-06-26** — Async pipeline + concurrency. Every stage fans out its
  independent calls via `asyncio.gather`; problems run concurrently in
  `run_all_debates` (capped by `max_concurrent_problems`). Added `agenerate` /
  `agenerate_structured` on the client. 9 hard problems ran in ~2m13 vs ~9x
  sequential.
- **2026-06-26** — Added `[run] dataset` option + `data/llm_hard_problems.json`
  (9 LLM-overfitting traps). Plotter now reads the config `output_file` and
  saves `<stem>_plot.png` with value labels.
- **2026-06-26** — Problem selection is config-driven: `[run] problem_ids` runs
  an explicit set (ignoring `problem_limit`); otherwise `problem_limit` picks a
  RANDOM sample (reproducible via `seed`). Output records the chosen ids + seed.
- **2026-06-26** — Type-aware answer checking. Added `answer_type` to all 25
  problems (16 number / 3 multiple_choice / 6 free_answer) and a single
  `Problem.check()` method that grades deterministically for number/MC and via
  an LLM grader (`[grader]` config) for free-answer. Evaluation metrics now
  delegate to `Problem.check`; removed the old `exact_match`/`normalize_answer`.
- **2026-06-26** — Picker round is now self-aware: each model is told its own
  OpenRouter id, the named peer roster, the debate flow, the current date, and a
  knowledge-cutoff warning (`build_role_preference_prompt` in `prompts.py`).
  Review/judge rounds stay **anonymized** on purpose (avoid brand bias).
- **2026-06-26** — Fixed truncated-JSON crash: `max_tokens` was unset (provider
  default) so long solver reasoning got cut mid-JSON. Added `max_tokens` to
  `OpenRouterClient` (config-driven, default 40000) + a clear `finish_reason ==
  "length"` guard instead of a cryptic Pydantic error.
- **2026-06-26** — Run output now carries a `config` block (models, temperatures,
  max_tokens, problem_limit) derived from the live clients.
- **2026-06-26** — Switched to OpenRouter as the single provider (removed the
  mock/openai/gemini abstraction) and made runs config-driven via
  `config/models.toml`. Simplified schemas (dropped confidence bounds).

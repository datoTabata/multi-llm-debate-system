# Project Index — Multi-LLM Collaborative Debate System

**Last updated:** 2026-06-26
**Due date:** 2026-07-03
**Staleness check:** If this date trails recent commits or the code clearly
diverged, trust the code and update this file.

## What this project is
Three LLMs solve a problem independently, peer-review each other, refine their
answers, and a fourth LLM judges and returns the best answer. Goal: beat single-
LLM and majority-vote baselines on a 25-problem dataset. Full spec in
[`../Assignment.md`](../Assignment.md). User-facing usage in
[`../README.md`](../README.md) (note: README is somewhat out of date).

## Current status by phase
| Phase | Item | Status |
|-------|------|--------|
| 1 | 25-problem dataset (`data/problems.json`) | ✅ done |
| 2 | 5-stage debate pipeline (`src/debate/pipeline.py`) | ✅ done |
| 2 | Model provider — **OpenRouter only**, config-driven | ✅ done |
| 3 | System metrics (accuracy, consensus, judge-acc, improvement) | ✅ done |
| 3 | **Baselines** (single-LLM, majority vote) | ❌ missing |
| 3 | Comparison plot (system vs baselines) | ⚠️ partial — only system metrics plotted |

## How it runs (no env vars)
Everything is driven by **`config/models.toml`**:
- `[participants.model_a..d]` — which model fills each slot (current mix:
  gpt-4o-mini, claude-haiku-4.5, gemini-2.5-flash, grok-4.3).
- `[run]` — `problem_limit`, `output_file`.
- API key lives in `.env` as `OPEN_ROUTER_KEY` (gitignored).

```bash
python scripts/run_demo.py        # runs the debate, writes outputs/<output_file>
python scripts/plot_metrics.py    # plots metrics from outputs/demo_result.json
```

## File map
- `config/models.toml` — models + run settings (single source of config).
- `src/debate/models.py` — `OpenRouterClient` + `load_config` (official openrouter SDK).
- `src/debate/pipeline.py` — the 5 debate stages.
- `src/debate/schemas.py` — Pydantic structured-output schemas.
- `src/debate/prompts.py` — prompt builders.
- `src/debate/evaluation.py` — metrics.
- `scripts/run_demo.py` — end-to-end runner.
- `scripts/plot_metrics.py` — plot generator.

## Known issues / TODO
- **Baselines not implemented** (required deliverable) — single-LLM and majority
  vote. Data already exists in each result; no new API calls needed.
- **Answer matching is brittle** — exact string compare; see [`../notes.md`](../notes.md).
- **Peer reviews are unstructured** — `generate_peer_reviews` uses `.generate()`
  (raw text) while every other stage is structured.
- **Cleanups**: unused `pandas` dep; dead `OpenRouterClient.name` field; the
  JSON-string passing smell (pipeline serializes objects to JSON strings and
  re-parses them downstream).

## Recent changes
- **2026-06-26** — Switched to OpenRouter as the single provider (removed the
  mock/openai/gemini abstraction) and made runs config-driven via
  `config/models.toml`. Simplified schemas (dropped confidence bounds).

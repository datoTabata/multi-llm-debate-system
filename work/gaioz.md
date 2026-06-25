# Gaioz — Work Log

## Provider integration: OpenRouter + config-driven runs

**Goal:** use one account/key to access many LLM providers instead of buying
each separately, and drive runs from a config file instead of env vars.

### What I did
- **Switched to OpenRouter** as the single provider — one API key
  (`OPEN_ROUTER_KEY` in `.env`) gives access to GPT, Claude, Gemini and Grok.
- **Removed the multi-client abstraction** (`ModelClient` base + mock/openai/
  gemini clients). Replaced it with one `OpenRouterClient` that uses the
  official `openrouter` Python SDK (`client.chat.send`).
- **Added `config/models.toml`** as the single source of run configuration:
  - `[participants.model_a..d]` — which model fills each of the four slots
    (currently a real mix: gpt-4o-mini, claude-haiku-4.5, gemini-2.5-flash,
    grok-4.3).
  - `[defaults]` temperature, with per-model override.
  - `[run]` — `problem_limit` and `output_file` (previously the
    `PROBLEM_LIMIT` / `OUTPUT_FILE` env vars).
- **Updated `run_demo.py`** to read run settings from the config; no more env
  vars to run.
- **Simplified schemas** — dropped the `confidence` `ge=0/le=1` bounds. They
  generated `minimum`/`maximum` in the JSON schema, which Anthropic's
  structured-output API rejects, and forced ugly workarounds (schema
  sanitizing + a retry loop). Removing the bounds let `generate_structured`
  become a clean ~14-line method.
- **Dependencies** — swapped `openai`/`google-genai` for `openrouter`.
- Removed the stale `test_openai_client.py`.

### Verified
- All four models resolve and respond on live calls.
- Structured outputs (`generate_structured`) work across all four models for
  every schema, including the nested `RefinementResponse`.

### Notes / follow-ups
- OpenRouter model slugs go stale — if a run 404s with "No endpoints found",
  update the slug in `config/models.toml` (no code change needed).
- Remaining cleanup ideas tracked separately: drop unused `pandas` dep, remove
  dead `OpenRouterClient.name` field, and the JSON-string passing refactor
  (pipeline serializes/re-parses objects between stages).
- Answer-matching brittleness is logged in `notes.md`.

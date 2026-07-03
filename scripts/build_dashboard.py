"""Generate an interactive HTML results dashboard from a debate run.

Reads a run's output JSON (default outputs/hard_result.json) plus the dataset,
and writes outputs/<run>_dashboard.html. Regenerable: re-run after any new
debate to refresh the dashboard. Also called automatically by run_demo.py.

    python scripts/build_dashboard.py [outputs/<run>.json]
"""

import json
import sys
from html import escape
from pathlib import Path

from debate.problem import Problem

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- data loading -----------------------------------------------------------


def _parse(value):
    """Nested fields are Pydantic objects in new runs, JSON strings in old ones."""
    return json.loads(value) if isinstance(value, str) else value


def _norm_conf(conf: float) -> float:
    """Some models report confidence on 0-100, others on 0-1. Normalize to 0-1."""
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def load(run_path: Path) -> dict:
    data = json.loads(run_path.read_text())
    dataset_name = data.get("config", {}).get("dataset", "llm_hard_problems.json")
    problems = {
        p["id"]: p
        for p in json.loads((PROJECT_ROOT / "data" / dataset_name).read_text())
    }
    return data, problems


# ---- derive view model ------------------------------------------------------


def participant_rows(config: dict) -> list[dict]:
    out = []
    for slot, spec in config.get("participants", {}).items():
        out.append({"slot": slot, "model": spec["model"], "temp": spec.get("temperature")})
    return out


def grid_cells(data: dict, problems: dict) -> list[dict]:
    evals = {e["problem_id"]: e for e in data["evaluations"]}
    cells = []
    for r in data["results"]:
        pid = r["problem_id"]
        e = evals[pid]
        cells.append(
            {
                "id": pid,
                "question": problems[pid]["question"],
                "predicted": e["predicted_answer"],
                "correct": e["correct_answer"],
                "is_correct": e["is_correct"],
                "answer_type": e["answer_type"],
            }
        )
    return cells


def drilldown(data: dict, problems: dict, pid: str) -> dict:
    r = next(x for x in data["results"] if x["problem_id"] == pid)
    e = next(x for x in data["evaluations"] if x["problem_id"] == pid)
    roles = r["roles"]
    models = {s: v["model"] for s, v in data["config"]["participants"].items()}
    solver_ids = ("solver_1", "solver_2", "solver_3")
    solvers = []
    for sid in solver_ids:
        init = _parse(r["solutions"][sid])
        ref = _parse(r["refined_solutions"][sid])
        # Reviews this solver's solution received. The pipeline appends them in a
        # fixed order: the other solvers, in solver_ids order (excluding self),
        # so we can recover each review's author.
        reviewers = [x for x in solver_ids if x != sid]
        reviews = r["peer_reviews"].get(sid, [])
        # strict=False: if a solver received fewer than 2 reviews, just show
        # what's there rather than crashing.
        reviews_received = [
            {"from": who, "text": text}
            for who, text in zip(reviewers, reviews, strict=False)
        ]
        solvers.append(
            {
                "sid": sid,
                "model": models[roles[sid]],
                "initial": init["answer"],
                "initial_conf": _norm_conf(init["confidence"]),
                "reasoning": init.get("reasoning", ""),
                "refined": ref["refined_answer"],
                "refined_conf": _norm_conf(ref["confidence"]),
                "refined_solution": ref.get("refined_solution", ""),
                "flipped": init["answer"].strip().lower() != ref["refined_answer"].strip().lower(),
                "reviews_received": reviews_received,
            }
        )
    # Stage 0: each model's Solver-vs-Judge self-assessment. Order the chosen
    # judge first.
    role_prefs = []
    for slot in data["config"]["participants"]:
        pref = _parse(r["role_preferences"][slot])
        role_prefs.append(
            {
                "model": models[slot],
                "preferred": pref["preferred_role"],
                "solver_conf": _norm_conf(pref["solver_confidence"]),
                "judge_conf": _norm_conf(pref["judge_confidence"]),
                "reasoning": pref.get("reasoning", ""),
                "is_judge": roles["judge"] == slot,
            }
        )
    role_prefs.sort(key=lambda p: not p["is_judge"])

    judgment = _parse(r["judgment"])
    return {
        "id": pid,
        "question": problems[pid]["question"],
        "correct": e["correct_answer"],
        "predicted": e["predicted_answer"],
        "is_correct": e["is_correct"],
        "explanation": problems[pid].get("explanation", ""),
        "judge_model": models[roles["judge"]],
        "winner": judgment["winner"],
        "judge_conf": _norm_conf(judgment["confidence"]),
        "judge_reasoning": judgment["reasoning"],
        "role_prefs": role_prefs,
        "solvers": solvers,
    }


def select_drilldown_id(data: dict) -> str:
    """Pick the most illustrative problem to feature in the drill-down.

    Prefer a problem the debate got right where the solvers disagreed and at
    least one changed its answer after review — that shows the whole mechanism
    working. Falls back through weaker criteria to the first result.
    """
    evals = {e["problem_id"]: e for e in data["evaluations"]}
    solver_ids = ("solver_1", "solver_2", "solver_3")

    def score(r: dict) -> tuple[int, int, int]:
        refined = {sid: _parse(r["refined_solutions"][sid])["refined_answer"] for sid in solver_ids}
        disagree = len({a.strip().lower() for a in refined.values()}) > 1
        flipped = any(
            _parse(r["solutions"][sid])["answer"].strip().lower() != refined[sid].strip().lower()
            for sid in solver_ids
        )
        is_correct = evals[r["problem_id"]]["is_correct"]
        # correctness dominates, then disagreement, then a visible flip.
        return (int(is_correct), int(disagree), int(flipped))

    best = max(data["results"], key=score, default=None)
    return best["problem_id"] if best else data["results"][0]["problem_id"]


# ---- rendering --------------------------------------------------------------

METRIC_LABELS = {
    "accuracy": ("Debate accuracy", "Judge's chosen answer, graded"),
    "single_llm_accuracy": ("Single-LLM baseline", "Each solver alone, pre-debate"),
    "consensus_rate": ("Consensus rate", "All 3 solvers agree after refinement"),
    "judge_accuracy_on_disagreements": ("Judge accuracy", "When solvers disagree"),
    "improvement_rate": ("Improvement rate", "Refinement fixed a wrong answer"),
}


def pct(x) -> str:
    return "—" if x is None else f"{round(x * 100)}%"


def sol_label(sid: str) -> str:
    return {"solver_1": "Solver 1", "solver_2": "Solver 2", "solver_3": "Solver 3"}[sid]


def render(data: dict, problems: dict) -> str:
    m = data["metrics"]
    parts = participant_rows(data["config"])
    cells = grid_cells(data, problems)
    default_pid = select_drilldown_id(data)
    n = len(cells)
    n_correct = sum(c["is_correct"] for c in cells)

    debate = m["accuracy"]
    single = m["single_llm_accuracy"]
    delta = round((debate - single) * 100)

    part_badges = "".join(
        f'<div class="flex items-center gap-2 rounded-box bg-base-200 px-3 py-2">'
        f'<span class="badge badge-sm badge-primary">{escape(p["slot"].replace("model_", "").upper())}</span>'
        f'<span class="font-mono text-sm text-base-content/80">{escape(p["model"])}</span></div>'
        for p in parts
    )

    # Plain bordered cards (not DaisyUI .stat, whose internal grid + ring
    # overflow tight grid cells).
    metric_tiles = ""
    for key, (label, sub) in METRIC_LABELS.items():
        val = m.get(key)
        primary = key == "accuracy"
        border = "border-2 border-primary" if primary else "border border-base-content/10"
        metric_tiles += (
            f'<div class="min-w-0 rounded-box bg-base-200 p-4 {border}">'
            f'<div class="truncate text-xs text-base-content/60">{escape(label)}</div>'
            f'<div class="mt-1 text-3xl font-bold leading-tight {"text-primary" if primary else ""}">{pct(val)}</div>'
            f'<div class="mt-1 text-xs text-base-content/50">{escape(sub)}</div></div>'
        )

    # per-problem grid cells; clicking one switches the drill-down below.
    grid_html = ""
    for c in cells:
        cls = "bg-success/80 hover:bg-success" if c["is_correct"] else "bg-error/80 hover:bg-error"
        active = " ring-2 ring-primary" if c["id"] == default_pid else ""
        grid_html += (
            f'<button id="cell_{escape(c["id"])}" onclick="showProblem(\'{escape(c["id"])}\')" '
            f'class="pcell {cls}{active} aspect-square rounded-md text-[10px] font-semibold '
            f'text-base-100 transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-primary" '
            f'title="{escape(c["id"])}">{escape(c["id"].replace("hard_", ""))}</button>'
        )

    # Drill-down as a vertical pipeline (Round 1 -> Peer review -> Round 2 ->
    # Judge). Each solver keeps a consistent accent colour across the stages.
    # One hidden panel is rendered per problem; clicking a grid cell swaps them.
    solver_accent = {"solver_1": "info", "solver_2": "secondary", "solver_3": "accent"}

    def sbadge(sid: str) -> str:
        color = solver_accent[sid]
        return f'<span class="badge badge-sm badge-{color}">{escape(sol_label(sid).replace("Solver ", "S"))}</span>'

    def scroll_text(text: str) -> str:
        return (
            '<div class="mt-2 max-h-56 overflow-y-auto whitespace-pre-wrap rounded bg-base-300/40 '
            f'p-2 text-[11px] leading-snug text-base-content/70">{escape(text)}</div>'
        )

    def cell_open(color: str) -> str:
        return f'<div class="min-w-0 rounded-box border-l-4 border-{color} bg-base-200 p-3">'

    def dd_panel(dd: dict, dd_problem: Problem) -> str:
        """Render one problem's full drill-down pipeline (initially hidden
        unless it is the default problem)."""

        def ans_badge(text: str) -> str:
            # number / multiple_choice grade deterministically (no API);
            # free_answer can't be graded per-solver cheaply, so show it neutral.
            if dd_problem.answer_type in ("number", "multiple_choice"):
                color = "badge-success" if dd_problem.check(text) else "badge-error"
            else:
                color = "badge-ghost"
            return f'<span class="badge {color} badge-lg font-mono">{escape(text)}</span>'

        # Round 1 and Round 2 stay 3 columns (one output per solver); each shows
        # the answer plus the full reasoning / refined solution text.
        round1_cells = round2_cells = ""
        for s in dd["solvers"]:
            color = solver_accent[s["sid"]]
            flip = (
                '<span class="text-[11px] font-semibold text-warning">↑ flipped after review</span>'
                if s["flipped"]
                else '<span class="text-[11px] text-base-content/40">unchanged</span>'
            )
            round1_cells += (
                f'{cell_open(color)}<div class="mb-1">{sbadge(s["sid"])}</div>'
                f'<div class="flex items-center gap-2">{ans_badge(s["initial"])}'
                f'<span class="text-[11px] text-base-content/50">conf {s["initial_conf"]:.2f}</span></div>'
                f'{scroll_text(s["reasoning"])}</div>'
            )
            round2_cells += (
                f'{cell_open(color)}<div class="mb-1">{sbadge(s["sid"])}</div>'
                f'<div class="flex items-center gap-2">{ans_badge(s["refined"])}'
                f'<span class="text-[11px] text-base-content/50">conf {s["refined_conf"]:.2f}</span>{flip}</div>'
                f'{scroll_text(s["refined_solution"])}</div>'
            )

        # Peer review: one block per reviewed solver, its two authored reviews
        # side by side.
        review_blocks = ""
        for s in dd["solvers"]:
            color = solver_accent[s["sid"]]
            cards = ""
            for rv in s["reviews_received"]:
                cards += (
                    '<div class="min-w-0 rounded bg-base-300/40 p-2">'
                    f'<div class="mb-1 text-[11px] text-base-content/60">from {sbadge(rv["from"])}</div>'
                    '<div class="max-h-56 overflow-y-auto whitespace-pre-wrap text-[11px] leading-snug '
                    f'text-base-content/70">{escape(rv["text"])}</div></div>'
                )
            review_blocks += (
                f'<div class="rounded-box border-l-4 border-{color} bg-base-200 p-3">'
                f'<div class="mb-2 text-xs font-semibold">Reviews of {sbadge(s["sid"])}\'s solution</div>'
                f'<div class="grid grid-cols-1 gap-3 md:grid-cols-2">{cards}</div></div>'
            )

        # Round 0: role selection. Model identity matters here (the model is
        # reasoning about its own strengths), so keep the model name.
        role_cells = ""
        for rp in dd["role_prefs"]:
            hl = "border-2 border-primary" if rp["is_judge"] else "border border-base-content/10"
            tag = (
                '<span class="badge badge-primary badge-sm">→ became Judge</span>'
                if rp["is_judge"]
                else f'<span class="badge badge-ghost badge-sm">wanted {escape(rp["preferred"])}</span>'
            )
            role_cells += (
                f'<div class="min-w-0 rounded-box {hl} bg-base-200 p-3">'
                '<div class="mb-1 flex items-center justify-between gap-2">'
                f'<span class="truncate font-mono text-[11px] text-base-content/80">{escape(rp["model"])}</span>{tag}</div>'
                f'<div class="text-[11px] text-base-content/60">solver {rp["solver_conf"]:.2f} · judge {rp["judge_conf"]:.2f}</div>'
                f'{scroll_text(rp["reasoning"])}</div>'
            )

        winner_label = sol_label(dd["winner"])
        verdict = (
            '<span class="badge badge-success">debate got it right</span>'
            if dd["is_correct"]
            else '<span class="badge badge-error">debate got it wrong</span>'
        )
        hidden = "" if dd["id"] == default_pid else " hidden"

        return f"""
    <div class="dd-panel" id="dd_{escape(dd["id"])}"{hidden}>
    <div class="mb-4 rounded-box bg-base-200 p-4">
      <p class="text-base-content/90">{escape(dd["question"])}</p>
      <div class="mt-2 flex flex-wrap items-center gap-2 text-sm">
        <span class="badge badge-success">correct: {escape(dd["correct"])}</span>
        <span class="badge badge-ghost font-mono">debate answered: {escape(dd["predicted"])}</span>
        {verdict}
        <span class="text-base-content/60">{escape(dd["explanation"])}</span>
      </div>
    </div>
    <!-- Pipeline: stages stack vertically, flow goes down; 3 solvers side by side per stage -->
    <div class="space-y-1">
      <!-- Round 0 · role selection -->
      <div class="rounded-box bg-base-300/40 p-3">
        <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-base-content/50">Role selection · each model picks Solver or Judge</div>
        <div class="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">{role_cells}</div>
        <div class="mt-2 text-[11px] text-base-content/50">The model with the highest judge-confidence becomes the Judge; the other three solve.</div>
      </div>
      <div class="text-center text-2xl text-base-content/30">↓</div>
      <!-- Round 1 -->
      <div class="rounded-box bg-base-300/40 p-3">
        <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-base-content/50">Round 1 · independent solutions</div>
        <div class="grid grid-cols-1 gap-3 md:grid-cols-3">{round1_cells}</div>
      </div>
      <div class="text-center text-2xl text-base-content/30">↓</div>
      <!-- Peer review: each solver's two authored reviews side by side -->
      <div class="rounded-box bg-base-300/40 p-3">
        <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-base-content/50">Peer review</div>
        <div class="space-y-3">{review_blocks}</div>
      </div>
      <div class="text-center text-2xl text-base-content/30">↓</div>
      <!-- Round 2 -->
      <div class="rounded-box bg-base-300/40 p-3">
        <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-base-content/50">Round 2 · refined solutions</div>
        <div class="grid grid-cols-1 gap-3 md:grid-cols-3">{round2_cells}</div>
      </div>
      <div class="text-center text-2xl text-base-content/30">↓</div>
      <!-- Judgment -->
      <div class="rounded-box border-2 border-primary bg-primary/10 p-4">
        <div class="flex flex-wrap items-center gap-2">
          <span class="badge badge-primary badge-lg">Judgment</span>
          <span class="font-mono text-[11px] text-base-content/70">{escape(dd["judge_model"])}</span>
          <span class="ml-auto text-sm">picks <b>{escape(winner_label)}</b> · conf {dd["judge_conf"]:.2f}</span>
        </div>
        <p class="mt-2 text-sm text-base-content/80">{escape(dd["judge_reasoning"])}</p>
      </div>
    </div>
    </div>"""

    dd_panels = "".join(
        dd_panel(drilldown(data, problems, r["problem_id"]), Problem(**problems[r["problem_id"]]))
        for r in data["results"]
    )

    return f"""<!doctype html>
<html lang="en" data-theme="luxury">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-LLM Debate — Results</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/daisyui.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/themes.css">
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.2.4/dist/index.global.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  :where(.grid, .flex) > * {{ min-width: 0; }}
  :where(p, h1, h2, h3, li, td, th, .badge) {{ overflow-wrap: anywhere; }}
</style>
</head>
<body class="bg-base-100 text-base-content">
<div class="mx-auto max-w-6xl px-4 py-8 md:px-8">

  <!-- Header -->
  <header class="mb-8">
    <div class="flex items-center gap-3">
      <h1 class="text-3xl font-bold md:text-4xl">Multi-LLM Collaborative Debate</h1>
      <span class="badge badge-primary badge-lg">{n} problems</span>
    </div>
    <p class="mt-2 text-base-content/70">
      Three LLMs solve independently, peer-review, refine, and a fourth judges.
      Benchmark: “Easy Problems That LLMs Get Wrong” (arXiv 2405.19616).
    </p>
    <div class="mt-4 flex flex-wrap gap-2">{part_badges}</div>
  </header>

  <!-- Headline comparison -->
  <section class="mb-8 rounded-box border border-base-content/10 bg-base-200 p-6">
    <div class="grid grid-cols-1 items-center gap-6 md:grid-cols-[1fr_auto_1fr]">
      <div class="text-center">
        <div class="text-sm uppercase tracking-wide text-base-content/50">Debate system</div>
        <div class="pb-1 text-6xl font-bold leading-tight text-primary">{pct(debate)}</div>
        <div class="mt-1 text-sm text-base-content/60">{n_correct}/{n} correct</div>
      </div>
      <div class="text-center">
        <div class="badge badge-success badge-lg">+{delta} pts</div>
      </div>
      <div class="text-center">
        <div class="text-sm uppercase tracking-wide text-base-content/50">Single-LLM baseline</div>
        <div class="pb-1 text-6xl font-bold leading-tight text-base-content/40">{pct(single)}</div>
        <div class="mt-1 text-sm text-base-content/60">solvers working alone</div>
      </div>
    </div>
    <div class="mt-6">
      <div class="flex h-3 w-full overflow-hidden rounded-full bg-base-300">
        <div class="bg-primary" style="width:{round(debate * 100)}%"></div>
      </div>
    </div>
  </section>

  <!-- Metric tiles -->
  <section class="mb-8">
    <h2 class="mb-3 text-xl font-semibold">Metrics</h2>
    <div class="grid grid-cols-2 gap-3 md:grid-cols-5">{metric_tiles}</div>
  </section>

  <!-- Per-problem grid -->
  <section class="mb-8">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="text-xl font-semibold">Every problem</h2>
      <div class="flex items-center gap-4 text-sm">
        <span class="flex items-center gap-1"><span class="inline-block h-3 w-3 rounded bg-success"></span> correct</span>
        <span class="flex items-center gap-1"><span class="inline-block h-3 w-3 rounded bg-error"></span> wrong</span>
      </div>
    </div>
    <p class="mb-3 text-sm text-base-content/60">Click any cell to load that problem's full debate below.</p>
    <div class="grid grid-cols-6 gap-2 sm:grid-cols-10 md:grid-cols-12">{grid_html}</div>
  </section>

  <!-- Drill-down: one panel per problem, switched by the grid above -->
  <section class="mb-8" id="dd_section">
    <h2 class="mb-1 text-xl font-semibold">How one debate unfolds — <span id="dd_title">{escape(default_pid)}</span></h2>
    {dd_panels}
  </section>

  <footer class="border-t border-base-content/10 pt-4 text-center text-xs text-base-content/40">
    Generated by scripts/build_dashboard.py · regenerate after each run
  </footer>
</div>

<script>
  function showProblem(pid) {{
    document.querySelectorAll('.dd-panel').forEach(p => p.hidden = true);
    document.getElementById('dd_' + pid).hidden = false;
    document.getElementById('dd_title').textContent = pid;
    document.querySelectorAll('.pcell').forEach(b => b.classList.remove('ring-2', 'ring-primary'));
    document.getElementById('cell_' + pid).classList.add('ring-2', 'ring-primary');
    document.getElementById('dd_section').scrollIntoView({{behavior: 'smooth'}});
  }}
</script>
</body>
</html>"""


def build(run_path: Path) -> Path:
    """Render the dashboard for one run and return the written HTML path.

    Canonical location: next to the run's JSON/PNG in outputs/, named off the
    run file (hard_result.json -> hard_result_dashboard.html), so each run keeps
    its own dashboard. Uses the Tailwind/DaisyUI CDN, so viewing it needs
    network; for a fully offline copy run `npx lavish-axi export`.
    """
    data, problems = load(run_path)
    out_path = PROJECT_ROOT / "outputs" / f"{run_path.stem}_dashboard.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(render(data, problems))
    return out_path


def main() -> None:
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "outputs" / "hard_result.json"
    print(f"Wrote {build(run_path)}")


if __name__ == "__main__":
    main()

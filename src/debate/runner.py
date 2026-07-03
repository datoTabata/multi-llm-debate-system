"""Batch orchestration: select problems, run their debates concurrently, and
assemble metrics. This is library code (importable from notebooks or the
run_demo CLI); it does no file I/O of its own except reading the dataset.
"""

import asyncio
import json
import random
from pathlib import Path

from debate.evaluation import (
    calculate_accuracy,
    calculate_consensus_rate,
    calculate_improvement_rate,
    calculate_judge_accuracy_on_disagreements,
    calculate_majority_vote_accuracy,
    calculate_single_llm_accuracy,
    evaluate_result,
)
from debate.models import (
    AppConfig,
    OpenRouterClient,
    RunConfig,
    create_grader_client,
    create_model_clients,
    make_llm_grader,
)
from debate.pipeline import run_debate_for_problem
from debate.problem import Grader, Problem
from debate.schemas import DebateResult, EvaluationRecord, RunMetrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_problems(dataset: str = "problems.json") -> list[Problem]:
    """Load and validate problems from a dataset file under data/."""
    with (DATA_DIR / dataset).open(encoding="utf-8") as file:
        return [Problem(**problem) for problem in json.load(file)]


def select_problems(problems: list[Problem], run: RunConfig) -> list[Problem]:
    """Pick which problems to run.

    1. If `problem_ids` is non-empty, run exactly those (in listed order).
    2. Otherwise run a RANDOM sample of `problem_limit` (seeded if `seed` set).
    """
    if run.problem_ids:
        by_id = {problem.id: problem for problem in problems}
        missing = [pid for pid in run.problem_ids if pid not in by_id]
        if missing:
            raise ValueError(f"Unknown problem_ids in config: {missing}")
        return [by_id[pid] for pid in run.problem_ids]

    limit = min(run.problem_limit or len(problems), len(problems))
    return random.Random(run.seed).sample(problems, k=limit)


async def run_all_debates(
    problems: list[Problem],
    models: dict[str, OpenRouterClient],
    max_concurrent: int = 0,
) -> list[DebateResult | None]:
    """Run every problem's debate concurrently (capped by max_concurrent; 0 =
    all at once). Results keep input order; a failed problem yields None instead
    of killing the batch.
    """
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
    total = len(problems)
    solved = 0

    async def run_one(problem: Problem) -> DebateResult | None:
        nonlocal solved
        try:
            if semaphore is None:
                result = await run_debate_for_problem(problem, models)
            else:
                async with semaphore:
                    result = await run_debate_for_problem(problem, models)
        except Exception as error:  # one bad problem must not kill the batch
            solved += 1
            print(
                f"Solved {solved}/{total} problems  (FAILED {problem.id}: {type(error).__name__})",
                flush=True,
            )
            return None
        solved += 1
        print(f"Solved {solved}/{total} problems", flush=True)
        return result

    print(f"Running {total} problems concurrently...", flush=True)
    return await asyncio.gather(*(run_one(problem) for problem in problems))


def compute_metrics(
    results: list[DebateResult],
    evaluations: list[EvaluationRecord],
    problems_by_id: dict[str, Problem],
    grader: Grader | None = None,
) -> RunMetrics:
    """Assemble every system metric and baseline for a batch of results."""
    return RunMetrics(
        accuracy=calculate_accuracy(evaluations),
        single_llm_accuracy=calculate_single_llm_accuracy(results, problems_by_id, grader),
        majority_vote_accuracy=calculate_majority_vote_accuracy(results, problems_by_id, grader),
        consensus_rate=calculate_consensus_rate(results),
        judge_accuracy_on_disagreements=calculate_judge_accuracy_on_disagreements(
            results, problems_by_id, grader
        ),
        improvement_rate=calculate_improvement_rate(results, problems_by_id, grader),
    )


def run_and_evaluate(config: AppConfig, grader: Grader | None = None) -> dict:
    """Run the whole pipeline for the config's [run] selection and return a
    self-describing output dict (config / results / evaluations / metrics).

    No file I/O — the caller decides whether to save it. Callable from a
    notebook (apply nest_asyncio first) or from the run_demo CLI.
    """
    run = config.run
    problems = select_problems(load_problems(run.dataset), run)
    models = create_model_clients(config)
    if grader is None:
        grader = make_llm_grader(create_grader_client(config))
    problems_by_id = {problem.id: problem for problem in problems}

    raw_results = asyncio.run(run_all_debates(problems, models, run.max_concurrent_problems))

    succeeded = [(r, p) for r, p in zip(raw_results, problems, strict=True) if r is not None]
    failed = len(raw_results) - len(succeeded)
    if failed:
        print(f"WARNING: {failed} problem(s) failed and were excluded.", flush=True)

    results = [r for r, _ in succeeded]
    evaluations = [evaluate_result(r, p, grader) for r, p in succeeded]
    metrics = compute_metrics(results, evaluations, problems_by_id, grader)

    participants = {
        slot: {
            "model": client.model_name,
            "temperature": client.temperature,
            "max_tokens": client.max_tokens,
        }
        for slot, client in models.items()
    }
    return {
        "config": {
            "participants": participants,
            "dataset": run.dataset,
            "problem_ids": [problem.id for problem in problems],
            "seed": run.seed,
        },
        "results": [result.model_dump() for result in results],
        "evaluations": [evaluation.model_dump() for evaluation in evaluations],
        "metrics": metrics.model_dump(),
    }


def save_output(output: dict, output_file: str) -> Path:
    """Write an output dict to outputs/<output_file> and return the path."""
    output_path = OUTPUTS_DIR / output_file
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    return output_path

import asyncio
import json
import random
from pathlib import Path

from debate.evaluation import (
    calculate_accuracy,
    calculate_consensus_rate,
    calculate_improvement_rate,
    calculate_judge_accuracy_on_disagreements,
    calculate_single_llm_accuracy,
    evaluate_result,
)
from debate.models import (
    OpenRouterClient,
    RunConfig,
    create_grader_client,
    create_model_clients,
    load_config,
    make_llm_grader,
)
from debate.pipeline import run_debate_for_problem
from debate.problem import Problem
from debate.schemas import DebateResult, RunMetrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_problems(dataset: str = "problems.json") -> list[Problem]:
    """Load and validate problems from a dataset file under data/."""

    problems_path = PROJECT_ROOT / "data" / dataset

    with problems_path.open("r", encoding="utf-8") as file:
        return [Problem(**problem) for problem in json.load(file)]


def select_problems(problems: list[Problem], run: RunConfig) -> list[Problem]:
    """Pick which problems to run.

    Precedence:
    1. If `problem_ids` is non-empty, run exactly those (in listed order) and
       ignore `problem_limit`.
    2. Otherwise run a RANDOM sample of `problem_limit` problems. A `seed` makes
       the sample reproducible; omit it for a fresh sample each run.
    """

    if run.problem_ids:
        by_id = {problem.id: problem for problem in problems}
        missing = [pid for pid in run.problem_ids if pid not in by_id]
        if missing:
            raise SystemExit(f"Unknown problem_ids in config: {missing}")
        return [by_id[pid] for pid in run.problem_ids]

    limit = run.problem_limit or len(problems)
    limit = min(limit, len(problems))
    rng = random.Random(run.seed)
    return rng.sample(problems, k=limit)


async def run_all_debates(
    problems: list[Problem],
    models: dict[str, OpenRouterClient],
    max_concurrent: int = 0,
) -> list[DebateResult | None]:
    """Run every problem's debate concurrently.

    With max_concurrent <= 0 all problems run at once (9 rounds take ~the wall
    time of 1). Set a positive cap to throttle if OpenRouter rate-limits the
    burst. Results are returned in the same order as `problems`; a problem
    whose debate failed yields None instead of killing the batch.
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


def main() -> None:
    config = load_config()
    run = config.run

    problems = select_problems(load_problems(run.dataset), run)
    models = create_model_clients(config)
    grader = make_llm_grader(create_grader_client(config))

    problems_by_id = {problem.id: problem for problem in problems}

    raw_results = asyncio.run(run_all_debates(problems, models, run.max_concurrent_problems))

    # Drop problems whose debate failed (raw result is None) so the run still
    # produces metrics for the ones that succeeded.
    succeeded = [(r, p) for r, p in zip(raw_results, problems, strict=True) if r is not None]
    failed = len(raw_results) - len(succeeded)
    if failed:
        print(f"WARNING: {failed} problem(s) failed and were excluded.", flush=True)

    results = [r for r, _ in succeeded]
    evaluations = [evaluate_result(r, p, grader) for r, p in succeeded]

    metrics = RunMetrics(
        accuracy=calculate_accuracy(evaluations),
        single_llm_accuracy=calculate_single_llm_accuracy(results, problems_by_id, grader),
        consensus_rate=calculate_consensus_rate(results),
        judge_accuracy_on_disagreements=calculate_judge_accuracy_on_disagreements(
            results, problems_by_id, grader
        ),
        improvement_rate=calculate_improvement_rate(results, problems_by_id, grader),
    )

    output_path = PROJECT_ROOT / "outputs" / run.output_file
    output_path.parent.mkdir(exist_ok=True)

    participants_config = {
        slot: {
            "model": client.model_name,
            "temperature": client.temperature,
            "max_tokens": client.max_tokens,
        }
        for slot, client in models.items()
    }

    output = {
        "config": {
            "participants": participants_config,
            "dataset": run.dataset,
            "problem_ids": [problem.id for problem in problems],
            "seed": run.seed,
        },
        "results": [result.model_dump() for result in results],
        "evaluations": [evaluation.model_dump() for evaluation in evaluations],
        "metrics": metrics.model_dump(),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Saved result to {output_path}")
    print(f"Problems evaluated: {len(results)}")
    print(f"Accuracy (debate system): {metrics.accuracy:.2f}")
    print(f"Accuracy (single-LLM baseline): {metrics.single_llm_accuracy:.2f}")
    print(f"Consensus rate: {metrics.consensus_rate:.2f}")
    print(f"Judge accuracy on disagreements: {metrics.judge_accuracy_on_disagreements}")
    print(f"Improvement rate: {metrics.improvement_rate:.2f}")

    print("\nEvaluations:")
    print(json.dumps([evaluation.model_dump() for evaluation in evaluations], indent=2))


if __name__ == "__main__":
    main()

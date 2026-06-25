import asyncio
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from debate.evaluation import (
    calculate_accuracy,
    calculate_consensus_rate,
    calculate_improvement_rate,
    calculate_judge_accuracy_on_disagreements,
    evaluate_result,
)
from debate.models import (
    create_grader_client,
    create_model_clients,
    load_config,
    make_llm_grader,
)
from debate.pipeline import run_debate_for_problem
from debate.problem import Problem


def load_problems(dataset: str = "problems.json") -> list[dict]:
    """Load problems from a dataset file under data/."""

    problems_path = PROJECT_ROOT / "data" / dataset

    with problems_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_problems(problems: list[dict], run_config: dict) -> list[dict]:
    """Pick which problems to run.

    Precedence:
    1. If `problem_ids` is non-empty, run exactly those (in listed order) and
       ignore `problem_limit`.
    2. Otherwise run a RANDOM sample of `problem_limit` problems. A `seed` makes
       the sample reproducible; omit it for a fresh sample each run.
    """

    problem_ids = run_config.get("problem_ids") or []

    if problem_ids:
        by_id = {problem["id"]: problem for problem in problems}
        missing = [pid for pid in problem_ids if pid not in by_id]
        if missing:
            raise SystemExit(f"Unknown problem_ids in config: {missing}")
        return [by_id[pid] for pid in problem_ids]

    limit = run_config.get("problem_limit") or len(problems)
    limit = min(limit, len(problems))
    rng = random.Random(run_config.get("seed"))
    return rng.sample(problems, k=limit)


async def run_all_debates(problems: list[dict], models: dict, max_concurrent: int = 0) -> list[dict]:
    """Run every problem's debate concurrently.

    With max_concurrent <= 0 all problems run at once (9 rounds take ~the wall
    time of 1). Set a positive cap to throttle if OpenRouter rate-limits the
    burst. Results are returned in the same order as `problems`.
    """

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None

    async def run_one(problem: dict) -> dict:
        if semaphore is None:
            return await run_debate_for_problem(problem, models)
        async with semaphore:
            return await run_debate_for_problem(problem, models)

    return await asyncio.gather(*(run_one(problem) for problem in problems))


def main() -> None:
    config = load_config()
    run_config = config.get("run", {})

    dataset = run_config.get("dataset", "problems.json")
    problems = select_problems(load_problems(dataset), run_config)
    models = create_model_clients(config)
    grader = make_llm_grader(create_grader_client(config))

    problem_objs = [Problem(**problem) for problem in problems]
    problems_by_id = {problem.id: problem for problem in problem_objs}

    max_concurrent = run_config.get("max_concurrent_problems", 0)
    results = asyncio.run(run_all_debates(problems, models, max_concurrent))
    evaluations = [
        evaluate_result(result, problem_obj, grader)
        for result, problem_obj in zip(results, problem_objs)
    ]

    accuracy = calculate_accuracy(evaluations)
    consensus_rate = calculate_consensus_rate(results)
    judge_accuracy_on_disagreements = calculate_judge_accuracy_on_disagreements(
        results,
        problems_by_id,
        grader,
    )
    improvement_rate = calculate_improvement_rate(results, problems_by_id, grader)

    metrics = {
        "accuracy": accuracy,
        "consensus_rate": consensus_rate,
        "judge_accuracy_on_disagreements": judge_accuracy_on_disagreements,
        "improvement_rate": improvement_rate,
    }

    output_file = run_config.get("output_file", "demo_result.json")
    output_path = PROJECT_ROOT / "outputs" / output_file
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
            "problem_ids": [problem["id"] for problem in problems],
            "seed": run_config.get("seed"),
        },
        "results": results,
        "evaluations": evaluations,
        "metrics": metrics,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Saved result to {output_path}")
    print(f"Problems evaluated: {len(problems)}")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Consensus rate: {metrics['consensus_rate']:.2f}")
    print(f"Judge accuracy on disagreements: {metrics['judge_accuracy_on_disagreements']}")
    print(f"Improvement rate: {metrics['improvement_rate']:.2f}")

    print("\nEvaluations:")
    print(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()
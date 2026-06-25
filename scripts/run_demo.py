import json
import os
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
from debate.models import create_model_clients
from debate.pipeline import run_debate_for_problem


def load_problems() -> list[dict]:
    """Load problems from the dataset file."""

    problems_path = PROJECT_ROOT / "data" / "problems.json"

    with problems_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    problems = load_problems()
    problem_limit = int(os.getenv("PROBLEM_LIMIT", len(problems)))
    problems = problems[:problem_limit]
    models = create_model_clients()

    results = []
    evaluations = []

    for problem in problems:
        result = run_debate_for_problem(problem, models)
        evaluation = evaluate_result(result, problem)

        results.append(result)
        evaluations.append(evaluation)

    accuracy = calculate_accuracy(evaluations)
    consensus_rate = calculate_consensus_rate(results)
    problems_by_id = {problem["id"]: problem for problem in problems}
    judge_accuracy_on_disagreements = calculate_judge_accuracy_on_disagreements(
        results,
        problems_by_id,
    )
    improvement_rate = calculate_improvement_rate(results, problems_by_id)

    metrics = {
        "accuracy": accuracy,
        "consensus_rate": consensus_rate,
        "judge_accuracy_on_disagreements": judge_accuracy_on_disagreements,
        "improvement_rate": improvement_rate,
    }

    output_file = os.getenv("OUTPUT_FILE", "demo_result.json")
    output_path = PROJECT_ROOT / "outputs" / output_file
    output_path.parent.mkdir(exist_ok=True)

    output = {
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
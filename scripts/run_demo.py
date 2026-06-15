import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from debate.evaluation import evaluate_result
from debate.models import create_model_clients
from debate.pipeline import run_debate_for_problem


def load_problems() -> list[dict]:
    """Load problems from the dataset file."""

    problems_path = PROJECT_ROOT / "data" / "problems.json"

    with problems_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    problems = load_problems()
    problem = problems[0]

    models = create_model_clients()
    result = run_debate_for_problem(problem, models)

    output_path = PROJECT_ROOT / "outputs" / "demo_result.json"
    output_path.parent.mkdir(exist_ok=True)

    evaluation = evaluate_result(result, problem)

    output = {
        "result": result,
        "evaluation": evaluation,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Saved result to {output_path}")

    print("Debate result:")
    print(json.dumps(result, indent=2))

    print("\nEvaluation:")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
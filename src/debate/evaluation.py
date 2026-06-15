import json


def normalize_answer(answer: str) -> str:
    """Normalize an answer before comparison."""

    return str(answer).strip().lower()


def exact_match(predicted_answer: str, correct_answer: str) -> bool:
    """Check whether two answers match exactly after normalization."""

    return normalize_answer(predicted_answer) == normalize_answer(correct_answer)


def parse_json_response(response: str) -> dict:
    """Parse a JSON model response."""

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def extract_winner(judgment: str) -> str:
    """Extract the winning solver ID from a judge response."""

    data = parse_json_response(judgment)
    return data.get("winner", "")


def extract_refined_answer(refined_solution: str) -> str:
    """Extract the final answer from a refined solution response."""

    data = parse_json_response(refined_solution)
    return data.get("refined_answer", "")


def evaluate_result(result: dict, problem: dict) -> dict:
    """Evaluate one debate result against the correct answer."""

    winner = extract_winner(result["judgment"])
    refined_solution = result["refined_solutions"].get(winner, "")
    predicted_answer = extract_refined_answer(refined_solution)
    correct_answer = problem["correct_answer"]

    is_correct = exact_match(predicted_answer, correct_answer)

    return {
        "problem_id": problem["id"],
        "winner": winner,
        "predicted_answer": predicted_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
    }


def calculate_accuracy(evaluations: list[dict]) -> float:
    """Calculate accuracy from evaluation records."""

    if not evaluations:
        return 0.0

    correct_count = sum(1 for evaluation in evaluations if evaluation["is_correct"])

    return correct_count / len(evaluations)


def get_solver_refined_answers(result: dict) -> dict[str, str]:
    """Extract refined answers for all solvers in one result."""

    answers = {}

    for solver_id, refined_solution in result["refined_solutions"].items():
        answers[solver_id] = extract_refined_answer(refined_solution)

    return answers


def has_consensus(result: dict) -> bool:
    """Check whether all solvers produced the same refined answer."""

    answers = get_solver_refined_answers(result)
    unique_answers = set(answers.values())

    return len(unique_answers) == 1


def calculate_consensus_rate(results: list[dict]) -> float:
    """Calculate the rate of full solver consensus."""

    if not results:
        return 0.0

    consensus_count = sum(1 for result in results if has_consensus(result))

    return consensus_count / len(results)
import json


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


def extract_initial_answer(solution: str) -> str:
    """Extract the final answer from an initial solver response."""

    data = parse_json_response(solution)
    return data.get("answer", "")


def extract_refined_answer(refined_solution: str) -> str:
    """Extract the final answer from a refined solution response."""

    data = parse_json_response(refined_solution)
    return data.get("refined_answer", "")


def evaluate_result(result: dict, problem, grader=None) -> dict:
    """Evaluate one debate result against the correct answer.

    `problem` is a Problem instance; grading is delegated to `problem.check`,
    which uses the LLM `grader` only for free-answer problems.
    """

    winner = extract_winner(result["judgment"])
    refined_solution = result["refined_solutions"].get(winner, "")
    predicted_answer = extract_refined_answer(refined_solution)

    is_correct = problem.check(predicted_answer, grader)

    return {
        "problem_id": problem.id,
        "answer_type": problem.answer_type,
        "winner": winner,
        "predicted_answer": predicted_answer,
        "correct_answer": problem.correct_answer,
        "is_correct": is_correct,
    }


def calculate_accuracy(evaluations: list[dict]) -> float:
    """Calculate accuracy from evaluation records."""

    if not evaluations:
        return 0.0

    correct_count = sum(1 for evaluation in evaluations if evaluation["is_correct"])

    return correct_count / len(evaluations)


def calculate_single_llm_accuracy(
    results: list[dict],
    problems_by_id: dict,
    grader=None,
) -> float:
    """Single-LLM baseline: accuracy of each solver's INDEPENDENT (pre-debate)
    answer, averaged over all solo attempts.

    This is the "one model solving alone" baseline the debate is measured
    against. It reuses the independent solutions already in each result, so it
    needs no extra debate calls (only grader calls for free-answer problems).
    """

    total = 0
    correct = 0

    for result in results:
        problem = problems_by_id[result["problem_id"]]

        for solution in result["solutions"].values():
            answer = extract_initial_answer(solution)
            total += 1
            if problem.check(answer, grader):
                correct += 1

    if total == 0:
        return 0.0

    return correct / total


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


def has_disagreement(result: dict) -> bool:
    """Check whether solvers produced different refined answers."""

    return not has_consensus(result)


def judge_selected_correct_answer(result: dict, problem, grader=None) -> bool:
    """Check whether the judge selected a solver with the correct answer."""

    winner = extract_winner(result["judgment"])
    refined_solution = result["refined_solutions"].get(winner, "")
    predicted_answer = extract_refined_answer(refined_solution)

    return problem.check(predicted_answer, grader)


def solver_improved(solution: str, refined_solution: str, problem, grader=None) -> bool:
    """Check whether refinement changed an incorrect answer into a correct one."""

    initial_answer = extract_initial_answer(solution)
    refined_answer = extract_refined_answer(refined_solution)

    initially_correct = problem.check(initial_answer, grader)
    finally_correct = problem.check(refined_answer, grader)

    return not initially_correct and finally_correct


def calculate_judge_accuracy_on_disagreements(
    results: list[dict],
    problems_by_id: dict,
    grader=None,
) -> float | None:
    """Calculate judge accuracy only for problems with solver disagreement."""

    disagreement_results = [result for result in results if has_disagreement(result)]

    if not disagreement_results:
        return None

    correct_count = 0

    for result in disagreement_results:
        problem = problems_by_id[result["problem_id"]]

        if judge_selected_correct_answer(result, problem, grader):
            correct_count += 1

    return correct_count / len(disagreement_results)


def calculate_improvement_rate(
    results: list[dict],
    problems_by_id: dict,
    grader=None,
) -> float:
    """Calculate how often refinement improved solver answers."""

    total_solver_attempts = 0
    improved_count = 0

    for result in results:
        problem = problems_by_id[result["problem_id"]]

        for solver_id, solution in result["solutions"].items():
            refined_solution = result["refined_solutions"][solver_id]
            total_solver_attempts += 1

            if solver_improved(solution, refined_solution, problem, grader):
                improved_count += 1

    if total_solver_attempts == 0:
        return 0.0

    return improved_count / total_solver_attempts
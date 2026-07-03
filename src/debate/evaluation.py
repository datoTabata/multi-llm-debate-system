from collections import Counter

from debate.problem import Grader, Problem, _normalize, _to_number
from debate.schemas import (
    DebateResult,
    EvaluationRecord,
    RefinementResponse,
    SolutionResponse,
)


def evaluate_result(
    result: DebateResult,
    problem: Problem,
    grader: Grader | None = None,
) -> EvaluationRecord:
    """Evaluate one debate result against the correct answer.

    Grading is delegated to `problem.check`, which uses the LLM `grader` only
    for free-answer problems.
    """

    winner = result.judgment.winner
    predicted_answer = result.refined_solutions[winner].refined_answer

    return EvaluationRecord(
        problem_id=problem.id,
        answer_type=problem.answer_type,
        winner=winner,
        predicted_answer=predicted_answer,
        correct_answer=problem.correct_answer,
        is_correct=problem.check(predicted_answer, grader),
    )


def calculate_accuracy(evaluations: list[EvaluationRecord]) -> float:
    """Calculate accuracy from evaluation records."""

    if not evaluations:
        return 0.0

    correct_count = sum(1 for evaluation in evaluations if evaluation.is_correct)

    return correct_count / len(evaluations)


def calculate_single_llm_accuracy(
    results: list[DebateResult],
    problems_by_id: dict[str, Problem],
    grader: Grader | None = None,
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
        problem = problems_by_id[result.problem_id]

        for solution in result.solutions.values():
            total += 1
            if problem.check(solution.answer, grader):
                correct += 1

    if total == 0:
        return 0.0

    return correct / total


def _vote_key(problem: Problem, answer: str) -> tuple:
    """Group equivalent answers for majority voting. Numbers vote by parsed
    value (so '23' and '23.0' agree); everything else by normalized string."""
    if problem.answer_type == "number":
        value = _to_number(answer)
        if value is not None:
            return ("num", round(value, 6))
    return ("str", _normalize(answer))


def calculate_majority_vote_accuracy(
    results: list[DebateResult],
    problems_by_id: dict[str, Problem],
    grader: Grader | None = None,
) -> float:
    """Simple-voting baseline: from the three solvers' INDEPENDENT (pre-debate)
    answers, pick the majority answer and grade it.

    This is the assignment's second baseline (no refinement, no judge). Ties
    where all three answers differ fall back to the first solver's answer. Like
    the single-LLM baseline it reuses answers already in each result, so it
    needs no extra debate calls.
    """
    total = 0
    correct = 0

    for result in results:
        problem = problems_by_id[result.problem_id]
        answers = [s.answer for s in result.solutions.values()]

        counts = Counter(_vote_key(problem, a) for a in answers)
        winning_key, _ = counts.most_common(1)[0]
        # Grade a representative answer for the winning group (first in solver
        # order), so grading stays type-aware even though voting is by key.
        majority = next(a for a in answers if _vote_key(problem, a) == winning_key)

        total += 1
        if problem.check(majority, grader):
            correct += 1

    return correct / total if total else 0.0


def get_solver_refined_answers(result: DebateResult) -> dict[str, str]:
    """Extract refined answers for all solvers in one result."""

    return {
        solver_id: refined.refined_answer for solver_id, refined in result.refined_solutions.items()
    }


def has_consensus(result: DebateResult) -> bool:
    """Check whether all solvers produced the same refined answer."""

    answers = get_solver_refined_answers(result)
    unique_answers = set(answers.values())

    return len(unique_answers) == 1


def calculate_consensus_rate(results: list[DebateResult]) -> float:
    """Calculate the rate of full solver consensus."""

    if not results:
        return 0.0

    consensus_count = sum(1 for result in results if has_consensus(result))

    return consensus_count / len(results)


def has_disagreement(result: DebateResult) -> bool:
    """Check whether solvers produced different refined answers."""

    return not has_consensus(result)


def judge_selected_correct_answer(
    result: DebateResult,
    problem: Problem,
    grader: Grader | None = None,
) -> bool:
    """Check whether the judge selected a solver with the correct answer."""

    return evaluate_result(result, problem, grader).is_correct


def solver_improved(
    solution: SolutionResponse,
    refined_solution: RefinementResponse,
    problem: Problem,
    grader: Grader | None = None,
) -> bool:
    """Check whether refinement changed an incorrect answer into a correct one."""

    initially_correct = problem.check(solution.answer, grader)
    finally_correct = problem.check(refined_solution.refined_answer, grader)

    return not initially_correct and finally_correct


def calculate_judge_accuracy_on_disagreements(
    results: list[DebateResult],
    problems_by_id: dict[str, Problem],
    grader: Grader | None = None,
) -> float | None:
    """Calculate judge accuracy only for problems with solver disagreement."""

    disagreement_results = [result for result in results if has_disagreement(result)]

    if not disagreement_results:
        return None

    correct_count = 0

    for result in disagreement_results:
        problem = problems_by_id[result.problem_id]

        if judge_selected_correct_answer(result, problem, grader):
            correct_count += 1

    return correct_count / len(disagreement_results)


def calculate_improvement_rate(
    results: list[DebateResult],
    problems_by_id: dict[str, Problem],
    grader: Grader | None = None,
) -> float:
    """Calculate how often refinement improved solver answers."""

    total_solver_attempts = 0
    improved_count = 0

    for result in results:
        problem = problems_by_id[result.problem_id]

        for solver_id, solution in result.solutions.items():
            refined_solution = result.refined_solutions[solver_id]
            total_solver_attempts += 1

            if solver_improved(solution, refined_solution, problem, grader):
                improved_count += 1

    if total_solver_attempts == 0:
        return 0.0

    return improved_count / total_solver_attempts

import json

from debate.prompts import (
    build_judge_prompt,
    build_peer_review_prompt,
    build_refinement_prompt,
    build_role_preference_prompt,
    build_solver_prompt,
)
from debate.schemas import (
    JudgeResponse,
    RefinementResponse,
    RolePreferenceResponse,
    SolutionResponse,
)


def collect_role_preferences(problem: dict, models: dict) -> dict[str, str]:
    """Collect role preferences from every model."""

    preferences = {}

    for model_name, model in models.items():
        prompt = build_role_preference_prompt(problem)
        preference = model.generate_structured(prompt, RolePreferenceResponse)
        preferences[model_name] = preference.model_dump_json()

    return preferences


def generate_independent_solutions(
    problem: dict,
    models: dict,
    roles: dict[str, str],
) -> dict[str, str]:
    """Generate independent solutions from the solver models."""

    solutions = {}

    for solver_id in ["solver_1", "solver_2", "solver_3"]:
        model_name = roles[solver_id]
        model = models[model_name]
        prompt = build_solver_prompt(problem)
        solution = model.generate_structured(prompt, SolutionResponse)
        solutions[solver_id] = solution.model_dump_json()

    return solutions


def assign_roles(models: dict, role_preferences: dict[str, str]) -> dict[str, str]:
    """Assign three solver roles and one judge role using role preferences."""

    judge_scores = {}

    for model_name in models:
        try:
            preference_data = json.loads(role_preferences[model_name])
            judge_scores[model_name] = preference_data.get("judge_confidence", 0)
        except json.JSONDecodeError:
            judge_scores[model_name] = 0

    judge_model = max(judge_scores, key=judge_scores.get)
    solver_models = [model_name for model_name in models if model_name != judge_model]

    return {
        "solver_1": solver_models[0],
        "solver_2": solver_models[1],
        "solver_3": solver_models[2],
        "judge": judge_model,
    }


def generate_peer_reviews(
    problem: dict,
    models: dict,
    roles: dict[str, str],
    solutions: dict[str, str],
) -> dict[str, list[str]]:
    """Generate peer reviews for each solver solution."""

    reviews = {solver_id: [] for solver_id in solutions}

    for reviewer_id in solutions:
        reviewer_model_name = roles[reviewer_id]
        reviewer_model = models[reviewer_model_name]

        for solution_id, solution in solutions.items():
            if reviewer_id == solution_id:
                continue

            prompt = build_peer_review_prompt(problem, solution_id, solution)
            review = reviewer_model.generate(prompt)
            reviews[solution_id].append(review)

    return reviews


def refine_solutions(
    problem: dict,
    models: dict,
    roles: dict[str, str],
    solutions: dict[str, str],
    peer_reviews: dict[str, list[str]],
) -> dict[str, str]:
    """Refine solver solutions using peer reviews."""

    refined_solutions = {}

    for solver_id, original_solution in solutions.items():
        model_name = roles[solver_id]
        model = models[model_name]
        reviews = peer_reviews[solver_id]
        prompt = build_refinement_prompt(problem, original_solution, reviews)
        refined_solution = model.generate_structured(prompt, RefinementResponse)
        refined_solutions[solver_id] = refined_solution.model_dump_json()

    return refined_solutions


def judge_solutions(
    problem: dict,
    models: dict,
    roles: dict[str, str],
    solutions: dict[str, str],
    peer_reviews: dict[str, list[str]],
    refined_solutions: dict[str, str],
) -> str:
    """Judge refined solutions and choose the best answer."""

    judge_model_name = roles["judge"]
    judge_model = models[judge_model_name]

    prompt = build_judge_prompt(
        problem,
        solutions,
        peer_reviews,
        refined_solutions,
    )

    judgment = judge_model.generate_structured(prompt, JudgeResponse)
    return judgment.model_dump_json()


def run_debate_for_problem(problem: dict, models: dict) -> dict:
    """Run the full debate workflow for one problem."""

    role_preferences = collect_role_preferences(problem, models)
    roles = assign_roles(models, role_preferences)
    solutions = generate_independent_solutions(problem, models, roles)
    peer_reviews = generate_peer_reviews(problem, models, roles, solutions)
    refined_solutions = refine_solutions(problem, models, roles, solutions, peer_reviews)
    judgment = judge_solutions(
        problem,
        models,
        roles,
        solutions,
        peer_reviews,
        refined_solutions,
    )

    return {
        "problem_id": problem["id"],
        "role_preferences": role_preferences,
        "roles": roles,
        "solutions": solutions,
        "peer_reviews": peer_reviews,
        "refined_solutions": refined_solutions,
        "judgment": judgment,
    }
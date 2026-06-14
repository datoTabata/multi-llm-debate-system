import json

from debate.prompts import (
    build_peer_review_prompt,
    build_role_preference_prompt,
    build_solver_prompt,
)

def collect_role_preferences(problem: dict, models: dict) -> dict[str, str]:
    """Collect role preferences from every model."""

    preferences = {}

    for model_name, model in models.items():
        prompt = build_role_preference_prompt(problem)
        preferences[model_name] = model.generate(prompt)

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
        solutions[solver_id] = model.generate(prompt)

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
from debate.prompts import build_role_preference_prompt, build_solver_prompt


def collect_role_preferences(problem: dict, models: dict) -> dict[str, str]:
    """Collect role preferences from every model."""

    preferences = {}

    for model_name, model in models.items():
        prompt = build_role_preference_prompt(problem)
        preferences[model_name] = model.generate(prompt)

    return preferences


def generate_independent_solutions(problem: dict, models: dict) -> dict[str, str]:
    """Generate independent solutions from every model."""

    solutions = {}

    for model_name, model in models.items():
        prompt = build_solver_prompt(problem)
        solutions[model_name] = model.generate(prompt)

    return solutions
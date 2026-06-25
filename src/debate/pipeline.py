import asyncio
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

SOLVER_IDS = ["solver_1", "solver_2", "solver_3"]


async def collect_role_preferences(problem: dict, models: dict) -> dict[str, str]:
    """Collect role preferences from every model, concurrently."""

    async def preference_for(model_name: str, model) -> tuple[str, str]:
        peers = [other.model_name for slot, other in models.items() if slot != model_name]
        prompt = build_role_preference_prompt(problem, model.model_name, peers)
        preference = await model.agenerate_structured(prompt, RolePreferenceResponse)
        return model_name, preference.model_dump_json()

    pairs = await asyncio.gather(
        *(preference_for(name, model) for name, model in models.items())
    )

    return dict(pairs)


async def generate_independent_solutions(
    problem: dict,
    models: dict,
    roles: dict[str, str],
) -> dict[str, str]:
    """Generate independent solutions from the solver models, concurrently."""

    async def solve(solver_id: str) -> tuple[str, str]:
        model = models[roles[solver_id]]
        prompt = build_solver_prompt(problem)
        solution = await model.agenerate_structured(prompt, SolutionResponse)
        return solver_id, solution.model_dump_json()

    pairs = await asyncio.gather(*(solve(solver_id) for solver_id in SOLVER_IDS))

    return dict(pairs)


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


async def generate_peer_reviews(
    problem: dict,
    models: dict,
    roles: dict[str, str],
    solutions: dict[str, str],
) -> dict[str, list[str]]:
    """Generate peer reviews for each solver solution, all reviews concurrently."""

    pairs = [
        (reviewer_id, solution_id)
        for reviewer_id in solutions
        for solution_id in solutions
        if reviewer_id != solution_id
    ]

    async def review(reviewer_id: str, solution_id: str) -> tuple[str, str]:
        reviewer_model = models[roles[reviewer_id]]
        prompt = build_peer_review_prompt(problem, solution_id, solutions[solution_id])
        text = await reviewer_model.agenerate(prompt)
        return solution_id, text

    results = await asyncio.gather(*(review(r, s) for r, s in pairs))

    reviews = {solution_id: [] for solution_id in solutions}
    for solution_id, text in results:
        reviews[solution_id].append(text)

    return reviews


async def refine_solutions(
    problem: dict,
    models: dict,
    roles: dict[str, str],
    solutions: dict[str, str],
    peer_reviews: dict[str, list[str]],
) -> dict[str, str]:
    """Refine solver solutions using peer reviews, concurrently."""

    async def refine(solver_id: str, original_solution: str) -> tuple[str, str]:
        model = models[roles[solver_id]]
        prompt = build_refinement_prompt(problem, original_solution, peer_reviews[solver_id])
        refined = await model.agenerate_structured(prompt, RefinementResponse)
        return solver_id, refined.model_dump_json()

    pairs = await asyncio.gather(
        *(refine(solver_id, solution) for solver_id, solution in solutions.items())
    )

    return dict(pairs)


async def judge_solutions(
    problem: dict,
    models: dict,
    roles: dict[str, str],
    solutions: dict[str, str],
    peer_reviews: dict[str, list[str]],
    refined_solutions: dict[str, str],
) -> str:
    """Judge refined solutions and choose the best answer."""

    judge_model = models[roles["judge"]]

    prompt = build_judge_prompt(
        problem,
        solutions,
        peer_reviews,
        refined_solutions,
    )

    judgment = await judge_model.agenerate_structured(prompt, JudgeResponse)
    return judgment.model_dump_json()


async def run_debate_for_problem(problem: dict, models: dict) -> dict:
    """Run the full debate workflow for one problem.

    Stages run in sequence (each depends on the previous), but the independent
    model calls *within* a stage run concurrently via asyncio.gather.
    """

    role_preferences = await collect_role_preferences(problem, models)
    roles = assign_roles(models, role_preferences)
    solutions = await generate_independent_solutions(problem, models, roles)
    peer_reviews = await generate_peer_reviews(problem, models, roles, solutions)
    refined_solutions = await refine_solutions(
        problem, models, roles, solutions, peer_reviews
    )
    judgment = await judge_solutions(
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

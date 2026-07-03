import asyncio

from debate.models import OpenRouterClient
from debate.problem import Problem
from debate.prompts import (
    build_judge_prompt,
    build_peer_review_prompt,
    build_refinement_prompt,
    build_role_preference_prompt,
    build_solver_prompt,
)
from debate.schemas import (
    DebateResult,
    JudgeResponse,
    RefinementResponse,
    RolePreferenceResponse,
    SolutionResponse,
)

SOLVER_IDS = ["solver_1", "solver_2", "solver_3"]

Models = dict[str, OpenRouterClient]


async def collect_role_preferences(
    problem: Problem,
    models: Models,
) -> dict[str, RolePreferenceResponse]:
    """Stage 0: ask each of the four models whether it wants to Solve or Judge.

    Each model sees the problem, its own OpenRouter id, and the ids of its three
    peers, so it can weigh its strengths against the actual roster. All four
    calls run concurrently.

    Returns {slot_name: RolePreferenceResponse} — the raw material
    `assign_roles` uses to pick the judge.
    """

    async def preference_for(
        slot_name: str, model: OpenRouterClient
    ) -> tuple[str, RolePreferenceResponse]:
        # The peer roster is the other three models' ids (not slot names), so
        # the prompt names real models, e.g. "anthropic/claude-haiku-4.5".
        peers = [other.model_name for slot, other in models.items() if slot != slot_name]
        prompt = build_role_preference_prompt(problem, model.model_name, peers)
        preference = await model.agenerate_structured(prompt, RolePreferenceResponse)
        return slot_name, preference

    # Fan out all four preference calls at once; gather preserves nothing about
    # ordering we rely on, since results carry their slot name with them.
    pairs = await asyncio.gather(*(preference_for(name, model) for name, model in models.items()))

    return dict(pairs)


def assign_roles(
    models: Models,
    role_preferences: dict[str, RolePreferenceResponse],
) -> dict[str, str]:
    """Assign three solver roles and one judge role using role preferences.

    The slot with the highest self-reported judge_confidence becomes the judge;
    the remaining three become solvers. Returns {role: slot_name}.
    """

    judge_scores = {slot: role_preferences[slot].judge_confidence for slot in models}

    judge_model = max(judge_scores, key=judge_scores.get)
    solver_models = [slot for slot in models if slot != judge_model]

    return {
        "solver_1": solver_models[0],
        "solver_2": solver_models[1],
        "solver_3": solver_models[2],
        "judge": judge_model,
    }


async def generate_independent_solutions(
    problem: Problem,
    models: Models,
    roles: dict[str, str],
) -> dict[str, SolutionResponse]:
    """Generate independent solutions from the solver models, concurrently."""

    async def solve(solver_id: str) -> tuple[str, SolutionResponse]:
        model = models[roles[solver_id]]
        prompt = build_solver_prompt(problem)
        solution = await model.agenerate_structured(prompt, SolutionResponse)
        return solver_id, solution

    pairs = await asyncio.gather(*(solve(solver_id) for solver_id in SOLVER_IDS))

    return dict(pairs)


async def generate_peer_reviews(
    problem: Problem,
    models: Models,
    roles: dict[str, str],
    solutions: dict[str, SolutionResponse],
) -> dict[str, list[str]]:
    """Generate peer reviews for each solver solution, all reviews concurrently.

    Every solver reviews the other two solutions (6 reviews total). Reviews are
    free text — this is the one unstructured stage. Returns
    {solution_id: [review, review]} grouped by the solution being reviewed.
    """

    pairs = [
        (reviewer_id, solution_id)
        for reviewer_id in solutions
        for solution_id in solutions
        if reviewer_id != solution_id
    ]

    async def review(reviewer_id: str, solution_id: str) -> tuple[str, str]:
        reviewer_model = models[roles[reviewer_id]]
        prompt = build_peer_review_prompt(
            problem, solution_id, solutions[solution_id].model_dump_json()
        )
        text = await reviewer_model.agenerate(prompt)
        return solution_id, text

    results = await asyncio.gather(*(review(r, s) for r, s in pairs))

    reviews = {solution_id: [] for solution_id in solutions}
    for solution_id, text in results:
        reviews[solution_id].append(text)

    return reviews


async def refine_solutions(
    problem: Problem,
    models: Models,
    roles: dict[str, str],
    solutions: dict[str, SolutionResponse],
    peer_reviews: dict[str, list[str]],
) -> dict[str, RefinementResponse]:
    """Refine solver solutions using peer reviews, concurrently."""

    async def refine(
        solver_id: str, original_solution: SolutionResponse
    ) -> tuple[str, RefinementResponse]:
        model = models[roles[solver_id]]
        prompt = build_refinement_prompt(
            problem, original_solution.model_dump_json(), peer_reviews[solver_id]
        )
        refined = await model.agenerate_structured(prompt, RefinementResponse)
        return solver_id, refined

    pairs = await asyncio.gather(
        *(refine(solver_id, solution) for solver_id, solution in solutions.items())
    )

    return dict(pairs)


async def judge_solutions(
    problem: Problem,
    models: Models,
    roles: dict[str, str],
    solutions: dict[str, SolutionResponse],
    peer_reviews: dict[str, list[str]],
    refined_solutions: dict[str, RefinementResponse],
) -> JudgeResponse:
    """Judge refined solutions and choose the best answer."""

    judge_model = models[roles["judge"]]

    # The judge prompt interpolates these dicts as text, so serialize the
    # schema objects here at the prompt boundary.
    prompt = build_judge_prompt(
        problem,
        {sid: s.model_dump_json() for sid, s in solutions.items()},
        peer_reviews,
        {sid: r.model_dump_json() for sid, r in refined_solutions.items()},
    )

    return await judge_model.agenerate_structured(prompt, JudgeResponse)


async def run_debate_for_problem(problem: Problem, models: Models) -> DebateResult:
    """Run the full debate workflow for one problem.

    Stages run in sequence (each depends on the previous), but the independent
    model calls *within* a stage run concurrently via asyncio.gather.
    """

    role_preferences = await collect_role_preferences(problem, models)
    roles = assign_roles(models, role_preferences)
    solutions = await generate_independent_solutions(problem, models, roles)
    peer_reviews = await generate_peer_reviews(problem, models, roles, solutions)
    refined_solutions = await refine_solutions(problem, models, roles, solutions, peer_reviews)
    judgment = await judge_solutions(
        problem,
        models,
        roles,
        solutions,
        peer_reviews,
        refined_solutions,
    )

    return DebateResult(
        problem_id=problem.id,
        role_preferences=role_preferences,
        roles=roles,
        solutions=solutions,
        peer_reviews=peer_reviews,
        refined_solutions=refined_solutions,
        judgment=judgment,
    )

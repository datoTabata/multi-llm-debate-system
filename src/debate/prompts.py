def build_role_preference_prompt(problem: dict) -> str:
    """Build a prompt for role preference selection."""

    return f"""
You are participating in a collaborative debate system.

Problem:
{problem["question"]}

Choose whether you would be better as a Solver or Judge for this problem.
Return your answer with:
- preferred role
- confidence
- short reasoning
"""


def build_solver_prompt(problem: dict) -> str:
    """Build a prompt for independent solution generation."""

    return f"""
Solve the following problem independently.

Problem:
{problem["question"]}

Return:
- step-by-step reasoning
- final answer
- confidence
"""


def build_peer_review_prompt(problem: dict, solution_id: str, solution: str) -> str:
    """Build a prompt for peer review."""

    return f"""
Review the following solution.

Problem:
{problem["question"]}

Solution ID:
{solution_id}

Solution:
{solution}

Return:
- strengths
- weaknesses
- possible errors
- suggested changes
- overall assessment
"""


def build_refinement_prompt(problem: dict, original_solution: str, reviews: list[str]) -> str:
    """Build a prompt for solution refinement."""

    joined_reviews = "\n\n".join(reviews)

    return f"""
Refine your solution using the peer reviews.

Problem:
{problem["question"]}

Original solution:
{original_solution}

Peer reviews:
{joined_reviews}

Return:
- changes made
- critiques accepted or rejected
- refined solution
- refined final answer
- confidence
"""


def build_judge_prompt(
    problem: dict,
    solutions: dict[str, str],
    peer_reviews: dict[str, list[str]],
    refined_solutions: dict[str, str],
) -> str:
    """Build a prompt for final judgment."""

    return f"""
Judge the final solutions.

Problem:
{problem["question"]}

Original solutions:
{solutions}

Peer reviews:
{peer_reviews}

Refined solutions:
{refined_solutions}

Return:
- winner
- confidence
- reasoning
"""
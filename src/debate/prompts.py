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
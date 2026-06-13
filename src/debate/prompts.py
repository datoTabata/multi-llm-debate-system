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
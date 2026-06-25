from datetime import date


def build_role_preference_prompt(
    problem: dict,
    model_identity: str,
    peers: list[str] | None = None,
    today: str | None = None,
) -> str:
    """Build a prompt for role preference selection.

    `model_identity` is the model's own OpenRouter id (e.g.
    "anthropic/claude-haiku-4.5") and `peers` are the other models' ids, so it
    can reason about its own likely strengths relative to the named roster.
    """

    today = today or date.today().isoformat()
    peer_list = "\n".join(f"- {peer}" for peer in peers) if peers else "- (unknown)"

    return f"""
You are the model "{model_identity}", taking part in a collaborative debate
system alongside three other models, each from a different provider.

The other participants in this debate are:
{peer_list}

How the debate works: all four models first pick whether they want to Solve or
Judge. Three become Solvers and answer the problem independently; one becomes the
Judge. Solvers then peer-review each other, refine their answers, and finally the
Judge picks the best refined answer.

Today's date is {today}. Your training knowledge has a cutoff in the past, so
newer and more capable models may have been released since then that you are
unaware of. Do not assume you are the newest or strongest model in the group.

Problem:
{problem["question"]}

Reason about your own likely strengths and weaknesses as "{model_identity}" on
this specific kind of problem, relative to the named peer models above, and
choose whether you would be more useful as a Solver or as the Judge.

Return your answer with:
- preferred role
- confidence
- short reasoning

Important:
preferred_role must be exactly "Solver" or "Judge".
solver_confidence and judge_confidence must be numbers between 0 and 1.
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

Important:
The answer field must contain only the final answer.
Do not include explanation, Markdown, or a full sentence in answer.
For example, use "4" instead of "The answer is 4."
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

Important:
The refined_answer field must contain only the final answer.
Do not include explanation, Markdown, or a full sentence in refined_answer.
For example, use "4" instead of "The answer is 4."
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
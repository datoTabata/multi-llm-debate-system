import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from debate.models import create_model_clients
from debate.pipeline import (
    assign_roles,
    collect_role_preferences,
    generate_independent_solutions,
    generate_peer_reviews,
    judge_solutions,
    refine_solutions,
)


def main() -> None:
    problem = {
        "id": "demo_001",
        "question": "What is 2 + 2?",
        "correct_answer": "4",
    }

    models = create_model_clients()
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

    print("Role preferences:")
    print(role_preferences)

    print("\nAssigned roles:")
    print(roles)

    print("\nIndependent solutions:")
    print(solutions)

    print("\nPeer reviews:")
    print(peer_reviews)

    print("\nRefined solutions:")
    print(refined_solutions)

    print("\nFinal judgment:")
    print(judgment)


if __name__ == "__main__":
    main()
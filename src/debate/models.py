from dataclasses import dataclass


class ModelClient:
    """Base interface for all model clients."""

    name: str

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class MockClient(ModelClient):
    """Simple model client used for local testing."""

    name: str

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        if "solver or judge" in prompt_lower:
            if self.name == "model_d":
                return (
                    '{"preferred_role": "Judge", '
                    '"solver_confidence": 0.6, '
                    '"judge_confidence": 0.9, '
                    '"reasoning": "This model is better suited for evaluating competing solutions."}'
                )

            return (
                '{"preferred_role": "Solver", '
                '"solver_confidence": 0.8, '
                '"judge_confidence": 0.6, '
                '"reasoning": "This model is better suited for solution generation."}'
            )

        if "solve the following problem independently" in prompt_lower:
            return (
                '{"reasoning": "This is an initial mock solution.", '
                '"answer": "5", '
                '"confidence": 0.65}'
            )

        if "review the following solution" in prompt_lower:
            return (
                '{"strengths": ["The solution gives a clear final answer."], '
                '"weaknesses": ["The reasoning should show more verification."], '
                '"possible_errors": [], '
                '"suggested_changes": ["Add a short check of the final answer."], '
                '"overall_assessment": "reasonable"}'
            )

        if "refine your solution" in prompt_lower:
            return (
                '{"changes_made": ['
                '{"critique": "The reasoning should show more verification.", '
                '"response": "Added a verification step.", '
                '"accepted": true}'
                '], '
                '"refined_solution": "The solution was checked and refined using peer feedback.", '
                '"refined_answer": "4", '
                '"confidence": 0.85}'
            )

        if "judge the final solutions" in prompt_lower:
            return (
                '{"winner": "solver_1", '
                '"confidence": 0.8, '
                '"reasoning": "Solver 1 provided the clearest refined answer."}'
            )

        return "Mock response"


def create_model_clients() -> dict[str, ModelClient]:
    """Create model clients for one debate run."""

    return {
        "model_a": MockClient("model_a"),
        "model_b": MockClient("model_b"),
        "model_c": MockClient("model_c"),
        "model_d": MockClient("model_d"),
    }
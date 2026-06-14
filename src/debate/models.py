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

        return "Mock response"


def create_model_clients() -> dict[str, ModelClient]:
    """Create model clients for one debate run."""

    return {
        "model_a": MockClient("model_a"),
        "model_b": MockClient("model_b"),
        "model_c": MockClient("model_c"),
        "model_d": MockClient("model_d"),
    }
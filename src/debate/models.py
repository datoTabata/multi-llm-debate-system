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
        return "Mock response"


def create_model_clients() -> dict[str, ModelClient]:
    """Create model clients for one debate run."""

    return {
        "model_a": MockClient("model_a"),
        "model_b": MockClient("model_b"),
        "model_c": MockClient("model_c"),
        "model_d": MockClient("model_d"),
    }
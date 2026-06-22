import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


class ModelClient:
    """Base interface for all model clients."""

    name: str

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_structured(self, prompt: str, response_format):
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

    def generate_structured(self, prompt: str, response_format):
        response = self.generate(prompt)
        return response_format.model_validate_json(response)


@dataclass
class OpenAIClient(ModelClient):
    """OpenAI model client."""

    name: str
    model_name: str

    def generate(self, prompt: str) -> str:
        client = OpenAI()

        response = client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        return response.output_text

    def generate_structured(self, prompt: str, response_format):
        client = OpenAI()

        response = client.responses.parse(
            model=self.model_name,
            input=prompt,
            text_format=response_format,
        )

        return response.output_parsed


def create_model_clients() -> dict[str, ModelClient]:
    """Create the four model clients used in one debate run."""

    load_dotenv()

    provider = os.getenv("MODEL_PROVIDER", "mock")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if provider == "openai":
        return {
            "model_a": OpenAIClient("model_a", model_name),
            "model_b": OpenAIClient("model_b", model_name),
            "model_c": OpenAIClient("model_c", model_name),
            "model_d": OpenAIClient("model_d", model_name),
        }

    return {
        "model_a": MockClient("model_a"),
        "model_b": MockClient("model_b"),
        "model_c": MockClient("model_c"),
        "model_d": MockClient("model_d"),
    }
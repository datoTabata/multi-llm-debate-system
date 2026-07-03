import asyncio
import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError

from debate.problem import Grader

# The schema class passed to agenerate_structured; the call returns an
# instance of that same class.
ResponseSchema = TypeVar("ResponseSchema", bound=BaseModel)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.toml"


class RunConfig(BaseModel):
    """[run] section: which problems to run and where to write results."""

    dataset: str = "problems.json"
    problem_ids: list[str] = []
    problem_limit: int = 0
    seed: int | None = None
    max_concurrent_problems: int = 0
    output_file: str = "demo_result.json"


class DefaultsConfig(BaseModel):
    """[defaults] section: fallbacks for any slot that doesn't override them."""

    temperature: float = 0.7
    max_tokens: int = 4000


class SlotConfig(BaseModel):
    """One model slot ([participants.*] or [grader]).

    `temperature` / `max_tokens` left unset fall back to [defaults].
    """

    model: str
    temperature: float | None = None
    max_tokens: int | None = None


class GraderConfig(SlotConfig):
    """[grader] slot. Unlike participants, grading should be deterministic, so
    an omitted temperature means 0.0 (not the [defaults] sampling temperature),
    and the model has a sensible default."""

    model: str = "google/gemini-2.5-flash"
    temperature: float | None = 0.0


class AppConfig(BaseModel):
    """The whole of config/models.toml, validated at load time."""

    run: RunConfig = RunConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    grader: GraderConfig = GraderConfig()
    participants: dict[str, SlotConfig]


def load_config(path: str | None = None) -> AppConfig:
    """Load and validate the TOML configuration file (models + run settings)."""

    config_path = Path(path) if path else CONFIG_PATH

    with config_path.open("rb") as file:
        return AppConfig(**tomllib.load(file))


def load_run_config(path: str | None = None) -> RunConfig:
    """Load only the [run] section.

    For tools that just need run settings (e.g. the plotter reading
    output_file) and shouldn't fail if the model slots are missing or mid-edit.
    """

    config_path = Path(path) if path else CONFIG_PATH

    with config_path.open("rb") as file:
        return RunConfig(**tomllib.load(file).get("run", {}))


@dataclass
class OpenRouterClient:
    """Thin wrapper over the official OpenRouter SDK.

    Holds the per-slot config (model + temperature) and centralizes the two
    call shapes the pipeline needs: plain text and schema-validated JSON.
    """

    # No defaults on purpose: [defaults] in config/models.toml is the single
    # source of truth for fallback values (applied by _make_client).
    model_name: str
    temperature: float
    max_tokens: int

    def _client(self) -> OpenRouter:
        return OpenRouter(api_key=os.getenv("OPEN_ROUTER_KEY"))

    def _send_kwargs(self, prompt: str, response_format: type[BaseModel] | None = None) -> dict:
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema_": response_format.model_json_schema(),
                },
            }

        return kwargs

    def _parse_structured(self, result, response_format: type[ResponseSchema]) -> ResponseSchema:
        choice = result.choices[0]

        if choice.finish_reason == "length":
            raise RuntimeError(
                f"{self.model_name} response was truncated at max_tokens="
                f"{self.max_tokens}; raise max_tokens in config/models.toml."
            )

        return response_format.model_validate_json(choice.message.content)

    async def agenerate(self, prompt: str) -> str:
        result = await self._client().chat.send_async(**self._send_kwargs(prompt))
        return result.choices[0].message.content

    async def agenerate_structured(
        self, prompt: str, response_format: type[ResponseSchema], retries: int = 3
    ) -> ResponseSchema:
        last_error = None
        for _ in range(retries):
            try:
                result = await self._client().chat.send_async(
                    **self._send_kwargs(prompt, response_format)
                )
                return self._parse_structured(result, response_format)
            except (ValidationError, RuntimeError, ValueError) as error:
                last_error = error
        raise last_error


def _make_client(slot: SlotConfig, defaults: DefaultsConfig) -> OpenRouterClient:
    """Build a client from a slot config, filling gaps from [defaults]."""

    return OpenRouterClient(
        model_name=slot.model,
        temperature=slot.temperature if slot.temperature is not None else defaults.temperature,
        max_tokens=slot.max_tokens if slot.max_tokens is not None else defaults.max_tokens,
    )


def create_grader_client(config: AppConfig | None = None) -> OpenRouterClient:
    """Create the client used to grade free-answer problems (the [grader] slot)."""

    load_dotenv()

    if config is None:
        config = load_config()

    return _make_client(config.grader, config.defaults)


def make_llm_grader(client: OpenRouterClient) -> Grader:
    """Return a grader callable: grader(question, predicted, correct) -> bool.

    Memoized: several metrics grade the same (question, predicted, correct)
    triple (e.g. the winner's answer is checked by both accuracy and
    judge-accuracy), and each grade is a paid LLM call — cache them.
    """

    from debate.prompts import build_grader_prompt
    from debate.schemas import GradeResponse

    @cache
    def grade(question: str, predicted: str, correct: str) -> bool:
        # Grading runs after the debate's event loop has finished, so it is
        # safe to spin up a fresh loop per call with asyncio.run.
        prompt = build_grader_prompt(question, predicted, correct)
        response = asyncio.run(client.agenerate_structured(prompt, GradeResponse))
        return response.is_correct

    return grade


def create_model_clients(config: AppConfig | None = None) -> dict[str, OpenRouterClient]:
    """Create the four OpenRouter clients used in one debate run."""

    load_dotenv()

    if config is None:
        config = load_config()

    return {slot: _make_client(spec, config.defaults) for slot, spec in config.participants.items()}

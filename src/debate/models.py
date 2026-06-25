import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openrouter import OpenRouter

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.toml"


def load_config(path: str | None = None) -> dict:
    """Load the TOML configuration file (models + run settings)."""

    config_path = Path(path) if path else CONFIG_PATH

    with config_path.open("rb") as file:
        return tomllib.load(file)


@dataclass
class OpenRouterClient:
    """Thin wrapper over the official OpenRouter SDK.

    Holds the per-slot config (model + temperature) and centralizes the two
    call shapes the pipeline needs: plain text and schema-validated JSON.
    """

    name: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4000

    def _client(self) -> OpenRouter:
        return OpenRouter(api_key=os.getenv("OPEN_ROUTER_KEY"))

    def _send_kwargs(self, prompt: str, response_format=None) -> dict:
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

    def _parse_structured(self, result, response_format):
        choice = result.choices[0]

        if choice.finish_reason == "length":
            raise RuntimeError(
                f"{self.model_name} response was truncated at max_tokens="
                f"{self.max_tokens}; raise max_tokens in config/models.toml."
            )

        return response_format.model_validate_json(choice.message.content)

    def generate(self, prompt: str) -> str:
        result = self._client().chat.send(**self._send_kwargs(prompt))
        return result.choices[0].message.content

    def generate_structured(self, prompt: str, response_format):
        result = self._client().chat.send(**self._send_kwargs(prompt, response_format))
        return self._parse_structured(result, response_format)

    async def agenerate(self, prompt: str) -> str:
        result = await self._client().chat.send_async(**self._send_kwargs(prompt))
        return result.choices[0].message.content

    async def agenerate_structured(self, prompt: str, response_format):
        result = await self._client().chat.send_async(
            **self._send_kwargs(prompt, response_format)
        )
        return self._parse_structured(result, response_format)


def create_grader_client(config: dict | None = None) -> OpenRouterClient:
    """Create the client used to grade free-answer problems (the [grader] slot)."""

    load_dotenv()

    if config is None:
        config = load_config()

    grader = config.get("grader", {})

    return OpenRouterClient(
        name="grader",
        model_name=grader.get("model", "google/gemini-2.5-flash"),
        temperature=grader.get("temperature", 0.0),
        max_tokens=grader.get("max_tokens", config.get("defaults", {}).get("max_tokens", 4000)),
    )


def make_llm_grader(client: OpenRouterClient):
    """Return a grader callable: grader(question, predicted, correct) -> bool."""

    from debate.prompts import build_grader_prompt
    from debate.schemas import GradeResponse

    def grade(question: str, predicted: str, correct: str) -> bool:
        prompt = build_grader_prompt(question, predicted, correct)
        return client.generate_structured(prompt, GradeResponse).is_correct

    return grade


def create_model_clients(config: dict | None = None) -> dict[str, OpenRouterClient]:
    """Create the four OpenRouter clients used in one debate run."""

    load_dotenv()

    if config is None:
        config = load_config()

    defaults = config.get("defaults", {})
    default_temperature = defaults.get("temperature", 0.7)
    default_max_tokens = defaults.get("max_tokens", 4000)

    clients = {}
    for slot, spec in config["participants"].items():
        clients[slot] = OpenRouterClient(
            name=slot,
            model_name=spec["model"],
            temperature=spec.get("temperature", default_temperature),
            max_tokens=spec.get("max_tokens", default_max_tokens),
        )

    return clients

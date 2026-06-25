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

    def _client(self) -> OpenRouter:
        return OpenRouter(api_key=os.getenv("OPEN_ROUTER_KEY"))

    def generate(self, prompt: str) -> str:
        result = self._client().chat.send(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        return result.choices[0].message.content

    def generate_structured(self, prompt: str, response_format):
        result = self._client().chat.send(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema_": response_format.model_json_schema(),
                },
            },
        )

        return response_format.model_validate_json(result.choices[0].message.content)


def create_model_clients(config: dict | None = None) -> dict[str, OpenRouterClient]:
    """Create the four OpenRouter clients used in one debate run."""

    load_dotenv()

    if config is None:
        config = load_config()

    default_temperature = config.get("defaults", {}).get("temperature", 0.7)

    clients = {}
    for slot, spec in config["participants"].items():
        clients[slot] = OpenRouterClient(
            name=slot,
            model_name=spec["model"],
            temperature=spec.get("temperature", default_temperature),
        )

    return clients

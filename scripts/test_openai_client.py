import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

from debate.models import OpenAIClient


def main() -> None:
    load_dotenv()

    client = OpenAIClient("test_model", "gpt-4o-mini")
    response = client.generate(
        "Return only this JSON: {\"answer\": \"ok\"}"
    )

    print(response)


if __name__ == "__main__":
    main()
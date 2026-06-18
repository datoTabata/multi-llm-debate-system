import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def load_metrics() -> dict:
    """Load metrics from the demo output file."""

    output_path = PROJECT_ROOT / "outputs" / "demo_result.json"

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["metrics"]


def main() -> None:
    metrics = load_metrics()

    plot_metrics = {
        key: value
        for key, value in metrics.items()
        if value is not None
    }

    names = list(plot_metrics.keys())
    values = list(plot_metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Debate System Evaluation Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "metrics_plot.png"
    plt.savefig(output_path)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
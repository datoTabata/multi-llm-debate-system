import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from debate.models import load_config


def output_filename() -> str:
    """The result file the last run wrote, per config."""

    return load_config().get("run", {}).get("output_file", "demo_result.json")


def load_data() -> dict:
    """Load the full result file (metrics + config) for the latest run."""

    output_path = PROJECT_ROOT / "outputs" / output_filename()

    with output_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    data = load_data()
    metrics = data["metrics"]
    dataset = data.get("config", {}).get("dataset") or output_filename()

    plot_metrics = {
        key: value
        for key, value in metrics.items()
        if value is not None
    }

    names = list(plot_metrics.keys())
    values = list(plot_metrics.values())
    n_problems = len(data.get("evaluations", []))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color="#4C72B0")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Debate System Metrics — {dataset} ({n_problems} problems)")
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")

    plt.tight_layout()

    stem = Path(output_filename()).stem
    output_path = PROJECT_ROOT / "outputs" / f"{stem}_plot.png"
    plt.savefig(output_path)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
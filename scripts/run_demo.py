"""CLI: run the debate for the [run] selection in config/models.toml, save the
result JSON, and generate its dashboard. All logic lives in the `debate`
package (debate.runner) so the same flow is callable from the notebooks.
"""

import json

from build_dashboard import build as build_dashboard
from debate.models import load_config
from debate.runner import run_and_evaluate, save_output


def main() -> None:
    config = load_config()
    output = run_and_evaluate(config)
    output_path = save_output(output, config.run.output_file)
    dashboard_path = build_dashboard(output_path)

    metrics = output["metrics"]
    print(f"Saved result to {output_path}")
    print(f"Saved dashboard to {dashboard_path}")
    print(f"Problems evaluated: {len(output['results'])}")
    print(f"Accuracy (debate system): {metrics['accuracy']:.2f}")
    print(f"Accuracy (single-LLM baseline): {metrics['single_llm_accuracy']:.2f}")
    print(f"Accuracy (majority-vote baseline): {metrics['majority_vote_accuracy']:.2f}")
    print(f"Consensus rate: {metrics['consensus_rate']:.2f}")
    print(f"Judge accuracy on disagreements: {metrics['judge_accuracy_on_disagreements']}")
    print(f"Improvement rate: {metrics['improvement_rate']:.2f}")

    print("\nEvaluations:")
    print(json.dumps(output["evaluations"], indent=2))


if __name__ == "__main__":
    main()

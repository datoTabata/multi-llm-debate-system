# Multi-LLM Collaborative Debate System

This project implements a collaborative debate workflow where multiple language models solve the same problem, review each other's solutions, refine their answers, and send final solutions to a judge model.

The current version uses mock model clients, so the full pipeline can be tested without API keys.

## Workflow

1. Load problems from `data/problems.json`.
2. Ask each model whether it prefers the Solver or Judge role.
3. Assign three Solver roles and one Judge role.
4. Generate independent solver solutions.
5. Generate peer reviews between solvers.
6. Refine solutions using peer feedback.
7. Ask the judge to choose the best final answer.
8. Evaluate the final answer against the known correct answer.
9. Save results and generate evaluation plots.

## Project Structure

```text
data/
  problems.json

src/
  debate/
    models.py
    prompts.py
    pipeline.py
    evaluation.py

scripts/
  run_demo.py
  plot_metrics.py

outputs/
  demo_result.json
  metrics_plot.png
```

`outputs/` is ignored by Git because it contains generated files.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Run Demo

Run the debate pipeline on the dataset:

```bash
python scripts/run_demo.py
```

This creates:

```text
outputs/demo_result.json
```

## Generate Plot

After running the demo, generate the metrics plot:

```bash
python scripts/plot_metrics.py
```

This creates:

```text
outputs/metrics_plot.png
```

## Current Metrics

The demo calculates:

- Accuracy
- Consensus rate
- Judge accuracy on disagreement cases
- Improvement rate

## Current Status

The project currently uses mock LLM clients. This allows the full debate, refinement, judging, evaluation, and plotting pipeline to run locally without paid API calls.

The next major step is connecting real LLM APIs and expanding the dataset to 25 challenging problems.

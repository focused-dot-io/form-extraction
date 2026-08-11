"""Runs one LangSmith experiment per provider against the uploaded dataset,
so the four extraction methods show up as separate, comparable experiments
in the LangSmith UI. Each experiment invokes the same extraction graph
(bench/extraction/graph.py), just with a different provider passed in as
runtime context.

Usage:
    python -m bench.eval.run_experiment --provider all
    python -m bench.eval.run_experiment --provider claude
"""

import argparse

from dotenv import load_dotenv

load_dotenv()  # before importing bench.* — those modules read provider API keys at import time

from langsmith import evaluate

from bench.eval.evaluators import exact_match_evaluator, field_accuracy_evaluator
from bench.extraction.graph import PROVIDERS, run_one


def make_run_function(provider: str):
    def run(inputs: dict) -> dict:
        return run_one(inputs["image_path"], provider)

    run.__name__ = f"extract_{provider}"
    return run


def run_provider(provider: str, dataset_name: str) -> None:
    evaluate(
        make_run_function(provider),
        data=dataset_name,
        evaluators=[field_accuracy_evaluator, exact_match_evaluator],
        experiment_prefix=f"extraction-{provider}",
        metadata={"provider": provider},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="form-extraction-bench")
    parser.add_argument("--provider", choices=[*PROVIDERS, "all"], default="all")
    args = parser.parse_args()

    providers = PROVIDERS if args.provider == "all" else [args.provider]
    for provider in providers:
        print(f"Running LangSmith experiment for {provider}...")
        run_provider(provider, args.dataset_name)


if __name__ == "__main__":
    main()

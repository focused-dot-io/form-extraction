"""Runs one LangSmith experiment per provider against the uploaded dataset,
so the four extraction methods show up as separate, comparable experiments
in the LangSmith UI. Each experiment invokes the same extraction graph
(bench/extraction/graph.py), just with a different provider passed in as
runtime context.

Usage:
    python -m bench.eval.run_experiment --provider all
    python -m bench.eval.run_experiment --provider claude
    python -m bench.eval.run_experiment --provider gemini --model gemini-flash-lite-latest
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()  # before importing bench.* — those modules read provider API keys at import time

from langsmith import evaluate

from bench.eval.evaluators import exact_match_evaluator, field_accuracy_evaluator
from bench.extraction.extractors import claude_extractor, gemini_extractor, openai_extractor
from bench.extraction.graph import PROVIDERS, run_one

MODEL_ENV_VAR = {
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
    "claude": "CLAUDE_MODEL",
}
# get_model() resolves the currently-active model for that provider (env
# override if set, else the extractor's own default) — used so every
# experiment logs which model actually ran, not just the ones with a
# one-off --model override.
GET_MODEL = {
    "openai": openai_extractor.get_model,
    "gemini": gemini_extractor.get_model,
    "claude": claude_extractor.get_model,
}


def make_run_function(provider: str):
    def run(inputs: dict) -> dict:
        return run_one(inputs["image_path"], provider)

    run.__name__ = f"extract_{provider}"
    return run


def run_provider(provider: str, dataset_name: str, model: str | None = None) -> None:
    env_var = MODEL_ENV_VAR.get(provider)
    original = os.environ.get(env_var) if env_var else None

    # Override just for this run so a one-off "try the lite model" doesn't
    # require editing .env — restored in the finally block either way.
    if model and env_var:
        os.environ[env_var] = model

    label = f"{provider}-{model}" if model else provider
    resolved_model = GET_MODEL[provider]() if provider in GET_MODEL else None
    metadata = {"provider": provider}
    if resolved_model:
        metadata["model"] = resolved_model

    try:
        evaluate(
            make_run_function(provider),
            data=dataset_name,
            evaluators=[field_accuracy_evaluator, exact_match_evaluator],
            experiment_prefix=f"extraction-{label}",
            metadata=metadata,
        )
    finally:
        if model and env_var:
            if original is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="form-extraction-bench")
    parser.add_argument("--provider", choices=[*PROVIDERS, "all"], default="all")
    parser.add_argument(
        "--model",
        help="Override the model for this run (only valid with a single --provider, not 'all')",
    )
    args = parser.parse_args()

    if args.model and args.provider == "all":
        parser.error("--model requires a specific --provider, not 'all'")

    providers = PROVIDERS if args.provider == "all" else [args.provider]
    for provider in providers:
        label = f"{provider} ({args.model})" if args.model else provider
        print(f"Running LangSmith experiment for {label}...")
        run_provider(provider, args.dataset_name, args.model)


if __name__ == "__main__":
    main()

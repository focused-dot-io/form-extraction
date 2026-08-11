"""A single-node LangGraph graph that extracts fields from one form image
using whichever provider is passed in as runtime context.

One graph, parameterized by `ExtractionContext.provider` — not a separate
branch per provider. The LangSmith eval harness invokes this same graph
once per provider (see bench/eval/run_experiment.py); the CLI comparison
tool (run.py) invokes it once per provider in a loop and combines the
results for a side-by-side view.
"""

from dataclasses import dataclass

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from bench.extraction.extractors import claude_extractor, gemini_extractor, ocr_extractor, openai_extractor

EXTRACTORS = {
    "openai": openai_extractor,
    "gemini": gemini_extractor,
    "claude": claude_extractor,
    "ocr": ocr_extractor,
}
PROVIDERS = list(EXTRACTORS)


@dataclass
class ExtractionContext:
    provider: str


class ExtractionState(TypedDict):
    image_path: str
    result: dict


def _extract(state: ExtractionState, runtime: Runtime[ExtractionContext]) -> dict:
    extractor = EXTRACTORS[runtime.context.provider]
    return {"result": extractor.extract(state["image_path"])}


def build_graph():
    graph = StateGraph(ExtractionState, context_schema=ExtractionContext)
    graph.add_node("extract", _extract)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", END)
    return graph.compile()


def run_one(image_path: str, provider: str) -> dict:
    """Runs the graph against a single provider — this is what the eval harness calls."""
    app = build_graph()
    result = app.invoke({"image_path": image_path}, context=ExtractionContext(provider=provider))
    return result["result"]


def run_all(image_path: str) -> dict:
    """Runs the graph once per provider on the same image, for side-by-side comparison."""
    return {provider: run_one(image_path, provider) for provider in PROVIDERS}

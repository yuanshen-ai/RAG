from typing import Any

from agents.query.schemas import QueryGraphStepInfo, QuerySourceInfo


def step_event(step: QueryGraphStepInfo) -> dict[str, Any]:
    return {
        "type": "step",
        "data": step.model_dump(),
    }


def answer_delta_event(delta: str) -> dict[str, Any]:
    return {
        "type": "answer_delta",
        "data": {"delta": delta},
    }


def sources_event(sources: list[QuerySourceInfo]) -> dict[str, Any]:
    return {
        "type": "source",
        "data": {"sources": [source.model_dump() for source in sources]},
    }
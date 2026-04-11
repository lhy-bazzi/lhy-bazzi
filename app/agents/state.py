"""LangGraph shared agent state."""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    query: str
    chat_history: list[dict]
    user_id: str
    kb_ids: list[str]

    # Query understanding
    query_plan: Optional[dict]
    sub_queries: list[str]

    # Retrieval
    retrieved_contexts: Annotated[list, operator.add]
    retrieval_rounds: int

    # Reasoning
    sub_answers: list[dict]
    reasoning_notes: str

    # Output
    final_answer: str
    citations: list[dict]
    stream_tokens: list[str]

    # Control
    routing_decision: str  # "simple" | "complex" | "conversational"
    quality_check: Optional[dict]
    iteration_count: int
    max_iterations: int
    should_continue: bool

    # Config passthrough
    retrieval_config: Optional[dict]
    model: Optional[str]

"""Router agent — decides simple vs complex execution path."""

from __future__ import annotations

from app.agents.state import AgentState
from app.models.enums import IntentType

_COMPLEX_INTENTS = {IntentType.MULTI_HOP, IntentType.ANALYTICAL, IntentType.COMPARATIVE}


async def router_agent(state: AgentState) -> dict:
    plan = state.get("query_plan") or {}
    intent_str = plan.get("intent", "factual")
    strategy = plan.get("strategy", "simple_rag")

    if strategy == "conversational":
        decision = "conversational"
    elif strategy == "multi_agent":
        decision = "complex"
    else:
        decision = "simple"

    return {"routing_decision": decision}

"""LangGraph graph definitions — simple RAG and deep QA."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.state import AgentState
from app.agents.router_agent import router_agent
from app.agents.retrieval_agent import retrieval_agent
from app.agents.reasoning_agent import reasoning_agent
from app.agents.synthesizer_agent import synthesizer_agent
from app.agents.critic_agent import critic_agent


# ---------------------------------------------------------------------------
# Edge condition functions
# ---------------------------------------------------------------------------

def route_decision(state: AgentState) -> str:
    return state.get("routing_decision", "simple")


def check_complexity(state: AgentState) -> str:
    decision = state.get("routing_decision", "simple")
    return "complex" if decision == "complex" else "simple"


def quality_gate(state: AgentState) -> str:
    qc = state.get("quality_check") or {}
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    if qc.get("passed", True) or iteration >= max_iter:
        return "pass"
    return "retry"


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_simple_rag_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retriever", retrieval_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


def build_deep_qa_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("retriever", retrieval_agent)
    graph.add_node("reasoner", reasoning_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("critic", critic_agent)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", route_decision, {
        "simple": "retriever",
        "complex": "retriever",
        "conversational": "synthesizer",
    })

    graph.add_conditional_edges("retriever", check_complexity, {
        "simple": "synthesizer",
        "complex": "reasoner",
    })

    graph.add_edge("reasoner", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    graph.add_conditional_edges("critic", quality_gate, {
        "pass": END,
        "retry": "retriever",
    })

    return graph.compile()

"""Main LangGraph React Agent implementation for tender research."""

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.react_nodes import (
    orientation_node,
    planner_node,
    reflection_node,
    synthesizer_node,
    tool_executor_node,
    triage_node,
)
from open_deep_research.state import ReactAgentState

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

def create_react_agent_graph():
    """Create and configure the React Agent LangGraph workflow."""
    react_graph_builder = StateGraph(
        ReactAgentState,
        config_schema=Configuration
    )
    
    react_graph_builder.add_node("triage", triage_node)
    react_graph_builder.add_node("orientation", orientation_node)
    react_graph_builder.add_node("planner", planner_node)
    react_graph_builder.add_node("tool_executor", tool_executor_node)
    react_graph_builder.add_node("reflection", reflection_node)
    react_graph_builder.add_node("synthesizer", synthesizer_node)

    react_graph_builder.add_edge(START, "triage")
    react_graph_builder.add_edge("synthesizer", END)

    react_agent = react_graph_builder.compile()
    
    return react_agent

react_agent = create_react_agent_graph()

app = react_agent
"""Main LangGraph React Agent implementation for tender research."""

from typing import Literal
from langchain.chat_models import init_chat_model
from langraph.graph import StateGraph, START, END
from langraph.types import Command

from open_deep_research.state import ReactAgentState
from open_deep_research.react_nodes import (
    triage_node,
    orientation_node,
    planner_node,
    tool_executor_node,
    reflection_node,
    synthesizer_node
)
from open_deep_research.react_tools import REACT_TOOLS
from open_deep_research.configuration import Configuration

# Initialize configurable model
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


def create_react_agent_graph():
    """Create and configure the React Agent LangGraph workflow."""
    
    # Create the state graph
    react_graph_builder = StateGraph(
        ReactAgentState,
        config_schema=Configuration
    )
    
    # Add all nodes
    react_graph_builder.add_node("triage", triage_node)
    react_graph_builder.add_node("orientation", orientation_node)
    react_graph_builder.add_node("planner", planner_node)
    react_graph_builder.add_node("tool_executor", tool_executor_node)
    react_graph_builder.add_node("reflection", reflection_node)
    react_graph_builder.add_node("synthesizer", synthesizer_node)
    
    # Define the workflow edges
    # Entry point always goes to triage
    react_graph_builder.add_edge(START, "triage")
    
    # Triage routes to either orientation (deep dive) or synthesizer (fast track)
    # This is handled by the Command return values in triage_node
    
    # Orientation always goes to planner
    # Handled by Command return in orientation_node
    
    # Planner routes to either tool_executor or synthesizer
    # Handled by Command return in planner_node
    
    # Tool executor always goes to reflection
    # Handled by Command return in tool_executor_node
    
    # Reflection routes to either planner (continue) or synthesizer (done)
    # Handled by Command return in reflection_node
    
    # Synthesizer always goes to END
    react_graph_builder.add_edge("synthesizer", END)
    
    # Compile the graph
    react_agent = react_graph_builder.compile()
    
    return react_agent


# Create the compiled react agent
react_agent = create_react_agent_graph()

# Export for deployment (following LangGraph patterns)
app = react_agent

"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str



class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class CompanyInfo(BaseModel):
    """Company information for research."""
    
    company_name: str = Field(description="The name of the company to research")
    cvr_number: str | None = Field(default=None, description="CVR number (if company is in Denmark)")
    country: str = Field(description="Country where the company is located (Denmark or EU)")
    primary_industry: str = Field(description="Primary industry/sector of the company")

class AgentInputState(TypedDict):
    """Input state containing company information for research."""
    
    company_info: dict

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    company_info: dict
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str | None
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


###################
# React Agent State for Tender Research
###################

class ScratchpadEntry(BaseModel):
    """Entry in the agent's scratchpad for tracking reasoning steps."""
    
    step: int = Field(description="Step number in the reasoning process")
    type: str = Field(description="Type of entry: Thought, Action, or Observation")
    content: str = Field(description="Content of the scratchpad entry")

class IntermediateResult(BaseModel):
    """Structured data gathered by tools during the reasoning process."""
    
    tool_name: str = Field(description="Name of the tool that generated this result")
    result_data: dict = Field(description="Structured result data from the tool")
    file_sources: list[str] = Field(default=[], description="File sources referenced in this result")
    web_sources: list[str] = Field(default=[], description="Web sources referenced in this result")

class ReactAgentState(MessagesState):
    """Agent state for the LangGraph react agent with iterative reasoning loop."""
    
    user_query: str = Field(description="The user's original query")
    tender_id: str | None = Field(default=None, description="Optional identifier for the current tender being analyzed")
    chat_history: Annotated[list[dict], operator.add] = Field(default=[], description="Conversational context")
    
    # Cached situational awareness data
    manifest_overview: dict = Field(default={}, description="Cached tender manifest overview data")
    
    # Agent's working memory and "Show Your Work" log (streamed to UI)
    scratchpad: Annotated[list[ScratchpadEntry], operator.add] = Field(default=[], description="Agent's reasoning steps")
    
    # Structured data gathered by tools (used for final synthesis)
    intermediate_results: Annotated[list[IntermediateResult], operator.add] = Field(default=[], description="Structured data from tool executions")
    
    # Loop control
    iterations: int = Field(default=0, description="Counter to prevent infinite loops")
    max_iterations: int = Field(default=10, description="Maximum allowed iterations")
    
    # Current plan and next actions
    current_plan: str = Field(default="", description="Current plan from the planner node")
    next_action: str = Field(default="", description="Next action to take")
    
    # Fast track optimization flags
    is_simple_query: bool = Field(default=False, description="Flag indicating if this is a simple query for fast track")
    confidence_score: float = Field(default=0.0, description="Confidence score for fast track optimization")
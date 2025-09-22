# LangGraph React Agent for Tender Research

## Overview

This document provides a comprehensive guide to the new LangGraph React Agent implementation for tender research. The agent follows the architectural pattern described in your requirements with a Plan -> Execute -> Reflect iterative reasoning loop, optimization triage, and foundational tool arsenal.

## Architecture

The React Agent implements the following architecture:

```
User Query → Triage Node → [Fast Track OR Deep Dive]
                ↓
            Orientation Node (Deep Dive only)
                ↓
            Planner Node ← ← ← ← ←
                ↓                ↑
            Tool Executor      ↑
                ↓                ↑
            Reflection Node → → ↑
                ↓
            Synthesizer Node → Final Answer
```

### Node Descriptions

1. **Triage Node** - Evaluates query complexity and determines routing (Fast Track vs Deep Dive)
2. **Orientation Node** - Establishes situational awareness with tender manifest data
3. **Planner Node** - Central reasoning engine using ReAct/Chain-of-Thought prompting
4. **Tool Executor Node** - Executes planned tools with support for parallel execution
5. **Reflection Node** - Evaluates tool effectiveness and determines next steps
6. **Synthesizer Node** - Generates final comprehensive answer with citations

## State Management

### ReactAgentState

The agent uses a comprehensive state schema that tracks:

- **User Query & Context**: Original query, tender ID, chat history
- **Situational Awareness**: Cached tender manifest overview
- **Reasoning Process**: Scratchpad entries showing "Show Your Work" 
- **Intermediate Results**: Structured data from tool executions
- **Loop Control**: Iteration counters and limits
- **Fast Track Optimization**: Complexity and confidence flags

```python
class ReactAgentState(MessagesState):
    user_query: str
    tender_id: str
    chat_history: list[dict]
    manifest_overview: dict
    scratchpad: list[ScratchpadEntry]
    intermediate_results: list[IntermediateResult]
    iterations: int
    max_iterations: int
    current_plan: str
    next_action: str
    is_simple_query: bool
    confidence_score: float
```

## Foundational Tool Arsenal

The agent has access to 5 core tools:

### 1. Consult_Tender_Manifest
**Purpose**: Rapid access to metadata and summaries for situational awareness.

**Capabilities**:
- `get_tender_overview()`: Returns 10-page summary
- `list_all_documents()`: Returns document inventory
- `map_names_to_ids(user_references)`: Maps ambiguous references to file IDs

### 2. Targeted_Hybrid_Search
**Purpose**: Primary RAG workhorse for deep content extraction.

**Features**:
- Hybrid search (Vector + BM25) with Cohere rerank
- Parent-Child retrieval logic (search child chunks, return parent chunks)
- Optional file ID filters for scoped searches

### 3. Iterative_Document_Analyzer
**Purpose**: Handles analysis of large documents using MapReduce strategy.

**Process**:
- Fetch all parent chunks for specified file ID
- Map phase: Process chunks iteratively with LLM calls
- Reduce phase: Synthesize extractions into final output

### 4. Web_Search
**Purpose**: Access external regulations, legal definitions, and market intelligence.

**Implementation**: Tavily API or similar for external information retrieval.

### 5. Wait_for_User_Input
**Purpose**: Enables proactive ambiguity resolution and user collaboration.

**Implementation**: Uses LangGraph interrupts to pause execution and await user input.

## Usage Examples

### Basic Usage

```python
from open_deep_research.react_agent import react_agent

# Define input state
input_state = {
    "user_query": "What are the penalty clauses in the IT infrastructure tender?",
    "tender_id": "tender_123",
    "messages": [HumanMessage(content="What are the penalty clauses?")]
}

# Run the agent
result = react_agent.invoke(input_state)
print(result["messages"][-1].content)
```

### With Configuration

```python
from open_deep_research.configuration import Configuration

config = {
    "configurable": {
        "model": "anthropic:claude-3-5-sonnet-20241022",
        "max_tokens": 4000,
        "max_iterations": 8
    }
}

result = react_agent.invoke(input_state, config=config)
```

### Streaming Results

```python
# Stream the agent's reasoning process
for event in react_agent.stream(input_state):
    if "scratchpad" in event:
        for entry in event["scratchpad"]:
            print(f"Step {entry.step} ({entry.type}): {entry.content}")
```

## Integration with Existing System

The React Agent is integrated into the existing deep research system:

### LangGraph Configuration

```json
{
  "graphs": {
    "Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher",
    "Tender Research Agent": "./src/open_deep_research/deep_researcher.py:tender_research_agent"
  }
}
```

### Deployment

The React Agent follows LangGraph deployment patterns:
- Exports compiled graph as `app` 
- No checkpointer (stateless by default)
- Compatible with LangGraph Cloud/Studio
- Supports authentication via existing auth module

## Key Features

### Fast Track Optimization
- Triage node evaluates query complexity and confidence
- Simple queries with high confidence bypass full reasoning loop
- Reduces latency for straightforward questions

### Reasoning Transparency
- Complete scratchpad shows agent's thinking process
- All tool executions and observations logged
- Facilitates debugging and user trust

### Iterative Improvement
- Reflection node evaluates tool effectiveness
- Agent can modify approach based on results
- Prevents getting stuck on poor retrieval results

### Comprehensive Citations
- All answers include clear source references
- File names, sections, and web sources tracked
- Enables user verification and follow-up research

## Mock Data for Testing

The implementation includes mock data structures for testing:

- **Tender Manifest**: Sample IT infrastructure tender with 3 documents
- **Document Content**: Mock contract text with penalties, SLAs, and requirements
- **Search Results**: Simulated hybrid search responses with confidence scores

## Production Considerations

For production deployment, replace mock implementations with:

1. **Vector Database**: Qdrant with parent-child chunking strategy
2. **Tender Manifest Store**: Dedicated metadata database or Qdrant collection
3. **Web Search**: Tavily API integration
4. **Document Analyzer**: Actual MapReduce implementation over document chunks
5. **User Input**: LangGraph interrupt mechanism for real user interaction

## Error Handling

The agent includes comprehensive error handling:
- Maximum iteration limits prevent infinite loops
- Tool execution failures are logged and handled gracefully
- Fallback responses provided when synthesis fails
- All errors captured in scratchpad for transparency

## Configuration Options

Key configuration parameters:
- `max_iterations`: Maximum reasoning loop iterations (default: 10)
- `model`: LLM model for reasoning (follows Anthropic > OpenAI > Google hierarchy)
- `max_tokens`: Token limit for model responses
- `search_api`: Backend search provider configuration

This React Agent implementation provides a robust, transparent, and efficient approach to tender research with clear reasoning visibility and comprehensive answer generation.

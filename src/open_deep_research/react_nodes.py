"""LangGraph nodes for the React Agent workflow."""

import asyncio
import json
from typing import Dict, Any, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.types import Command

from open_deep_research.state import ReactAgentState, ScratchpadEntry, IntermediateResult
from open_deep_research.react_tools import REACT_TOOLS, targeted_hybrid_search, consult_tender_manifest
from open_deep_research.prompts import (
    triage_system_prompt,
    orientation_system_prompt, 
    planner_system_prompt,
    reflection_system_prompt,
    synthesizer_system_prompt
)

# Initialize configurable model for nodes
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def triage_node(state: ReactAgentState, config: RunnableConfig) -> Command[Literal["orientation", "synthesizer"]]:
    """
    Triage node for optimization fast track - determines if query is simple enough for direct processing.
    Routes to either fast track (synthesizer) or deep dive (orientation).
    """
    user_query = state["user_query"]
    tender_id = state["tender_id"]
    
    # Add triage step to scratchpad
    step_num = len(state["scratchpad"]) + 1
    scratchpad_entry = ScratchpadEntry(
        step=step_num,
        type="Thought",
        content=f"Starting triage evaluation for query: {user_query}"
    )
    
    # Perform initial broad search to assess complexity
    try:
        search_results = await targeted_hybrid_search.ainvoke({
            "query": user_query,
            "tender_id": tender_id,
            "file_id_filters": None
        })
        
        # Calculate average confidence score
        if search_results and isinstance(search_results, list):
            confidence_scores = [r.get("confidence_score", 0.0) for r in search_results if isinstance(r, dict)]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        else:
            avg_confidence = 0.0
        
        # Use LLM to evaluate query complexity and confidence
        evaluation_prompt = f"""
        {triage_system_prompt}
        
        USER QUERY: {user_query}
        
        SEARCH RESULTS CONFIDENCE: {avg_confidence:.2f}
        NUMBER OF RESULTS: {len(search_results) if search_results else 0}
        
        Based on the query and initial search results, provide your assessment:
        """
        
        model = configurable_model.with_config(config)
        response = await model.ainvoke([
            SystemMessage(content=evaluation_prompt)
        ])
        
        # Parse response to determine routing
        response_content = response.content.lower()
        
        # Simple heuristics for routing decision
        is_simple = any(word in response_content for word in ["simple", "direct", "straightforward", "basic"])
        high_confidence = "high" in response_content and ("confidence" in response_content or "certain" in response_content)
        fast_track_mentioned = "fast_track" in response_content or "fast track" in response_content
        
        # Routing logic
        if (is_simple and high_confidence and avg_confidence > 0.5) or fast_track_mentioned:
            route = "synthesizer"
            decision = "FAST_TRACK"
        else:
            route = "orientation"
            decision = "DEEP_DIVE"
        
        # Update scratchpad with decision
        decision_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Triage Decision: {decision} - Routing to {route}. Confidence: {avg_confidence:.2f}"
        )
        
        updated_state = {
            "scratchpad": [scratchpad_entry, decision_entry],
            "is_simple_query": route == "synthesizer",
            "confidence_score": avg_confidence
        }
        
        # Store initial search results for fast track
        if route == "synthesizer" and search_results:
            intermediate_result = IntermediateResult(
                tool_name="initial_search",
                result_data={"search_results": search_results},
                file_sources=[r.get("file_name", "") for r in search_results if isinstance(r, dict)],
                web_sources=[]
            )
            updated_state["intermediate_results"] = [intermediate_result]
        
        return Command(
            update=updated_state,
            goto=route
        )
        
    except Exception as e:
        # If triage fails, default to deep dive
        error_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation", 
            content=f"Triage failed with error: {str(e)}. Defaulting to deep dive."
        )
        
        return Command(
            update={
                "scratchpad": [scratchpad_entry, error_entry],
                "is_simple_query": False,
                "confidence_score": 0.0
            },
            goto="orientation"
        )


async def orientation_node(state: ReactAgentState, config: RunnableConfig) -> Command[Literal["planner"]]:
    """
    Orientation node for situational awareness - gathers tender context and document inventory.
    """
    tender_id = state["tender_id"]
    step_num = len(state["scratchpad"]) + 1
    
    # Add orientation step to scratchpad
    orientation_entry = ScratchpadEntry(
        step=step_num,
        type="Thought",
        content="Starting orientation phase to gather tender context and document inventory"
    )
    
    try:
        # Get tender overview
        overview_result = await consult_tender_manifest.ainvoke({
            "action": "get_overview",
            "tender_id": tender_id
        })
        
        # Get document inventory
        documents_result = await consult_tender_manifest.ainvoke({
            "action": "list_documents", 
            "tender_id": tender_id
        })
        
        # Combine manifest information
        manifest_overview = {
            "overview": overview_result,
            "documents": documents_result.get("documents", []),
            "total_documents": overview_result.get("total_documents", 0)
        }
        
        # Use LLM to analyze tender structure
        analysis_prompt = f"""
        {orientation_system_prompt}
        
        TENDER OVERVIEW:
        {json.dumps(overview_result, indent=2)}
        
        DOCUMENT INVENTORY:
        {json.dumps(documents_result, indent=2)}
        
        USER QUERY: {state["user_query"]}
        
        Please provide your orientation analysis:
        """
        
        model = configurable_model.with_config(config)
        response = await model.ainvoke([
            SystemMessage(content=analysis_prompt)
        ])
        
        # Add analysis to scratchpad
        analysis_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Orientation Analysis: {response.content}"
        )
        
        return Command(
            update={
                "scratchpad": [orientation_entry, analysis_entry],
                "manifest_overview": manifest_overview
            },
            goto="planner"
        )
        
    except Exception as e:
        error_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Orientation failed with error: {str(e)}. Proceeding with limited context."
        )
        
        return Command(
            update={
                "scratchpad": [orientation_entry, error_entry],
                "manifest_overview": {}
            },
            goto="planner"
        )


async def planner_node(state: ReactAgentState, config: RunnableConfig) -> Command[Literal["tool_executor", "synthesizer"]]:
    """
    Central reasoning engine - analyzes state and determines next actions using ReAct/Chain-of-Thought.
    """
    step_num = len(state["scratchpad"]) + 1
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 10)
    
    # Check iteration limit
    if iterations >= max_iterations:
        limit_entry = ScratchpadEntry(
            step=step_num,
            type="Thought",
            content=f"Reached maximum iterations ({max_iterations}). Proceeding to synthesis."
        )
        
        return Command(
            update={
                "scratchpad": [limit_entry],
                "iterations": iterations + 1
            },
            goto="synthesizer"
        )
    
    # Add planning step to scratchpad
    planning_entry = ScratchpadEntry(
        step=step_num,
        type="Thought",
        content="Analyzing current state and planning next actions"
    )
    
    try:
        # Build context for planning
        context = {
            "user_query": state["user_query"],
            "tender_id": state["tender_id"],
            "manifest_overview": state.get("manifest_overview", {}),
            "previous_results": state.get("intermediate_results", []),
            "scratchpad_history": state.get("scratchpad", []),
            "iteration": iterations
        }
        
        planning_prompt = f"""
        {planner_system_prompt}
        
        CURRENT CONTEXT:
        User Query: {context["user_query"]}
        Tender ID: {context["tender_id"]}
        Iteration: {context["iteration"]}
        
        MANIFEST OVERVIEW:
        {json.dumps(context["manifest_overview"], indent=2)}
        
        PREVIOUS RESULTS:
        {json.dumps([r.result_data for r in context["previous_results"]], indent=2)}
        
        SCRATCHPAD HISTORY:
        {chr(10).join([f"Step {s.step} ({s.type}): {s.content}" for s in context["scratchpad_history"]])}
        
        Based on this information, what should be the next action?
        """
        
        model = configurable_model.with_config(config)
        response = await model.ainvoke([
            SystemMessage(content=planning_prompt)
        ])
        
        plan_content = response.content
        
        # Parse response to determine next action
        plan_lower = plan_content.lower()
        
        # Check if ready for synthesis
        if any(phrase in plan_lower for phrase in ["synthesize", "synthesis", "sufficient information", "ready to answer", "recommend synthesis"]):
            synthesis_entry = ScratchpadEntry(
                step=step_num + 1,
                type="Action",
                content="Decision: Sufficient information gathered. Proceeding to synthesis."
            )
            
            return Command(
                update={
                    "scratchpad": [planning_entry, synthesis_entry],
                    "current_plan": plan_content,
                    "iterations": iterations + 1
                },
                goto="synthesizer"
            )
        
        # Otherwise, plan tool execution
        action_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Action", 
            content=f"Plan: {plan_content}"
        )
        
        return Command(
            update={
                "scratchpad": [planning_entry, action_entry],
                "current_plan": plan_content,
                "iterations": iterations + 1
            },
            goto="tool_executor"
        )
        
    except Exception as e:
        error_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Planning failed with error: {str(e)}. Proceeding to synthesis."
        )
        
        return Command(
            update={
                "scratchpad": [planning_entry, error_entry],
                "iterations": iterations + 1
            },
            goto="synthesizer"
        )


async def tool_executor_node(state: ReactAgentState, config: RunnableConfig) -> Command[Literal["reflection"]]:
    """
    Tool executor node - executes the plan by calling appropriate tools.
    """
    step_num = len(state["scratchpad"]) + 1
    current_plan = state.get("current_plan", "")
    tender_id = state["tender_id"]
    
    # Add execution step to scratchpad
    execution_entry = ScratchpadEntry(
        step=step_num,
        type="Action",
        content="Executing planned tools based on current plan"
    )
    
    intermediate_results = []
    
    try:
        # Parse plan to determine which tools to execute
        # This is a simplified implementation - in production, you'd want more sophisticated plan parsing
        plan_lower = current_plan.lower()
        
        if "search" in plan_lower or "targeted_hybrid_search" in plan_lower:
            # Execute targeted search
            search_query = state["user_query"]  # Could be extracted more intelligently from plan
            
            search_results = await targeted_hybrid_search.ainvoke({
                "query": search_query,
                "tender_id": tender_id,
                "file_id_filters": None  # Could be parsed from plan
            })
            
            if search_results:
                intermediate_result = IntermediateResult(
                    tool_name="targeted_hybrid_search",
                    result_data={"search_results": search_results, "query": search_query},
                    file_sources=[r.get("file_name", "") for r in search_results if isinstance(r, dict)],
                    web_sources=[]
                )
                intermediate_results.append(intermediate_result)
        
        if "manifest" in plan_lower or "consult_tender_manifest" in plan_lower:
            # Execute manifest consultation
            manifest_results = await consult_tender_manifest.ainvoke({
                "action": "list_documents",
                "tender_id": tender_id
            })
            
            if manifest_results:
                intermediate_result = IntermediateResult(
                    tool_name="consult_tender_manifest",
                    result_data=manifest_results,
                    file_sources=[],
                    web_sources=[]
                )
                intermediate_results.append(intermediate_result)
        
        # Create observation entry
        observation_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Executed tools and gathered {len(intermediate_results)} new results"
        )
        
        return Command(
            update={
                "scratchpad": [execution_entry, observation_entry],
                "intermediate_results": intermediate_results
            },
            goto="reflection"
        )
        
    except Exception as e:
        error_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Tool execution failed with error: {str(e)}"
        )
        
        return Command(
            update={
                "scratchpad": [execution_entry, error_entry],
                "intermediate_results": intermediate_results
            },
            goto="reflection"
        )


async def reflection_node(state: ReactAgentState, config: RunnableConfig) -> Command[Literal["planner", "synthesizer"]]:
    """
    Reflection node - evaluates tool effectiveness and determines next steps.
    """
    step_num = len(state["scratchpad"]) + 1
    recent_results = state.get("intermediate_results", [])
    
    # Add reflection step to scratchpad
    reflection_entry = ScratchpadEntry(
        step=step_num,
        type="Thought",
        content="Reflecting on recent tool results and planning next steps"
    )
    
    try:
        # Build context for reflection
        results_summary = []
        for result in recent_results[-3:]:  # Look at last 3 results
            results_summary.append({
                "tool": result.tool_name,
                "data_size": len(str(result.result_data)),
                "sources": len(result.file_sources) + len(result.web_sources)
            })
        
        reflection_prompt = f"""
        {reflection_system_prompt}
        
        USER QUERY: {state["user_query"]}
        
        RECENT RESULTS SUMMARY:
        {json.dumps(results_summary, indent=2)}
        
        SCRATCHPAD HISTORY (last 5 steps):
        {chr(10).join([f"Step {s.step} ({s.type}): {s.content}" for s in state.get("scratchpad", [])[-5:]])}
        
        Evaluate the progress and recommend next steps:
        """
        
        model = configurable_model.with_config(config)
        response = await model.ainvoke([
            SystemMessage(content=reflection_prompt)
        ])
        
        reflection_content = response.content
        reflection_lower = reflection_content.lower()
        
        # Determine next action based on reflection
        if "synthesize" in reflection_lower or "sufficient" in reflection_lower:
            route = "synthesizer"
            decision = "SYNTHESIZE"
        else:
            route = "planner"
            decision = "CONTINUE"
        
        decision_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Reflection Decision: {decision} - {reflection_content[:200]}..."
        )
        
        return Command(
            update={
                "scratchpad": [reflection_entry, decision_entry]
            },
            goto=route
        )
        
    except Exception as e:
        error_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Reflection failed with error: {str(e)}. Continuing to planner."
        )
        
        return Command(
            update={
                "scratchpad": [reflection_entry, error_entry]
            },
            goto="planner"
        )


async def synthesizer_node(state: ReactAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Synthesizer node - generates final comprehensive answer with citations.
    """
    step_num = len(state["scratchpad"]) + 1
    
    # Add synthesis step to scratchpad
    synthesis_entry = ScratchpadEntry(
        step=step_num,
        type="Action",
        content="Synthesizing final answer from all gathered information"
    )
    
    try:
        # Gather all information for synthesis
        all_results = state.get("intermediate_results", [])
        user_query = state["user_query"]
        
        # Build comprehensive context
        file_sources = set()
        web_sources = set()
        result_data = []
        
        for result in all_results:
            result_data.append({
                "tool": result.tool_name,
                "data": result.result_data
            })
            file_sources.update(result.file_sources)
            web_sources.update(result.web_sources)
        
        synthesis_prompt = f"""
        {synthesizer_system_prompt}
        
        USER QUERY: {user_query}
        
        GATHERED INFORMATION:
        {json.dumps(result_data, indent=2)}
        
        AVAILABLE FILE SOURCES: {list(file_sources)}
        AVAILABLE WEB SOURCES: {list(web_sources)}
        
        Please provide a comprehensive, well-sourced answer:
        """
        
        model = configurable_model.with_config(config)
        response = await model.ainvoke([
            SystemMessage(content=synthesis_prompt)
        ])
        
        final_answer = response.content
        
        # Add completion to scratchpad
        completion_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content="Final answer synthesized successfully"
        )
        
        # Create final AI message with the answer
        final_message = AIMessage(content=final_answer)
        
        return {
            "scratchpad": [synthesis_entry, completion_entry],
            "messages": [final_message]
        }
        
    except Exception as e:
        error_entry = ScratchpadEntry(
            step=step_num + 1,
            type="Observation",
            content=f"Synthesis failed with error: {str(e)}"
        )
        
        # Provide fallback answer
        fallback_message = AIMessage(
            content=f"I apologize, but I encountered an error while synthesizing the final answer. "
                   f"Based on the information gathered, I can provide a partial response to your query: {state['user_query']}. "
                   f"Error details: {str(e)}"
        )
        
        return {
            "scratchpad": [synthesis_entry, error_entry],
            "messages": [fallback_message]
        }

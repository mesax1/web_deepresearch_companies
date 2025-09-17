"""Foundational tools for the LangGraph react agent."""

import asyncio
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import json

# Initialize configurable model for tool usage
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


class TenderOverview(BaseModel):
    """Schema for tender overview data."""
    
    tender_id: str = Field(description="Unique identifier for the tender")
    summary: str = Field(description="10-page structured summary with overview, evaluation criteria, and timelines")
    total_documents: int = Field(description="Total number of documents in the tender")


class DocumentInventoryItem(BaseModel):
    """Schema for a document in the tender inventory."""
    
    file_id: str = Field(description="Primary key for document retrieval")
    file_name: str = Field(description="Exact file name")
    document_type: str = Field(description="Type: Rammeaftale, Bilag, Pricing, etc.")
    summary: str = Field(description="10-line concise overview of document purpose")


class TenderManifest(BaseModel):
    """Complete tender manifest structure."""
    
    overview: TenderOverview
    documents: List[DocumentInventoryItem]


class SearchResult(BaseModel):
    """Result from targeted hybrid search."""
    
    content: str = Field(description="Relevant content from the search")
    file_id: str = Field(description="Source file identifier")
    file_name: str = Field(description="Source file name")
    confidence_score: float = Field(description="Relevance confidence score")


class AnalysisResult(BaseModel):
    """Result from iterative document analysis."""
    
    summary: str = Field(description="Analysis summary")
    key_findings: List[str] = Field(description="Key findings extracted")
    relevant_sections: List[str] = Field(description="Relevant document sections")
    file_id: str = Field(description="Analyzed file identifier")


# Mock data store for demonstration (in production, this would connect to actual databases)
MOCK_TENDER_MANIFEST = {
    "tender_123": TenderManifest(
        overview=TenderOverview(
            tender_id="tender_123",
            summary="IT Infrastructure Modernization Tender - 10-page overview including evaluation criteria focusing on technical capability, pricing structure, and implementation timeline. Key requirements include cloud migration, security compliance, and 24/7 support.",
            total_documents=15
        ),
        documents=[
            DocumentInventoryItem(
                file_id="doc_001",
                file_name="01_Rammeaftale_Hovedaftale.pdf",
                document_type="Rammeaftale",
                summary="Main framework agreement defining overall terms, conditions, penalties, and legal obligations for the IT infrastructure project."
            ),
            DocumentInventoryItem(
                file_id="doc_002", 
                file_name="02_Bilag_A_Tekniske_Krav.pdf",
                document_type="Bilag",
                summary="Technical requirements specification covering cloud infrastructure, security standards, performance metrics, and compliance requirements."
            ),
            DocumentInventoryItem(
                file_id="doc_003",
                file_name="03_Bilag_B_Prisstruktur.pdf", 
                document_type="Pricing",
                summary="Pricing structure guidelines including cost models, payment terms, penalties for SLA breaches, and pricing evaluation criteria."
            )
        ]
    )
}

MOCK_DOCUMENT_CONTENT = {
    "doc_001": "Framework Agreement Main Content: This agreement establishes penalties of 2% of monthly service fee for each hour of unplanned downtime exceeding 4 hours per month. Performance requirements include 99.5% uptime SLA...",
    "doc_002": "Technical Requirements: Cloud infrastructure must comply with ISO 27001 and GDPR. Minimum 99.5% availability required. Disaster recovery with RTO < 4 hours and RPO < 1 hour...",
    "doc_003": "Pricing Structure: Base monthly fee structure ranges from 50,000-200,000 DKK depending on service level. Additional costs for premium support and extended SLA coverage..."
}


@tool
async def consult_tender_manifest(
    action: str,
    tender_id: str,
    user_references: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Consult the tender manifest for rapid access to metadata and summaries.
    
    Args:
        action: Action to perform - 'get_overview', 'list_documents', or 'map_names_to_ids'
        tender_id: Identifier for the tender
        user_references: List of user reference strings to map to file IDs (only for map_names_to_ids action)
    
    Returns:
        Dictionary containing the requested information
    """
    if tender_id not in MOCK_TENDER_MANIFEST:
        return {"error": f"Tender {tender_id} not found"}
    
    manifest = MOCK_TENDER_MANIFEST[tender_id]
    
    if action == "get_overview":
        return {
            "tender_id": manifest.overview.tender_id,
            "summary": manifest.overview.summary,
            "total_documents": manifest.overview.total_documents
        }
    
    elif action == "list_documents":
        return {
            "documents": [
                {
                    "file_id": doc.file_id,
                    "file_name": doc.file_name,
                    "document_type": doc.document_type,
                    "summary": doc.summary
                }
                for doc in manifest.documents
            ]
        }
    
    elif action == "map_names_to_ids":
        if not user_references:
            return {"error": "No user references provided for mapping"}
        
        # Simple fuzzy matching for demonstration
        mapped_results = []
        for ref in user_references:
            ref_lower = ref.lower()
            for doc in manifest.documents:
                if (ref_lower in doc.file_name.lower() or 
                    ref_lower in doc.document_type.lower() or
                    any(word in doc.summary.lower() for word in ref_lower.split())):
                    mapped_results.append({
                        "user_reference": ref,
                        "file_id": doc.file_id,
                        "file_name": doc.file_name,
                        "confidence": 0.8  # Mock confidence score
                    })
                    break
        
        return {"mapped_files": mapped_results}
    
    else:
        return {"error": f"Unknown action: {action}"}


@tool
async def targeted_hybrid_search(
    query: str,
    tender_id: str,
    file_id_filters: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Primary RAG workhorse for deep content extraction using hybrid search.
    
    Args:
        query: Search query string
        tender_id: Identifier for the tender
        file_id_filters: Optional list of file IDs to restrict search scope
    
    Returns:
        List of relevant search results with content and metadata
    """
    # Mock implementation - in production this would use Qdrant + Cohere rerank
    results = []
    
    # Get available documents
    if tender_id not in MOCK_TENDER_MANIFEST:
        return [{"error": f"Tender {tender_id} not found"}]
    
    manifest = MOCK_TENDER_MANIFEST[tender_id]
    documents_to_search = manifest.documents
    
    # Apply file filters if provided
    if file_id_filters:
        documents_to_search = [doc for doc in documents_to_search if doc.file_id in file_id_filters]
    
    # Mock search logic
    query_lower = query.lower()
    for doc in documents_to_search:
        content = MOCK_DOCUMENT_CONTENT.get(doc.file_id, "No content available")
        
        # Simple relevance scoring based on keyword matching
        relevance_score = 0.0
        for word in query_lower.split():
            if word in content.lower():
                relevance_score += 0.2
            if word in doc.summary.lower():
                relevance_score += 0.1
        
        if relevance_score > 0.1:  # Threshold for relevance
            results.append({
                "content": content,
                "file_id": doc.file_id,
                "file_name": doc.file_name,
                "confidence_score": min(relevance_score, 1.0),
                "document_type": doc.document_type
            })
    
    # Sort by confidence score
    results.sort(key=lambda x: x["confidence_score"], reverse=True)
    return results[:5]  # Return top 5 results


@tool
async def iterative_document_analyzer(
    file_id: str,
    analysis_objective: str,
    tender_id: str
) -> Dict[str, Any]:
    """
    Analyze or summarize large documents using MapReduce strategy.
    
    Args:
        file_id: Identifier for the specific document to analyze
        analysis_objective: The analysis goal (e.g., "Extract all penalty clauses")
        tender_id: Identifier for the tender
    
    Returns:
        Dictionary containing analysis results
    """
    # Mock implementation - in production this would implement MapReduce over document chunks
    if tender_id not in MOCK_TENDER_MANIFEST:
        return {"error": f"Tender {tender_id} not found"}
    
    content = MOCK_DOCUMENT_CONTENT.get(file_id, "No content available for this file")
    
    if content == "No content available for this file":
        return {"error": f"File {file_id} not found"}
    
    # Mock analysis using LLM
    analysis_prompt = f"""
    Analyze the following document content based on this objective: {analysis_objective}
    
    Document Content:
    {content}
    
    Please provide:
    1. A summary of findings related to the objective
    2. Key findings as bullet points
    3. Relevant sections that support the analysis
    """
    
    # This would be an actual LLM call in production
    mock_analysis = {
        "summary": f"Analysis completed for {analysis_objective}. Found relevant information about penalties, SLA requirements, and compliance standards.",
        "key_findings": [
            "2% monthly service fee penalty for unplanned downtime exceeding 4 hours",
            "99.5% uptime SLA requirement",
            "ISO 27001 and GDPR compliance mandatory",
            "Disaster recovery RTO < 4 hours, RPO < 1 hour"
        ],
        "relevant_sections": [
            "Section 3.2: Performance Requirements and Penalties",
            "Section 4.1: Service Level Agreements",
            "Section 5.3: Compliance and Security Standards"
        ],
        "file_id": file_id
    }
    
    return mock_analysis


@tool
async def web_search(query: str) -> List[Dict[str, Any]]:
    """
    Search external sources for regulations, legal definitions, and market intelligence.
    
    Args:
        query: Search query for external information
    
    Returns:
        List of web search results with content and sources
    """
    # Mock implementation - in production this would use Tavily API or similar
    mock_results = [
        {
            "title": f"Search Results for: {query}",
            "content": f"External information related to {query}. This would include relevant regulations, legal definitions, and market intelligence.",
            "url": "https://example.com/search-result-1",
            "source": "External Database"
        },
        {
            "title": f"Regulatory Information: {query}",
            "content": f"Regulatory context and legal framework information for {query}.",
            "url": "https://example.com/search-result-2", 
            "source": "Legal Database"
        }
    ]
    
    return mock_results


@tool
async def wait_for_user_input(clarification_question: str) -> str:
    """
    Wait for user input to resolve ambiguity or get clarification.
    
    Args:
        clarification_question: Question to ask the user for clarification
    
    Returns:
        User's response (in production, this would use LangGraph interrupts)
    """
    # In production, this would use LangGraph's interrupt mechanism
    # For now, we'll return a mock response
    return f"Mock user response to: {clarification_question}"


# Tool list for easy access
REACT_TOOLS = [
    consult_tender_manifest,
    targeted_hybrid_search,
    iterative_document_analyzer,
    web_search,
    wait_for_user_input
]

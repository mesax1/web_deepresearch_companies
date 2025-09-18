"""Foundational tools for the LangGraph react agent."""

import os
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from pymongo import MongoClient
from bson import ObjectId

from tool_utils import getVectorStore, CustomRetriever, get_proposal_id, get_proposal_files, get_proposal_summary, get_proposal_total_documents, get_proposal_compliance_matrix_analysis, get_proposal_files_summary

from dotenv import load_dotenv
load_dotenv()

uri = os.getenv("MONGODB_URI")
mongo_client = MongoClient(uri)
db = mongo_client["org_1"]
proposals = db["proposals"]
proposal_files = db["proposal_files"]

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


class FileMapping(BaseModel):
    """Schema for mapping user references to file IDs with confidence scores."""
    
    user_reference: str = Field(description="The original user reference string")
    file_id: str = Field(description="The mapped file ID")
    file_name: str = Field(description="The mapped file name")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of why this mapping was chosen")


class FileMappings(BaseModel):
    """Schema for the complete file mapping result."""
    
    mapped_files: List[FileMapping] = Field(description="List of file mappings with confidence scores")


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
    """Consult the tender manifest for rapid access to metadata and summaries.
    
    Args:
        action: Action to perform - 'get_overview', 'list_documents', or 'map_names_to_ids'
        tender_id: Identifier for the tender
        user_references: List of user reference strings to map to file IDs (only for map_names_to_ids action)
    
    Returns:
        Dictionary containing the requested information
    """
    proposal = proposals.find_one({ "_id" : ObjectId(tender_id) })
    if proposal is None:
        return {"error": f"Tender {tender_id} not found"}

    summary = ""
    analysis = proposal['compliance_matrix_analysis']
    for _, v in analysis.items():
        summary += f"{v}\n"

    file_id = proposal['requirement_cluster_id']
    files = proposal_files.find({ "cluster_id" : file_id })
    total_documents = len(files)
    
    if action == "get_overview":
        return {
            "tender_id": proposal._id,
            "summary": summary,
            "total_documents": total_documents
        }
    
    elif action == "list_documents":
        return {
            "documents": [
                {
                    "file_id": doc._id,
                    "file_name": doc.file_name,
                    "document_type": doc.file_extension,
                    "summary": doc.requirements_summary["en"] if "en" in doc.requirements_summary else list(doc.requirements_summary.values())[0]
                }
                for doc in files
            ]
        }
    
    elif action == "map_names_to_ids":
        if not user_references:
            return {"error": "No user references provided for mapping"}
        
        document_info = []
        for doc in files:
            doc_summary = doc.requirements_summary.get("en", list(doc.requirements_summary.values())[0] if doc.requirements_summary else "")
            document_info.append({
                "file_id": str(doc._id),
                "file_name": doc.file_name,
                "file_extension": doc.file_extension,
                "summary": doc_summary
            })

        prompt = f"""
You are tasked with mapping user reference strings to the most appropriate file IDs from a tender document collection.
User References to Map: {', '.join(user_references)}
Available Documents:
{chr(10).join([f"- ID: {doc['file_id']}, Name: {doc['file_name']}, Type: {doc['file_extension']}, Summary: {doc['summary']}" for doc in document_info])}
For each user reference, find the best matching document and provide:
1. The exact file_id from the available documents
2. A confidence score between 0.0 and 1.0 (1.0 = perfect match, 0.0 = no match)
3. Brief reasoning for your choice
Consider matches based on:
- Exact filename matches
- Partial filename matches
- File extension matches
- Content summary relevance
- Semantic similarity
Return your response in the following JSON format:
{{
  "mapped_files": [
    {{
      "user_reference": "original reference string",
      "file_id": "matched_file_id",
      "file_name": "matched_file_name",
      "confidence": 0.95,
      "reasoning": "Brief explanation of why this mapping was chosen"
    }}
  ]
}}
"""  
        try:
            model_with_structure = configurable_model.with_structured_output(FileMappings)
            message = HumanMessage(content=prompt)
            response = model_with_structure.invoke([message])
            
            mapped_results = []
            for mapping in response.mapped_files:
                mapped_results.append({
                    "user_reference": mapping.user_reference,
                    "file_id": mapping.file_id,
                    "file_name": mapping.file_name,
                    "confidence": mapping.confidence,
                    "reasoning": mapping.reasoning
                })
            
            return {"mapped_files": mapped_results}
            
        except Exception as e:
            mapped_results = []
            for ref in user_references:
                ref_lower = ref.lower()
                best_match = None
                best_confidence = 0.0
                
                for doc in files:
                    confidence = 0.0
                    doc_summary = doc.requirements_summary.get("en", list(doc.requirements_summary.values())[0] if doc.requirements_summary else "")
                    
                    if ref_lower in doc.file_name.lower():
                        confidence += 0.6
                    if ref_lower in doc.file_extension.lower():
                        confidence += 0.4
                    if any(word in doc_summary.lower() for word in ref_lower.split()):
                        confidence += 0.3
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = doc
                
                if best_match and best_confidence > 0.1:
                    mapped_results.append({
                        "user_reference": ref,
                        "file_id": str(best_match._id),
                        "file_name": best_match.file_name,
                        "confidence": min(best_confidence, 1.0),
                        "reasoning": f"Fallback matching based on filename/content similarity (confidence: {best_confidence:.2f})"
                    })
            
            return {"mapped_files": mapped_results}
    
    else:
        return {"error": f"Unknown action: {action}"}

@tool
async def targeted_hybrid_search(
    query: str,
    tender_id: str,
    file_id_filters: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Primary RAG workhorse for deep content extraction using hybrid search.
    
    Args:
        query: Search query string
        tender_id: Identifier for the tender
        file_id_filters: Optional list of file IDs to restrict search scope
    
    Returns:
        List of relevant search results with content and metadata
    """
    try:
        proposal = proposals.find_one({ "_id" : ObjectId(tender_id) })
        file_id = proposal['requirement_cluster_id']
        current_filter = Filter(
            must=[
                FieldCondition(
                    key="cluster_id",
                    match=MatchValue(value=file_id)
                )
            ]
        )
        vectorstore = getVectorStore("proposal_testing")
        retriever = CustomRetriever(
            [vectorstore.as_retriever(search_kwargs={"filter": current_filter})],
            k=50,
            p=10,
        )
        
        documents = retriever.get_docs_without_callbacks(query)
        
        context = "\n\n".join([
            f"**Document: {doc.metadata.get('title', 'Unknown')}**\n{doc.page_content}"
            for doc in documents
        ])
        
        return {
            "context": context,
            "documents": documents,
            "num_results": len(documents)
        }
    except Exception as e:
        return {"error": f"Chunk search failed: {str(e)}", "documents": []}

@tool
async def iterative_document_analyzer(
    file_id: str,
    analysis_objective: str,
    tender_id: str
) -> Dict[str, Any]:
    """Analyze or summarize large documents using MapReduce strategy.
    
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
    """Search external sources for regulations, legal definitions, and market intelligence.
    
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


# @tool
# async def wait_for_user_input(clarification_question: str) -> str:
#     """
#     Wait for user input to resolve ambiguity or get clarification.
    
#     Args:
#         clarification_question: Question to ask the user for clarification
    
#     Returns:
#         User's response (in production, this would use LangGraph interrupts)
#     """
#     # In production, this would use LangGraph's interrupt mechanism
#     # For now, we'll return a mock response
#     return f"Mock user response to: {clarification_question}"


# Tool list for easy access
REACT_TOOLS = [
    consult_tender_manifest,
    targeted_hybrid_search,
    iterative_document_analyzer,
    web_search,
    # wait_for_user_input
]

"""
Multi-Agent Research System with Self-Correction Loop

This module implements an autonomous research agent that:
- Retrieves relevant sources (web + user uploads)
- Extracts content via OCR/parsing
- Generates structured competitive intelligence reports
- Self-evaluates and iterates until quality thresholds are met

Architecture:
    StateGraph (LangGraph)
        ├── Researcher: Source discovery
        ├── OCR Engine: Document extraction
        ├── Analyst: Report synthesis
        └── Evaluator: Quality control & routing

Author: Hosseini
Version: 1.0
Date: January 2026
"""

import os
import logging
import time
from typing import TypedDict, List, Optional, Literal
from dataclasses import dataclass
from functools import wraps
from contextlib import contextmanager
from string import Template

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from IPython.display import Image, display

# External tools
from app.tools import search_competitors, process_visual_url, query_knowledge_base


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    MAX_RETRIES: int = 2
    MODEL_NAME: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0
    MAX_SOURCES: int = 2
    RETRIEVAL_TOP_K: int = 20
    MIN_CONTEXT_LENGTH: int = 50
    LOG_LEVEL: str = "INFO"


config = AgentConfig()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT CHECK
# =============================================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from .env")

if not os.getenv("TAVILY_API_KEY"):
    logger.warning("TAVILY_API_KEY missing — web search may fail.")


# =============================================================================
# STATE & SCHEMAS
# =============================================================================

class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    task: str
    image_urls: List[str]
    report: str
    messages: List[str]
    request_id: str
    feedback: str
    retry_count: int
    quality_score: Optional[int]
    decision: Optional[str]


class EvaluationResult(BaseModel):
    """Structured evaluator output."""
    score: int = Field(ge=1, le=10)
    feedback: str
    decision: Literal["APPROVE", "RETRY_RESEARCH", "RETRY_ANALYSIS"]


# =============================================================================
# MODEL
# =============================================================================

llm = ChatGoogleGenerativeAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    max_retries=2,
)


# =============================================================================
# UTILITIES
# =============================================================================

def update_progress_sync(state: AgentState, message: str) -> None:
    """Push progress updates to the API (best-effort)."""
    try:
        from app.api import progress_tracker
        request_id = state.get("request_id")
        if request_id:
            progress_tracker.add_update_sync(request_id, message)
    except Exception as e:
        logger.warning(f"Progress update failed: {e}")


@contextmanager
def error_boundary(step: str, state: AgentState):
    """Unified error handler for all nodes."""
    try:
        yield
    except Exception as e:
        logger.exception(f"{step} failed")
        update_progress_sync(state, f"{step} error: {e}")
        raise


def track_node_execution(func):
    """Logs execution time and failures for each node."""
    @wraps(func)
    def wrapper(state: AgentState):
        start = time.time()
        name = func.__name__.replace("_node", "").upper()
        try:
            result = func(state)
            logger.info(f"{name} completed in {time.time() - start:.2f}s")
            return result
        except Exception:
            logger.error(f"{name} failed after {time.time() - start:.2f}s")
            raise
    return wrapper


def score_url(url: str) -> int:
    """Heuristic scoring for source quality."""
    score = 0
    u = url.lower()

    if ".pdf" in u:
        score += 10
    if any(ext in u for ext in [".png", ".jpg", ".jpeg"]):
        score += 5
    if any(dom in u for dom in [".gov", ".edu", ".org"]):
        score += 3
    if any(bad in u for bad in ["reddit", "quora", "forum"]):
        score -= 5
    if any(pay in u for pay in ["paywall", "subscription"]):
        score -= 3

    return score


# =============================================================================
# NODES
# =============================================================================

@track_node_execution
def research_node(state: AgentState) -> dict:
    """Finds and ranks relevant sources."""
    with error_boundary("Research", state):
        iteration = state.get("retry_count", 0)
        feedback = state.get("feedback", "")
        
        logger.info(f"Research iteration {iteration + 1} for: {state['task']}")
        
        # Check for user-uploaded files
        existing_files = state.get("image_urls", [])
        if existing_files:
            logger.info(f"User uploaded {len(existing_files)} file(s)")
            update_progress_sync(state, f"Analyzing {len(existing_files)} uploaded file(s) + web search...")
        else:
            update_progress_sync(state, "Searching for relevant sources...")
        
        # Build adaptive search queries
        base_queries = [
            f"{state['task']} comprehensive analysis filetype:pdf",
            f"{state['task']} market research report 2025 2026",
            f"{state['task']} competitive intelligence pricing cost"
        ]
        
        # Inject feedback into search strategy
        if feedback:
            logger.info(f"Incorporating feedback: {feedback[:100]}...")
            base_queries.insert(0, f"{state['task']} {feedback}")
        
        # Execute searches
        all_results = []
        for query in base_queries:
            try:
                update_progress_sync(state, f"🔎 Query: '{query[:50]}...'")
                results = search_competitors(query)
                if results:
                    all_results.extend(results)
                    logger.info(f"Query returned {len(results)} results")
            except Exception as e:
                logger.warning(f"Search query failed: {query} - {e}")
        
        if not all_results and not existing_files:
            logger.warning("No search results found and no uploaded files")
            return {
                "image_urls": [],
                "messages": ["Search yielded no results"]
            }
        
        # Deduplicate and rank URLs
        unique_results = {r['url']: r for r in all_results}.values()
        candidates = sorted(
            [r['url'] for r in unique_results],
            key=score_url,
            reverse=True
        )[:config.MAX_SOURCES]
        
        # Combine uploaded files with web search results
        final_urls = existing_files + candidates
        
        logger.info(f"Final source list: {len(final_urls)} items "
                   f"({len(existing_files)} uploaded, {len(candidates)} web)")
        update_progress_sync(state, f"Aggregated {len(final_urls)} source(s)")
        
        return {
            "image_urls": final_urls,
            "messages": [f"Retrieved {len(candidates)} web sources, "
                        f"{len(existing_files)} uploads"]
        }


@track_node_execution
def ocr_node(state: AgentState) -> dict:
    """
    Document extraction and processing node.
    """
    with error_boundary("OCR", state):
        logger.info("Starting OCR/extraction pipeline")
        update_progress_sync(state, "Starting document extraction...")
        
        urls = state.get('image_urls', [])
        if not urls:
            logger.warning("No URLs found to process")
            update_progress_sync(state, "No documents to process")
            return {"messages": ["Skipped OCR: No URLs found"]}
        
        update_progress_sync(state, f"Processing {len(urls)} document(s)...")
        
        logs = []
        processed_count = 0
        
        # Process sources with rate limiting
        for idx, url in enumerate(urls[:config.MAX_SOURCES], 1):
            try:
                update_progress_sync(state, f"Reading document {idx}/{min(len(urls), config.MAX_SOURCES)}...")
                status = process_visual_url(url)
                logs.append(status)
                processed_count += 1
                logger.info(f"Processed {idx}/{len(urls)}: {status}")
            except Exception as e:
                error_msg = f"Failed to process {url}: {str(e)}"
                logger.error(error_msg)
                logs.append(error_msg)
        
        update_progress_sync(state, f"Processed {processed_count}/{len(urls)} documents")
        return {"messages": logs}


@track_node_execution
def analysis_node(state: AgentState) -> dict:
    """Generates the final market analysis report."""
    with error_boundary("Analysis", state):
        feedback = state.get("feedback", "")
        
        logger.info("Querying knowledge base for context")
        update_progress_sync(state, "Retrieving context from vector database...")
        
        # Query vector database
        context = query_knowledge_base(
            f"Comprehensive information on {state['task']} with source citations"
        )
        
        # Validate context quality
        has_context = context and len(context) > config.MIN_CONTEXT_LENGTH
        if not has_context:
            logger.warning("Limited context retrieved, using general knowledge")
            context = "No specific data extracted. Analysis will rely on expert knowledge."
            update_progress_sync(state, "Limited data, using general expertise")
        else:
            logger.info(f"Retrieved {len(context)} characters of context")
            update_progress_sync(state, f"Retrieved {len(context)} chars of context")
        
        # Build feedback section
        feedback_section = ""
        if feedback:
            feedback_section = f"""
                🚨 **REVISION REQUIRED**
                Previous report was rejected.
                **Feedback to address**: "{feedback}"
                Ensure this specific issue is resolved in this version.
            """
        
        # Construct prompt using template
        prompt_template = Template("""
            You are a Senior Market Research Analyst for a top-tier Venture Capital firm.

            **OBJECTIVE**: Create a structured competitive intelligence report on: '$task'

            $feedback_section

            **RETRIEVED CONTEXT**:
            $context

            **CRITICAL INSTRUCTIONS**:
            1. PRIMARY GOAL: Answer the user's specific task comprehensively
            2. Analyze the RETRIEVED CONTEXT carefully:
            - If relevant to the task, use it and cite sources
            - If irrelevant or off-topic, IGNORE IT and rely on expert knowledge
            3. Do NOT write about irrelevant data just because it was retrieved
            4. Provide concrete, actionable insights backed by data where available

            **FORMAT REQUIREMENTS** (Markdown with emojis):

            # $task - Market Analysis Report

            ## 📋 Executive Summary
            [2-3 sentence high-level strategic takeaway]

            ## ⚡ Market & Opportunity Analysis
            - **Key Requirements**: [What capabilities/resources are needed?]
            - **Market Demand**: [Why is this relevant now? Market size/growth?]
            - **Differentiation**: [What makes winners stand out?]

            ## 💰 Financial Model
            - **Revenue Potential**: [Cite specific numbers if available, or provide market ranges]
            - **Cost Structure**: [Initial investment, operational costs]
            - **Unit Economics**: [Margins, break-even analysis]

            ## 🧭 SWOT Analysis
            ### ✅ Strengths
            [3-4 key strengths]

            ### ⚠️ Weaknesses
            [3-4 vulnerabilities]

            ### 🚀 Opportunities
            [3-4 growth vectors]

            ### 🛑 Threats
            [3-4 risk factors]

            ## 🎯 Strategic Recommendation
            [One clear, actionable conclusion for stakeholders]

            ---
            *Sources: [List key sources if cited]*
        """)
        
        prompt = prompt_template.substitute(
            task=state['task'],
            feedback_section=feedback_section,
            context=context
        )
        
        logger.info("Generating report with LLM")
        update_progress_sync(state, "Generating final report...")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"Report generated: {len(response.content)} characters")
            update_progress_sync(state, "Report generation complete")
            return {"report": response.content}
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            update_progress_sync(state, f"{error_msg}")
            return {"report": error_msg}


@track_node_execution
def evaluator_node(state: AgentState) -> dict:
    """Scores report quality and routes the workflow."""
    with error_boundary("Evaluation", state):
        logger.info("⚖️ Evaluating report quality")
        update_progress_sync(state, "Evaluating report quality...")
        
        report = state.get("report", "")
        task = state.get("task", "")
        retry_count = state.get("retry_count", 0)
        
        # Configure structured output
        evaluator = llm.with_structured_output(EvaluationResult)
        
        prompt = f"""
            Evaluate this research report for the task: "{task}".

            REPORT TO EVALUATE:
            {report}

            EVALUATION CRITERIA:
            1. **Data Grounding**: Is it backed by concrete data/sources? (Not generic platitudes)
            2. **Task Alignment**: Does it directly answer the user's specific question?
            3. **Completeness**: Are all required sections present with substance?
            4. **Actionability**: Does it provide clear, specific insights?

            SCORING RUBRIC (1-10):
            - 1-3: Severely lacking, major issues
            - 4-6: Adequate but missing key elements
            - 7-8: Good quality, minor improvements needed
            - 9-10: Excellent, comprehensive and actionable

            DECISION LOGIC:
            - If score >= 7 AND all sections complete → **APPROVE**
            - If missing critical data/sources → **RETRY_RESEARCH**
            - If structure/logic/clarity issues → **RETRY_ANALYSIS**

            Current iteration: {retry_count + 1}/{config.MAX_RETRIES + 1}

            Provide your evaluation.
        """
        
        try:
            result = evaluator.invoke(prompt)
            logger.info(f"Evaluation: Score={result.score}, Decision={result.decision}")
            
            return {
                "quality_score": result.score,
                "feedback": result.feedback,
                "decision": result.decision,
                "retry_count": retry_count + 1,
                "messages": [f"Decision: {result.decision} (Score: {result.score}/10)"]
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Fail-safe: approve if evaluation fails to prevent infinite loops
            return {
                "quality_score": 7,
                "feedback": f"Evaluation failed: {str(e)}",
                "decision": "APPROVE",
                "retry_count": retry_count + 1,
                "messages": ["Evaluation failed, auto-approving"]
            }


# =============================================================================
# ROUTING
# =============================================================================

def decide_next_step(state: AgentState) -> str:
    """Routes based on evaluator decision."""
    retry_count = state.get("retry_count", 0)
    decision = state.get("decision", "")
    
    logger.info(f"Routing decision: {decision} (Retry {retry_count}/{config.MAX_RETRIES})")
    
    # Safety brake: prevent infinite loops
    if retry_count > config.MAX_RETRIES:
        logger.warning(f"Max retries ({config.MAX_RETRIES}) reached. Terminating.")
        return END
    
    # Route based on evaluator decision
    if decision == "APPROVE":
        logger.info("Report approved. Workflow complete.")
        return END
    elif decision == "RETRY_RESEARCH":
        logger.info("Retrying research phase...")
        return "researcher"
    elif decision == "RETRY_ANALYSIS":
        logger.info("Retrying analysis phase...")
        return "analyst"
    
    # Default: end if decision unclear
    logger.warning("Unclear decision, terminating workflow")
    return END


# =============================================================================
# GRAPH
# =============================================================================

def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph workflow."""
    g = StateGraph(AgentState)

    g.add_node("researcher", research_node)
    g.add_node("ocr_engine", ocr_node)
    g.add_node("analyst", analysis_node)
    g.add_node("evaluator", evaluator_node)

    g.set_entry_point("researcher")
    g.add_edge("researcher", "ocr_engine")
    g.add_edge("ocr_engine", "analyst")
    g.add_edge("analyst", "evaluator")

    g.add_conditional_edges(
        "evaluator",
        decide_next_step,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            END: END,
        },
    )

    return g.compile()


app_graph = build_graph()

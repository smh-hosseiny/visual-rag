import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import the tools from app/tools.py
from app.tools import search_competitors, process_visual_url, query_knowledge_base

# Load environment variables
load_dotenv()

# --- VALIDATION ---
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY is missing from .env")
if not os.getenv("TAVILY_API_KEY"):
    print("⚠️  Warning: TAVILY_API_KEY is missing. Search WILL fail.")

# 1. Define Agent State
class AgentState(TypedDict):
    task: str           # User input
    image_urls: List[str] 
    report: str         # Final Output
    messages: List[str] # Logs
    request_id: str     # For progress tracking

# 2. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    max_retries=2
)

# 3. Progress tracking helper
def update_progress_sync(state: AgentState, message: str):
    """Update progress for the current request"""
    try:
        from app.api import progress_updates
        request_id = state.get("request_id")
        if request_id and request_id in progress_updates:
            from datetime import datetime
            progress_updates[request_id].append({
                "step": message,
                "status": "running",
                "timestamp": datetime.now().isoformat()
            })
            print(f"📍 Progress [{request_id}]: {message}")
    except Exception as e:
        print(f"⚠️ Progress update failed: {e}")

# 4. Node Functions

def research_node(state: AgentState):
    print(f"--- 🕵️ RESEARCHING: {state['task']} ---")
    
    # Grab any user-uploaded files first
    existing_files = state.get("image_urls", [])
    
    if existing_files:
        print(f"📂 User uploaded {len(existing_files)} file(s). Enhancing with web research...")
        update_progress_sync(state, f"📂 Analyzing uploaded file(s) + searching web...")
    else:
        update_progress_sync(state, "🕵️ Searching for relevant sources...")

    # --- ENHANCED: Generalized "Sniper" Search Strategy ---
    # We broadened the queries to work for non-hardware topics too (like side hustles)
    queries = [
        f"{state['task']} comprehensive guide filetype:pdf",
        f"{state['task']} market analysis price cost report"
    ]
    
    all_results = []
    
    try:
        for q in queries:
            update_progress_sync(state, f"🔎 Sniper Search: '{q}'")
            results = search_competitors(q)
            if results:
                all_results.extend(results)
                
    except Exception as e:
        print(f"❌ Search Error: {e}")
        update_progress_sync(state, f"❌ Search failed: {str(e)}")
        if existing_files:
            return {"image_urls": existing_files, "messages": ["Search failed, using uploaded files only"]}
        return {"messages": ["Search failed"], "image_urls": []}

    # Deduplicate results by URL
    unique_results = {r['url']: r for r in all_results}.values()
    
    candidates = []
    for r in unique_results:
        url = r.get('url', '')
        # 1. Prioritize "Gold Standard" sources (PDFs, Images)
        if any(ext in url.lower() for ext in ['.pdf', '.png', '.jpg', '.jpeg']):
            candidates.append(url)
    
    # 2. FALLBACK: If no PDFs found, grab standard HTML pages
    if not candidates:
        print("⚠️ No PDFs found. Switching to standard HTML pages.")
        update_progress_sync(state, "⚠️ No PDFs found, adding HTML sources")
        candidates = [r['url'] for r in list(unique_results)[:2]]
            
    # COMBINE: Uploaded Files + Web Search Results
    final_urls = existing_files + candidates
    
    print(f"✅ Final Source List: {len(final_urls)} items.")
    update_progress_sync(state, f"✅ Aggregated {len(final_urls)} source(s) (Uploads + Web)")
    
    return {"image_urls": final_urls, "messages": [f"Found {len(candidates)} web urls"]}

def ocr_node(state: AgentState):
    print("--- 👁️ RUNNING DEEPSEEK OCR ---")
    update_progress_sync(state, "👁️ Starting OCR processing...")
    
    urls = state.get('image_urls', [])
    if not urls:
        print("⚠️ No URLs found to process.")
        update_progress_sync(state, "⚠️ No documents to process")
        return {"messages": ["Skipped OCR: No URLs found"]}
        
    update_progress_sync(state, f"📥 Processing {len(urls)} document(s)...")
    
    logs = []
    # Process all sources (limit to top 3 to prevent timeouts)
    limit = 3
    for idx, url in enumerate(urls[:limit], 1):
        update_progress_sync(state, f"🔄 Reading document {idx}/{min(len(urls), limit)}...")
        status = process_visual_url(url)
        logs.append(status)
        print(f"   > {status}")
        
    update_progress_sync(state, "✅ OCR processing complete")
    return {"messages": logs}

def analysis_node(state: AgentState):
    print("--- 🧠 ANALYZING DATA ---")
    update_progress_sync(state, "🧠 Analyzing extracted data...")
    
    # Contextual Retrieval
    update_progress_sync(state, "📚 Retrieving relevant context from Vector DB...")
    
    # Note: Ensure app/rag.py has k=20 to retrieve enough context!
    context = query_knowledge_base(f"market analysis features pricing SWOT for {state['task']}")
    
    if not context or len(context) < 50:
        context = "No specific data extracted. Base analysis on general expert knowledge."
        update_progress_sync(state, "⚠️ Limited data found, using general knowledge")
    else:
        update_progress_sync(state, f"✅ Retrieved {len(context)} chars of context")
    
    # --- CRITICAL UPDATE: ROBUST PROMPT ---
    # This prompt forces the AI to prioritize the USER TASK over irrelevant retrieved data.
    prompt = f"""
    You are a Senior Market Research Analyst for a top-tier Venture Capital firm.
    
    **USER TASK**: Produce a highly structured competitive intelligence report on: '{state['task']}'.
    
    **RETRIEVED CONTEXT** (May contain noise):
    {context}
    
    **CRITICAL INSTRUCTION**: 
    1. Your PRIMARY goal is to answer the **USER TASK**.
    2. Analyze the RETRIEVED CONTEXT. If it contains data relevant to the task, use it and cite it.
    3. **IF THE CONTEXT IS IRRELEVANT** (e.g., talks about GPUs when the task is about Side Hustles), **IGNORE IT COMPLETELY**. Do NOT write a report about the irrelevant data. Instead, rely on your internal expert knowledge to generate a high-quality report for the USER TASK.
    
    **FORMATTING INSTRUCTIONS**:
    1. Use a main title with a single hash (#).
    2. Use `##` for section headers with EMOJIS.
    3. Use `###` for sub-headers.
    4. Use bullet points for readability.
    5. **Bold** key numbers and prices.
    
    **REQUIRED STRUCTURE**:
    
    # Report: {state['task']}
    
    ## 📋 Executive Summary
    (2-3 sentences summarizing the key finding. Be direct.)
    
    ## ⚡ Market & Opportunity Analysis
    * **Key Capabilities/Requirements:** (What is needed to succeed?)
    * **Market Demand:** (Why is this relevant now?)
    * **Unique Angles:** (What differentiates a winner in this space?)
    
    ## 💰 Financial Model
    (Extract EXACT prices/rates if found. If not, provide estimated market ranges based on expertise.)
    * **Revenue Potential:** ...
    * **Cost Structure:** ...
    
    ## 🧭 SWOT Analysis
    ### ✅ Strengths
    * ...
    ### ⚠️ Weaknesses
    * ...
    ### 🚀 Opportunities
    * ...
    ### 🛑 Threats
    * ...
    
    ## 🎯 Conclusion & Recommendation
    (One final strategic takeaway.)
    """
    
    try:
        update_progress_sync(state, "🤖 Generating final report...")
        response = llm.invoke([HumanMessage(content=prompt)])
        update_progress_sync(state, "✅ Report generation complete")
        return {"report": response.content}
    except Exception as e:
        update_progress_sync(state, f"❌ Analysis failed: {str(e)}")
        return {"report": f"Analysis failed: {str(e)}"}

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", research_node)
workflow.add_node("ocr_engine", ocr_node)
workflow.add_node("analyst", analysis_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "ocr_engine")
workflow.add_edge("ocr_engine", "analyst")
workflow.add_edge("analyst", END)

app_graph = workflow.compile()
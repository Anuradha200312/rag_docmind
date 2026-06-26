import os
import sys
from typing import TypedDict, List, Dict, Any, Optional

# Ensure directory is in python path for standalone execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import existing helpers
from pdf_helper import extract_pages, count_tokens, chunk_pages
from vector_store import ChromaStore, qdrant_upsert_chunks, qdrant_search, get_sentence_embeddings
from database import db_save_document

# ─────────────────────────────────────────────────────────────────
# 1. Ingestion Graph Definition
# ─────────────────────────────────────────────────────────────────

class IngestionState(TypedDict):
    file_bytes: bytes
    filename: str
    user_id: str
    chat_id: str
    pages: Optional[List[Any]]
    token_count: Optional[int]
    pipeline_used: Optional[str]
    chunks: Optional[List[Any]]
    qdrant_collection: Optional[str]
    success: bool
    error: Optional[str]

def extract_pdf_text_node(state: IngestionState) -> Dict[str, Any]:
    try:
        pages = extract_pages(state["file_bytes"])
        return {"pages": pages, "success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": f"PDF text extraction failed: {str(e)}"}

def route_pipeline_node(state: IngestionState) -> Dict[str, Any]:
    if not state.get("success", True) or state.get("error"):
        return {}
    try:
        full_text = "\n\n".join(p.text for p in state["pages"])
        token_count = count_tokens(full_text)
        q_url = os.getenv("QDRANT_URL", "")
        q_key = os.getenv("QDRANT_API_KEY", "")
        if token_count <= 20000 or not q_url or not q_key:
            pipeline_used = "direct"
        else:
            pipeline_used = "rag"
        return {
            "token_count": token_count,
            "pipeline_used": pipeline_used
        }
    except Exception as e:
        return {"success": False, "error": f"Pipeline routing failed: {str(e)}"}

def index_chroma_node(state: IngestionState) -> Dict[str, Any]:
    if not state.get("success", True) or state.get("error"):
        return {}
    try:
        chroma_store = ChromaStore()
        chunks = chunk_pages(state["pages"], chunk_size=4000, chunk_overlap=800)
        chroma_store.index_chunks(state["chat_id"], chunks)
        
        db_save_document(
            user_id=state["user_id"],
            chat_id=state["chat_id"],
            filename=state["filename"],
            token_count=state["token_count"],
            pipeline_used="direct",
            qdrant_collection=None,
            chunks=chunks
        )
        return {"chunks": chunks, "success": True}
    except Exception as e:
        return {"success": False, "error": f"Chroma indexing failed: {str(e)}"}

def index_qdrant_node(state: IngestionState) -> Dict[str, Any]:
    if not state.get("success", True) or state.get("error"):
        return {}
    try:
        chunks = chunk_pages(state["pages"], chunk_size=4000, chunk_overlap=800)
        chunk_texts = [c["chunk_text"] for c in chunks]
        embeddings = get_sentence_embeddings(chunk_texts)
        qdrant_col = qdrant_upsert_chunks(state["chat_id"], state["user_id"], state["filename"], chunks, embeddings)
        
        db_save_document(
            user_id=state["user_id"],
            chat_id=state["chat_id"],
            filename=state["filename"],
            token_count=state["token_count"],
            pipeline_used="rag",
            qdrant_collection=qdrant_col,
            chunks=chunks
        )
        return {"chunks": chunks, "qdrant_collection": qdrant_col, "success": True}
    except Exception as e:
        return {"success": False, "error": f"Qdrant indexing failed: {str(e)}"}

def decide_ingestion_path(state: IngestionState) -> str:
    if state.get("error") or not state.get("success", True):
        return "end"
    return state["pipeline_used"]

# Assembly Ingestion Graph
ingestion_workflow = StateGraph(IngestionState)
ingestion_workflow.add_node("extract_pdf_text", extract_pdf_text_node)
ingestion_workflow.add_node("route_pipeline", route_pipeline_node)
ingestion_workflow.add_node("index_chroma", index_chroma_node)
ingestion_workflow.add_node("index_qdrant", index_qdrant_node)

ingestion_workflow.set_entry_point("extract_pdf_text")
ingestion_workflow.add_edge("extract_pdf_text", "route_pipeline")
ingestion_workflow.add_conditional_edges(
    "route_pipeline",
    decide_ingestion_path,
    {
        "direct": "index_chroma",
        "rag": "index_qdrant",
        "end": END
    }
)
ingestion_workflow.add_edge("index_chroma", END)
ingestion_workflow.add_edge("index_qdrant", END)

ingestion_graph = ingestion_workflow.compile()

# ─────────────────────────────────────────────────────────────────
# 2. Query Graph Definition
# ─────────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    chat_id: str
    user_id: str
    question: str
    history: List[Dict[str, Any]]
    doc_info: Dict[str, Any]
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    context: Optional[str]
    response_stream: Optional[Any]
    answer: Optional[str]

def retrieve_context_node(state: QueryState) -> Dict[str, Any]:
    doc_info = state["doc_info"]
    chat_id = state["chat_id"]
    question = state["question"]
    
    retrieved_chunks = []
    
    if doc_info["pipeline_used"] == "direct":
        chroma_store = ChromaStore()
        results = chroma_store.search(chat_id, question, top_k=3)
        retrieved_chunks = results
        
        SIMILARITY_THRESHOLD = 2.0
        if results and results[0].get("distance", 0) > SIMILARITY_THRESHOLD:
            retrieved_chunks = []
    else:
        results = qdrant_search(doc_info["qdrant_collection"], question, top_k=5)
        retrieved_chunks = results
        
    if not retrieved_chunks:
        context = ""
    else:
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            page_ref = f"[Page {chunk['page_number']}]" if chunk.get('page_number') else ""
            context_parts.append(f"--- Chunk {idx} {page_ref} ---\n{chunk['chunk_text']}")
        context = "\n\n".join(context_parts)
        
    return {
        "retrieved_chunks": retrieved_chunks,
        "context": context
    }

def generate_llm_response_node(state: QueryState) -> Dict[str, Any]:
    context = state["context"]
    question = state["question"]
    history = state["history"]
    
    if not context:
        def fallback_stream():
            yield "I couldn't find relevant information about that in the uploaded document. Please try rephrasing, or ask about something covered in the document."
        return {"response_stream": fallback_stream()}
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not defined in the environment variables.")
        
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
        api_key=api_key
    )
    
    langchain_history = []
    for msg in history[-8:]:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_history.append(AIMessage(content=msg["content"]))
            
    messages = [
        SystemMessage(
            content=(
                "You are a precise AI assistant. Answer questions ONLY based on the "
                "document context provided. If the answer is not in the context, say "
                "'I don't have that information in this document.' "
                "Do NOT fabricate facts. Be concise and accurate. Cite page numbers where available."
            )
        ),
        *langchain_history,
        HumanMessage(content=f"Document Context:\n{context}\n\nQuestion: {question}")
    ]
    
    response_stream = llm.stream(messages)
    return {"response_stream": response_stream}

# Assembly Query Graph
query_workflow = StateGraph(QueryState)
query_workflow.add_node("retrieve_context", retrieve_context_node)
query_workflow.add_node("generate_response", generate_llm_response_node)

query_workflow.set_entry_point("retrieve_context")
query_workflow.add_edge("retrieve_context", "generate_response")
query_workflow.add_edge("generate_response", END)

query_graph = query_workflow.compile()

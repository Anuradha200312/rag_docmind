import os
import sys
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Ensure the hybrid_pipeline directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Import LangGraph workflows
from workflow import ingestion_graph, query_graph

# Import custom helpers
from database import (
    init_db,
    db_register_user,
    db_verify_user,
    db_create_chat,
    db_get_user_chats,
    db_delete_chat,
    db_load_chat_history,
    db_save_message,
    db_save_document,
    db_get_chat_document,
    db_get_document_chunks,
)
from pdf_helper import extract_pages, count_tokens, chunk_pages
from vector_store import (
    ChromaStore,
    qdrant_upsert_chunks,
    qdrant_search,
    qdrant_delete_collection,
    get_sentence_embeddings,
)

# Load env
load_dotenv()

# Initialize DB on import synchronously
try:
    init_db()
except Exception as e:
    st.warning(f"Database initialization warning: {e}")

# ─────────────────────────────────────────────────────────────────
# Page Config & Theme Styling
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI — Hybrid Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit header/footer */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar Styles ── */
[data-testid="stSidebar"] {
    background: #090b14 !important;
    border-right: 1px solid #1c1e30;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* ── Custom Cards ── */
.auth-container {
    background: #111425;
    border: 1px solid #1c1e30;
    border-radius: 14px;
    padding: 32px;
    max-width: 500px;
    margin: 40px auto;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}
.hero {
    text-align: center;
    padding: 40px 20px 20px;
}
.hero h1 {
    font-size: 38px;
    font-weight: 700;
    color: #f0f4ff;
    margin-bottom: 8px;
}
.hero h1 span {
    background: linear-gradient(135deg, #6C63FF, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 15px;
    color: #8892aa;
    max-width: 500px;
    margin: 0 auto 24px;
}

.doc-card {
    background: linear-gradient(135deg, #16182c 0%, #1d2142 100%);
    border: 1px solid #292d52;
    border-left: 4px solid #6C63FF;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 12px 0;
}
.doc-card .label {
    font-size: 10px;
    font-weight: 600;
    color: #887fff;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.doc-card .value {
    font-size: 13px;
    color: #cbd5e1;
    font-weight: 500;
    word-break: break-all;
}
.doc-stat {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.stat-pill {
    background: #6C63FF1e;
    border: 1px solid #6C63FF3f;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: #a78bfa;
    font-weight: 500;
}

/* ── Chat Header ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    background: #111425;
    border: 1px solid #1c1e30;
    border-radius: 12px;
    margin-bottom: 20px;
}
.chat-header-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #6C63FF, #a78bfa);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.chat-header-text h3 {
    font-size: 15px;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0 0 2px;
}
.chat-header-text p {
    font-size: 12px;
    color: #64748b;
    margin: 0;
}
.status-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    box-shadow: 0 0 8px #10b981cc;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 20px;
    border-bottom: 1px solid #1c1e30;
    margin-bottom: 20px;
}
.brand-icon {
    font-size: 24px;
}
.brand-name {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.3px;
}
.brand-name span {
    color: #6C63FF;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 20px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Helper and State Management
# ─────────────────────────────────────────────────────────────────
def process_and_index_pdf(uploaded_file, chat_id, user_id):
    file_bytes = uploaded_file.read()
    state = ingestion_graph.invoke({
        "file_bytes": file_bytes,
        "filename": uploaded_file.name,
        "user_id": user_id,
        "chat_id": chat_id
    })
    if not state.get("success", True) or state.get("error"):
        raise ValueError(state.get("error", "Failed to index PDF document."))

def init_session_state():
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "chroma_store" not in st.session_state:
        st.session_state.chroma_store = ChromaStore()
    if "loaded_chroma_chats" not in st.session_state:
        st.session_state.loaded_chroma_chats = set()
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

# ─────────────────────────────────────────────────────────────────
# Auth Logic Screen
# ─────────────────────────────────────────────────────────────────
def render_auth():
    st.markdown("""
    <div class="hero" style="padding-top: 40px; padding-bottom: 5px;">
        <h1>🧠 <span>DocMind AI</span></h1>
        <p>Your intelligent, secure multi-user document pipeline assistant.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_space1, col_center, col_space2 = st.columns([0.33, 0.34, 0.33])
    
    with col_center:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        if st.session_state.auth_mode == "login":
            st.subheader("Login to your Account")
            email = st.text_input("Email Address", placeholder="name@domain.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pwd")
            
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("Log In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    try:
                        user = db_verify_user(email.strip().lower(), password)
                        if user:
                            st.session_state.authenticated_user = user
                            st.success(f"Welcome back, {user['name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")
                    except Exception as e:
                        st.error(f"❌ Database connection error: {e}")
                        st.info("💡 Ensure PostgreSQL is running and your DATABASE_URL in .env is correct.")
            
            st.markdown("<hr style='border-color: #1c1e30; margin-top: 25px; margin-bottom: 20px;' />", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 13.5px; color: #8892aa; margin-bottom: 10px;'>Don't have an account?</p>", unsafe_allow_html=True)
            if st.button("📝 Create an Account", use_container_width=True):
                st.session_state.auth_mode = "signup"
                st.rerun()
                
        else:
            st.subheader("Create a New Account")
            name = st.text_input("Full Name", placeholder="Jane Doe", key="signup_name")
            email = st.text_input("Email Address", placeholder="name@domain.com", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="signup_pwd")
            confirm_pwd = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="signup_confirm")
            
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("Register", use_container_width=True, type="primary"):
                if not name or not email or not password or not confirm_pwd:
                    st.error("All fields are required.")
                elif password != confirm_pwd:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        user = db_register_user(name.strip(), email.strip().lower(), password)
                        if user:
                            st.session_state.authenticated_user = user
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.error("An account with this email already exists.")
                    except Exception as e:
                        st.error(f"❌ Database connection error: {e}")
                        st.info("💡 Ensure PostgreSQL is running and your DATABASE_URL in .env is correct.")
                        
            st.markdown("<hr style='border-color: #1c1e30; margin-top: 25px; margin-bottom: 20px;' />", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 13.5px; color: #8892aa; margin-bottom: 10px;'>Already have an account?</p>", unsafe_allow_html=True)
            if st.button("🔑 Log In Instead", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Main Interface Screen
# ─────────────────────────────────────────────────────────────────
def render_main():
    user = st.session_state.authenticated_user
    init_session_state()

    # Load chats for user synchronously
    chats = db_get_user_chats(user["user_id"])
    
    # ── Sidebar Panel ──
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <span class="brand-icon">🧠</span>
            <span class="brand-name">Doc<span>Mind</span> AI</span>
        </div>
        """, unsafe_allow_html=True)
        
        # User details
        st.markdown(f"👤 **{user['name']}**")
        st.caption(f"{user['email']}")
        
        if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
            st.session_state.authenticated_user = None
            st.session_state.active_chat_id = None
            st.rerun()
            
        st.markdown('<div class="section-label">💬 Chat Sessions</div>', unsafe_allow_html=True)
        
        # New Chat Button in Sidebar
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_chat = db_create_chat(user["user_id"], f"Chat {len(chats) + 1}")
            st.session_state.active_chat_id = new_chat["chat_id"]
            st.rerun()

        # Render chat list
        for c in chats:
            col_chat, col_del = st.columns([0.8, 0.2])
            with col_chat:
                is_active = (st.session_state.active_chat_id == c["chat_id"])
                btn_type = "primary" if is_active else "secondary"
                title_disp = c["title"]
                if len(title_disp) > 22:
                    title_disp = title_disp[:20] + "…"
                if st.button(f"💬 {title_disp}", key=f"chat_{c['chat_id']}", use_container_width=True, type=btn_type):
                    st.session_state.active_chat_id = c["chat_id"]
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{c['chat_id']}", use_container_width=True, help="Delete Chat"):
                    db_delete_chat(c["chat_id"])
                    # Clean up vector collections
                    qdrant_delete_collection(c["chat_id"])
                    try:
                        st.session_state.chroma_store.client.delete_collection(
                            st.session_state.chroma_store.get_collection_name(c["chat_id"])
                        )
                    except Exception:
                        pass
                    
                    if st.session_state.active_chat_id == c["chat_id"]:
                        st.session_state.active_chat_id = None
                    st.rerun()

    # ── Main Chat Area ──

    active_chat_id = st.session_state.active_chat_id
    
    # Render Welcome/Empty Screen if no active chat
    if not active_chat_id:
        st.markdown("""
        <div style="text-align: center; margin-top: 40px; padding: 20px;">
            <div style="font-size: 50px;">🧠</div>
            <h2 style="color: #f1f5f9; margin-top:15px;">Welcome to DocMind AI Assistant</h2>
            <p style="color: #64748b; max-width: 500px; margin: 10px auto 10px;">
                Directly upload a PDF document below to start a new chat session.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Directly show PDF Upload Card
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: #111425; border: 2px dashed #292d52; border-radius: 12px; margin-bottom: 20px;">
            <div style="font-size: 40px; margin-bottom:10px;">📤</div>
            <h3 style="color: #f1f5f9; margin-top:0;">Upload PDF to Start Chatting</h3>
            <p style="color: #64748b; font-size:13px; max-width: 480px; margin: 0 auto 15px;">
                Our pipeline will analyze the size, automatically create a chat session, and index the document.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a document to start",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_uploader_welcome"
        )
        
        if uploaded_file is not None:
            with st.spinner("⚙️ Preparing new chat session and analyzing PDF..."):
                try:
                    # Automatically create a new chat session named after the PDF
                    chat_title = uploaded_file.name.replace(".pdf", "")
                    if len(chat_title) > 30:
                        chat_title = chat_title[:27] + "..."
                    new_chat = db_create_chat(user["user_id"], chat_title)
                    target_chat_id = new_chat["chat_id"]
                    
                    # Process and index document
                    process_and_index_pdf(uploaded_file, target_chat_id, user["user_id"])
                    
                    # Set active chat and reload
                    st.session_state.active_chat_id = target_chat_id
                    st.success("✅ Document processed and chat session created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error uploading document: {e}")

        # Features Grid
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="medium")
        
        cards = [
            (
                "⚡ Fast ChromaDB Pipeline", 
                "For documents under 20k tokens. Chunks are cached and indexed locally for instant search responses.",
                [
                    "💡 Summarize the key findings.",
                    "💡 List the main conclusions.",
                    "💡 What are the key takeaways from this document?"
                ]
            ),
            (
                "☁️ Enterprise RAG Pipeline", 
                "For large documents over 20k tokens. Uses sentence embeddings and indices in Qdrant Cloud for deep search.",
                [
                    "💡 Search for specific technical details.",
                    "💡 Analyze comparisons and data tables.",
                    "💡 Compare findings across different chapters."
                ]
            ),
            (
                "🔐 Multi-user Security", 
                "Secure user isolation. Authentication and history records are kept strictly private inside PostgreSQL.",
                [
                    "💡 Retrieve my personal analysis history.",
                    "💡 Ensure only I can access my stored files.",
                    "💡 How does user isolation protect my private data?"
                ]
            )
        ]
        
        for col, (title, desc, questions) in zip([col1, col2, col3], cards):
            q_html = "".join(f"<div style='margin-top:8px; font-style:italic; font-size:12px; color:#a78bfa;'>{q}</div>" for q in questions)
            with col:
                st.markdown(f"""
                <div style="background: #111425; border: 1px solid #1c1e30; border-radius: 12px; padding: 20px; height: 100%;">
                    <h4 style="color: #f8fafc; margin-top:0; margin-bottom:8px;">{title}</h4>
                    <p style="color: #8892aa; font-size: 12.5px; line-height: 1.4; margin-bottom:12px;">{desc}</p>
                    <div style="border-top: 1px solid #1c1e30; padding-top:10px;">
                        <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:#64748b; letter-spacing:0.5px;">Common Queries:</span>
                        {q_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        return

    # Fetch active chat title
    current_chat = None
    for c in chats:
        if c["chat_id"] == active_chat_id:
            current_chat = c
            break
            
    if not current_chat:
        st.session_state.active_chat_id = None
        st.rerun()

    # Load active document details
    doc_info = db_get_chat_document(active_chat_id)
    
    # Chat Header
    status_text = "No PDF uploaded yet"
    if doc_info:
        status_text = f"📄 {doc_info['filename']} · {doc_info['pipeline_used'].upper()} Pipeline Ready"
        
    st.markdown(f"""
    <div class="chat-header" style="margin-top: 15px;">
        <div class="chat-header-icon">💬</div>
        <div class="chat-header-text">
            <h3>{current_chat['title']}</h3>
            <p><span class="status-dot"></span>{status_text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Re-hydrate ChromaDB if needed (ephemeral loss prevention)
    if doc_info and doc_info["pipeline_used"] == "direct":
        col_name = st.session_state.chroma_store.get_collection_name(active_chat_id)
        is_empty = False
        try:
            col = st.session_state.chroma_store.client.get_collection(col_name)
            if col.count() == 0:
                is_empty = True
        except Exception:
            is_empty = True

        if is_empty:
            with st.spinner("🔄 Re-hydrating local document vector store..."):
                chunks = db_get_document_chunks(doc_info["document_id"])
                st.session_state.chroma_store.index_chunks(active_chat_id, chunks)

    # ── Display Active Document Info on Main screen (if exists) ──
    if doc_info:
        short_fname = doc_info["filename"]
        if len(short_fname) > 40:
            short_fname = short_fname[:37] + "…"
        
        pipeline_badge = "⚡ ChromaDB (Local)" if doc_info["pipeline_used"] == "direct" else "☁️ Qdrant Cloud"
        
        st.markdown(f"""
        <div class="doc-card" style="margin-bottom: 20px;">
            <div class="label">📄 Active PDF Document</div>
            <div class="value">{short_fname}</div>
            <div class="doc-stat">
                <span class="stat-pill">📊 {doc_info['token_count']:,} tokens</span>
                <span class="stat-pill">{pipeline_badge}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render History
    history = db_load_chat_history(active_chat_id)
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("timestamp"):
                align = "right" if msg["role"] == "user" else "left"
                st.markdown(
                    f"<div style='font-size: 10px; color: #475569; text-align: {align}; margin-top: 4px;'>{msg['timestamp']}</div>",
                    unsafe_allow_html=True
                )

    # ── Render PDF Uploader directly on main page (if no doc loaded) ──
    if not doc_info:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: #111425; border: 2px dashed #292d52; border-radius: 12px; margin-top: 20px; margin-bottom: 20px;">
            <div style="font-size: 40px; margin-bottom:10px;">📤</div>
            <h3 style="color: #f1f5f9; margin-top:0;">Upload PDF Document</h3>
            <p style="color: #64748b; font-size:14px; max-width: 480px; margin: 0 auto 20px;">
                Before starting the conversation, please upload a PDF. 
                Our pipeline will analyze the document's size and route it dynamically.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a document to chat",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_uploader_main"
        )
        
        if uploaded_file is not None:
            with st.spinner("⚙️ Reading and analyzing PDF document..."):
                try:
                    process_and_index_pdf(uploaded_file, active_chat_id, user["user_id"])
                    st.success("✅ Document indexed and saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error uploading document: {e}")
        return

    # User Input Panel (Suggestion chips)
    suggestion_clicked_text = None
    if not history:
        st.markdown("<div style='margin-top: 16px; font-size: 13px; color: #64748b; font-weight: 500;'>Quick Suggestions:</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        suggestions = [
            "📝 Summarize key sections",
            "🔑 Extract main findings",
            "📊 Detail important conclusions"
        ]
        for col, sug in zip([col1, col2, col3], suggestions):
            with col:
                if st.button(sug, use_container_width=True):
                    suggestion_clicked_text = sug.split(" ", 1)[1]

    # Chat Input
    question = st.chat_input("Ask a question about the uploaded document...")
    if suggestion_clicked_text:
        question = suggestion_clicked_text

    if question:
        t_user = datetime.now().strftime("%I:%M %p")
        with st.chat_message("user"):
            st.markdown(question)
            st.markdown(
                f"<div style='font-size: 10px; color: #475569; text-align: right; margin-top: 4px;'>{t_user}</div>",
                unsafe_allow_html=True
            )
            
        db_save_message(active_chat_id, "user", question)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    state = query_graph.invoke({
                        "chat_id": active_chat_id,
                        "user_id": user["user_id"],
                        "question": question,
                        "history": history,
                        "doc_info": doc_info
                    })
                    response_stream = state["response_stream"]
                    sources = state.get("retrieved_chunks") or []
                except Exception as e:
                    st.error(f"❌ Error invoking query graph: {e}")
                    st.stop()

            # Create clean generator that handles LangChain chunk properties safely
            def response_generator():
                for chunk in response_stream:
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, str):
                        yield chunk
                    else:
                        yield str(chunk)

            answer = st.write_stream(response_generator())
            
            t_assist = datetime.now().strftime("%I:%M %p")
            st.markdown(
                f"<div style='font-size: 10px; color: #475569; text-align: left; margin-top: 4px;'>{t_assist}</div>",
                unsafe_allow_html=True
            )
            
            if sources:
                with st.expander("📖 View references from document", expanded=False):
                    for idx, s in enumerate(sources, 1):
                        pg_lbl = f"Page {s['page_number']}" if s.get('page_number') else f"Chunk {idx}"
                        st.markdown(f"**Reference: {pg_lbl}**")
                        st.markdown(
                            f"<div style='background:#18192a; border-left:3px solid #6C63FF; "
                            f"padding:10px 14px; border-radius:6px; font-size:13px; "
                            f"color:#cbd5e1; margin-bottom:10px'>{s['chunk_text']}</div>",
                            unsafe_allow_html=True,
                        )

        db_save_message(active_chat_id, "assistant", answer)
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# App Entrance
# ─────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    if st.session_state.authenticated_user is None:
        render_auth()
    else:
        render_main()

if __name__ == "__main__":
    main()

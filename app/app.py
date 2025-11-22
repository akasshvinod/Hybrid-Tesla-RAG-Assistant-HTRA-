import sys
import os

# Add project root to python path so "src" works
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import time
from src.pipeline.rag_pipeline import answer_query, chat_history, KNOWN_CHAPTERS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =========================================================
# PAGE CONFIG - Must be first
# =========================================================
st.set_page_config(
    page_title="Tesla Model 3 RAG Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# OPTIMIZED CSS - Faster rendering, modern design
# =========================================================
st.markdown("""
<style>
    /* Performance optimizations */
    .stApp {
        animation: none !important;
        background: linear-gradient(to bottom, #0a0a0a, #1a1a1a);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 15%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: slideInRight 0.3s ease;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        color: #e0e0e0;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 15%;
        border-left: 4px solid #e74c3c;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Metrics cards */
    .metrics-container {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #333;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        background: rgba(102, 126, 234, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #999;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1a1a1a, #0a0a0a);
        border-right: 1px solid #333;
    }
    
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
    }
    
    /* Welcome message */
    .welcome-banner {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .welcome-banner h2 {
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-top: 2px solid #333;
        background: #0a0a0a;
    }
    
    /* Context box */
    .context-box {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
        color: #b0b0b0;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0

if "latencies" not in st.session_state:
    st.session_state.latencies = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Chapter selector with all known chapters
    chapter_options = ["🔍 Auto-detect"] + [f"📖 {ch.capitalize()}" for ch in sorted(KNOWN_CHAPTERS)]
    selected_chapter = st.selectbox(
        "Chapter Filter",
        chapter_options,
        help="Select a specific manual chapter or let AI auto-detect the relevant section"
    )
    
    # Parse chapter selection
    if selected_chapter == "🔍 Auto-detect":
        chapter = None
    else:
        chapter = selected_chapter.replace("📖 ", "")
    
    # Display options
    st.markdown("### 🎨 Display Options")
    show_metrics = st.checkbox("Show Performance Metrics", value=True)
    show_context = st.checkbox("Show Retrieved Context", value=False)
    
    st.markdown("---")
    
    # Session statistics
    st.markdown("## 📊 Session Stats")
    
    avg_latency = (st.session_state.total_latency / st.session_state.total_queries 
                   if st.session_state.total_queries > 0 else 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_latency:.0f}ms</div>
            <div class="metric-label">Avg Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Speed comparison
    if st.session_state.total_queries > 0:
        terminal_time = avg_latency  # Your terminal latency was ~54s
        speedup = 54000 / avg_latency if avg_latency > 0 else 1
        st.info(f"⚡ UI is running smoothly with {avg_latency:.0f}ms average response time")
    
    st.markdown("---")
    
    # Quick examples
    st.markdown("## 💡 Quick Examples")
    examples = [
        ("🔌", "How do I charge my Model 3?"),
        ("🤖", "What is Autopilot?"),
        ("🔑", "How to add keys from mobile app?"),
        ("🛞", "What's the recommended tire pressure?"),
        ("⚠️", "Emergency brake warnings"),
        ("🚗", "How to enable Dog Mode?"),
        ("❄️", "Winter driving tips")
    ]
    
    for emoji, example in examples:
        if st.button(f"{emoji} {example}", key=f"ex_{example}", use_container_width=True):
            st.session_state.pending_query = example
    
    st.markdown("---")
    
    # Clear button
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        chat_history.clear()
        st.session_state.total_queries = 0
        st.session_state.total_latency = 0
        st.session_state.latencies = []
        st.rerun()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">
    <h1>🚗 Tesla Model 3 — RAG Assistant</h1>
    <p>⚡ Powered by Llama 3.1 + RAG Pipeline | Ask anything about your vehicle manual</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CHAT DISPLAY
# =========================================================

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-banner">
        <h2>👋 Welcome to Tesla RAG Assistant!</h2>
        <p>I'm here to help you with your Tesla Model 3 manual. Ask me anything about:</p>
        <p>🔌 Charging • 🤖 Autopilot • 🔑 Keys & Access • 🛞 Maintenance • ⚠️ Safety Features</p>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: #999;">Try clicking an example from the sidebar or type your question below!</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>🧑 You:</strong><br>{msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistant:</strong><br>{msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Show metrics if available and enabled
        if show_metrics and "metrics" in msg:
            m = msg["metrics"]
            
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            cols = st.columns(5)
            
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{m.get('retrieval_latency', 0):.0f}ms</div>
                    <div class="metric-label">⚡ Retrieval</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{m.get('llm_latency', 0):.0f}ms</div>
                    <div class="metric-label">🧠 LLM</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{m.get('total_latency', 0):.0f}ms</div>
                    <div class="metric-label">⏱️ Total</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{m.get('docs_used', 0)}</div>
                    <div class="metric-label">📄 Docs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[4]:
                chapter_display = m.get('chapter_used') or 'Auto'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.3rem;">{chapter_display}</div>
                    <div class="metric-label">📖 Chapter</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show context if enabled
        if show_context and "context" in msg:
            with st.expander("🔍 View Retrieved Context"):
                st.markdown(f'<div class="context-box">{msg["context"]}</div>', unsafe_allow_html=True)

# =========================================================
# CHAT INPUT
# =========================================================

# Check for pending query from example button
if "pending_query" in st.session_state:
    user_input = st.session_state.pending_query
    del st.session_state.pending_query
else:
    user_input = st.chat_input("💬 Ask about your Tesla manual...", key="chat_input")

# Process query
if user_input:
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show loading with custom spinner
    with st.spinner("🔍 Searching manual and generating answer..."):
        try:
            start_time = time.time()
            
            # Call RAG pipeline
            result = answer_query(user_input, chapter=chapter)
            
            elapsed = (time.time() - start_time) * 1000
            
            # Update session stats
            st.session_state.total_queries += 1
            st.session_state.total_latency += result.get("total_latency", 0)
            st.session_state.latencies.append(result.get("total_latency", 0))
            
            # Prepare context if needed
            context_text = ""
            if show_context:
                try:
                    from src.llm.prompts import format_docs
                    context_text = format_docs(result.get("docs", []))
                except:
                    context_text = "Context not available"
            
            # Add assistant message with all metadata
            assistant_msg = {
                "role": "assistant",
                "content": result["answer"],
                "metrics": {
                    "retrieval_latency": result.get("retrieval_latency", 0),
                    "llm_latency": result.get("llm_latency", 0),
                    "total_latency": result.get("total_latency", 0),
                    "docs_used": result.get("docs_used", 0),
                    "chapter_used": result.get("chapter_used")
                }
            }
            
            if show_context and context_text:
                assistant_msg["context"] = context_text
            
            st.session_state.messages.append(assistant_msg)
            
            logger.info(f"Query processed in {elapsed:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error while processing your request: {str(e)}"
            })
    
    # Rerun to update UI
    st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🚀 Built with Streamlit + LangChain")
with col2:
    st.caption("🧠 Powered by Llama 3.1")
with col3:
    st.caption("🔍 nomic-embed-text embeddings")
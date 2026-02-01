"""
Streamlit Chat UI for the RAG Agent.
Uses lazy imports to ensure instant UI load, with heavy modules loaded in background.
"""
import asyncio
import logging
import sys
from pathlib import Path
import uuid
import threading
import time
import concurrent.futures

import streamlit as st

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging FIRST (lightweight)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Lazy-loaded module placeholders (will be populated by background thread)
_graph_module = None
_db_module = None
_ingest_func = None
_embedding_model_func = None


def get_graph_module():
    """Get graph module (lazy load if needed)."""
    global _graph_module
    if _graph_module is None:
        from agent_app import graph
        _graph_module = graph
    return _graph_module


def get_db():
    """Get database module (lazy load if needed)."""
    global _db_module
    if _db_module is None:
        from mcp_server.db import db
        _db_module = db
    return _db_module


def get_ingest_func():
    """Get ingest function (lazy load if needed)."""
    global _ingest_func
    if _ingest_func is None:
        from mcp_server.ingestion import ingest_file
        _ingest_func = ingest_file
    return _ingest_func


def get_embedding_model():
    """Get embedding model function (lazy load if needed)."""
    global _embedding_model_func
    if _embedding_model_func is None:
        from mcp_server.ingestion import _get_embedding_model
        _embedding_model_func = _get_embedding_model
    return _embedding_model_func


# Background resource loading (runs in daemon thread, does NOT block UI)
def warm_up_resources():
    """Background thread to load heavy resources in parallel."""
    logger.info("Starting background resource warmup (parallel)...")
    start_time = time.time()
    
    def load_db():
        """Connect to DB."""
        try:
            db = get_db()
            if db.client is None:
                logger.info("Connecting to Milvus...")
                db.connect()
                logger.info("Milvus connected.")
        except Exception as e:
            logger.error(f"DB load failed: {e}")

    def load_graph():
        """Load graph module."""
        try:
            get_graph_module()
            logger.info("Graph module loaded.")
        except Exception as e:
            logger.error(f"Graph load failed: {e}")

    def load_ingest():
        """Load ingestion module."""
        try:
            get_ingest_func()
            logger.info("Ingestion module loaded.")
        except Exception as e:
            logger.error(f"Ingestion load failed: {e}")

    def load_model():
        """Load embedding model (heaviest)."""
        try:
            logger.info("Loading embedding model...")
            func = get_embedding_model()
            func()  # Actually load the model
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

    try:
        # Run independent tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(load_db),
                executor.submit(load_graph),
                executor.submit(load_ingest),
                executor.submit(load_model)
            ]
            concurrent.futures.wait(futures)
            
        logger.info(f"Background warmup complete in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Background warmup failed: {e}")


# Start background loading BEFORE any Streamlit UI code
if "resources_warming" not in st.session_state:
    st.session_state.resources_warming = True
    t = threading.Thread(target=warm_up_resources, daemon=True)
    t.start()
    logger.info("Background loading thread started")

# Page config (instant)
st.set_page_config(
    page_title="RAG Agent",
    layout="wide",
)

# Title (instant)
st.title("Agentic RAG System")
st.caption("Upload documents and ask questions")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def run_async(coro):
    """Helper to run async code in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(lambda: asyncio.run(coro)).result()
            return result
    except RuntimeError:
        pass
    return asyncio.run(coro)


# Sidebar - Document Upload
with st.sidebar:
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md"],
    )
    
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                temp_path = Path(f"./temp_{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.getvalue())
                
                try:
                    db = get_db()
                    ingest_file = get_ingest_func()
                    
                    if db.client is None:
                        db.connect()
                    
                    count = run_async(ingest_file(str(temp_path), db))
                    
                    if count > 0:
                        st.success(f"Ingested {count} chunks from {uploaded_file.name}")
                        st.session_state.last_uploaded_file = uploaded_file.name
                    else:
                        st.warning("No content extracted from file")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
        else:
             st.info(f"File '{uploaded_file.name}' already ingested.")
    
    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Hide Session ID from user
    # st.caption("Session: " + st.session_state.session_id[:8])


# Hide Streamlit toolbar and footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph_module = get_graph_module()
                db = get_db()
                
                if db.client is None:
                    db.connect()
                
                response = run_async(graph_module.run_agent(prompt, st.session_state.session_id))
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                logger.error(f"Agent error: {e}")

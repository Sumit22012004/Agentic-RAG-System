"""
Streamlit Chat UI for the RAG Agent.
"""
import asyncio
import logging
import sys
from pathlib import Path
import uuid
import threading
import time

import streamlit as st

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports placeholders
_graph_module = None
_db = None
_ingest_func = None

def get_graph_module():
    """Lazy load the graph module."""
    global _graph_module
    if _graph_module is None:
        from agent_app import graph
        _graph_module = graph
    return _graph_module

def get_db():
    """Lazy load and connect to database."""
    global _db
    if _db is None:
        from mcp_server.db import db
        _db = db
        # We don't force connect here to avoid blocking UI if called on main thread
        # The background thread checks this too
    return _db

def get_ingest_func():
    """Lazy load the ingest function."""
    global _ingest_func
    if _ingest_func is None:
        from mcp_server.ingestion import ingest_file
        _ingest_func = ingest_file
    return _ingest_func

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Start background resource loading
def warm_up_resources():
    """Background thread to load heavy resources in parallel."""
    import concurrent.futures
    logger.info("Starting background resource warmup (parallel)...")
    start_time = time.time()
    
    def load_db():
        """Connect to DB."""
        try:
            db_instance = get_db()
            if db_instance.client is None:
                logger.info("Connecting to Milvus...")
                db_instance.connect()
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
            # This triggers the heavy import and download/load
            from mcp_server.ingestion import _get_embedding_model
            logger.info("Loading embedding model...")
            _get_embedding_model()
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

    try:
        # Run independent tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            futures = [
                executor.submit(load_db),
                executor.submit(load_graph),
                executor.submit(load_ingest),
                executor.submit(load_model)
            ]
            
            # Wait for completion (optional, since this is a daemon thread)
            concurrent.futures.wait(futures)
            
        logger.info(f"Background warmup complete in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Background warmup failed: {e}")

if "resources_warming" not in st.session_state:
    st.session_state.resources_warming = True
    t = threading.Thread(target=warm_up_resources, daemon=True)
    t.start()
    logger.info("Background loading thread started")

# Page config
st.set_page_config(
    page_title="RAG Agent",
    layout="wide",
)

# Title
st.title("Agentic RAG System")
st.caption("Upload documents and ask questions")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def run_async(coro):
    """Helper to run async code in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Sidebar - Document Upload
with st.sidebar:
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md"],
    )
    
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Processing..."):
                # Save uploaded file temporarily
                temp_path = Path(f"./temp_{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.getvalue())
                
                try:
                    # Get resources
                    db = get_db()
                    ingest_file = get_ingest_func()
                    
                    # Connect to DB if needed
                    if db.client is None:
                        db.connect()
                    
                    # Run ingestion
                    count = run_async(ingest_file(str(temp_path), db))
                    
                    if count > 0:
                        st.success(f"Ingested {count} chunks from {uploaded_file.name}")
                    else:
                        st.warning("No content extracted from file")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    # Cleanup temp file
                    if temp_path.exists():
                        temp_path.unlink()
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Session: " + st.session_state.session_id[:8])


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get resources
                graph_module = get_graph_module()
                db = get_db()
                
                # Connect to DB if needed
                if db.client is None:
                    db.connect()
                
                # Run agent
                response = run_async(graph_module.run_agent(prompt, st.session_state.session_id))
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                logger.error(f"Agent error: {e}")

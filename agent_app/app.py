"""
Streamlit Chat UI for the RAG Agent.
"""
import asyncio
import logging
import sys
from pathlib import Path
import uuid

import streamlit as st

# Added parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_app.graph import run_agent
from agent_app.memory import ConversationMemory, get_redis, close_redis
from mcp_server.db import db
from mcp_server.ingestion import ingest_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
                # Connect to DB if needed
                if db.client is None:
                    db.connect()
                
                # Run agent
                response = run_async(run_agent(prompt, st.session_state.session_id))
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                logger.error(f"Agent error: {e}")

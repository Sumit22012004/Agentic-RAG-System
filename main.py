"""
Main entry point for the application.
Starts both the MCP server and Streamlit UI.
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the application."""
    project_root = Path(__file__).parent
    
    logger.info("Starting Agentic RAG System...")
    logger.info("Make sure Docker services are running: docker-compose up -d")
    
    # Start Streamlit
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(project_root / "agent_app" / "app.py"),
        "--server.port", "8501",
    ]
    
    logger.info("Starting Streamlit UI on http://localhost:8501")
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()

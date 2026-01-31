"""
Config for the Agent App.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama LLM
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# Redis for checkpointing
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# MCP Server (knowledge base)
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))


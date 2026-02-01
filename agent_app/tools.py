"""
Tools for the agent to interact with external services.
"""
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import OLLAMA_BASE_URL, LLM_MODEL
from mcp_server.db import db
from mcp_server.ingestion import generate_embeddings

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def call_llm(prompt: str, system_prompt: str = None) -> str:
    """Call Ollama LLM."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


async def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Query the MCP knowledge base.
    For now, directly imports from mcp_server (same process).
    In production, this would be an HTTP call to the MCP server.
    """
    try:
        # Generate embedding for query
        embeddings = await generate_embeddings([query])
        if not embeddings:
            return "Error: Could not generate query embedding"
        
        # Search
        results = await db.search(embeddings[0], top_k=top_k)
        
        if not results:
            return "No relevant documents found."
        
        # Format
        output = []
        for i, doc in enumerate(results, 1):
            output.append(f"[{i}] {doc['text']}")
        
        return "\n\n".join(output)
        
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return f"Error searching knowledge base: {e}"

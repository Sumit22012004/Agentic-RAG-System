"""
MCP Server - exposes knowledge base tools.
"""
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .db import db
from .ingestion import ingest_file, generate_embeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# MCP server instance
mcp = FastMCP(
    "Knowledge Base",
    description="Document ingestion and semantic search.",
)


@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """
    Add a document to the knowledge base.
    Supports: PDF, DOCX, PPTX, XLSX, TXT, MD
    """
    logger.info(f"[ingest_document] {file_path}")
    
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found - {file_path}"
        
        count = await ingest_file(file_path, db)
        
        if count == 0:
            return f"Warning: No content extracted from {path.name}"
        
        return f"Ingested {count} chunks from '{path.name}'"
        
    except ValueError as e:
        logger.error(f"[ingest_document] {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"[ingest_document] Failed: {e}")
        return f"Error: Ingestion failed - {e}"


@mcp.tool()
async def query_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base.
    Uses semantic vector search.
    """
    logger.info(f"[query_knowledge_base] '{query[:50]}...'")
    
    if not query.strip():
        return "Error: Query cannot be empty"
    
    try:
        # Get query embedding
        embeddings = await generate_embeddings([query])
        if not embeddings:
            return "Error: Failed to generate query embedding"
        
        # Search
        results = await db.search(embeddings[0], top_k=top_k)
        
        if not results:
            return "No relevant documents found."
        
        # Format results
        output = []
        for i, doc in enumerate(results, 1):
            source = Path(doc["source"]).name if doc["source"] != "unknown" else "unknown"
            output.append(f"[{i}] Source: {source} | Score: {doc['score']:.3f}\n{doc['text']}")
        
        return "\n\n---\n\n".join(output)
        
    except Exception as e:
        logger.error(f"[query_knowledge_base] Failed: {e}")
        return f"Error: Search failed - {e}"


@mcp.tool()
async def list_supported_formats() -> str:
    """List supported file formats."""
    return "Supported: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, TXT, MD"


def main():
    """Start the MCP server."""
    logger.info("Starting MCP Server...")
    
    try:
        db.connect()
    except Exception as e:
        logger.error(f"Cannot connect to Milvus: {e}")
        logger.error("Run: docker-compose up -d")
        sys.exit(1)
    
    mcp.run()


if __name__ == "__main__":
    main()

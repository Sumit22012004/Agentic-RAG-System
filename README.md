# Agentic RAG System

This is a production-grade, local-first **Agentic RAG System** designed for high-performance retrieval and intelligent query answering. I built this using **LangGraph**, **Milvus**, and **Ollama**, implementing the **Model Context Protocol (MCP)** to ensure modularity.

## System Overview

I designed this system to solving the "dumb retrieval" problem common in basic RAG applications. Instead of a linear fetch-and-generate pipeline, I implemented a cyclic agent that reasons about its data:

1.  **Self-Correction**: The agent evaluates retrieved documents. If they are irrelevant, it automatically rewrites the query and tries again.
2.  **Hybrid Search**: By using Milvus, the system combines semantic search (Vectors) with exact keyword matching (BM25), ensuring it doesn't miss specific identifiers (e.g., "Error 503").
3.  **Decoupled Architecture**: I separated the "Brain" (Agent) from the "Knowledge" (Database) using an MCP Server. This makes the retrieval engine reusable by other agents or tools in the future.

## Tech Stack

*   **LLM**: Ollama (Llama 3 / Mistral) - Chosen for local data privacy and zero cost.
*   **Vector DB**: Milvus - Selected for its production-readiness and Hybrid Search capabilities.
*   **Orchestration**: LangGraph - Used for managing complex, stateful conversation loops.
*   **Storage**: Redis - Implements persistent conversation memory (checkpointing).
*   **Protocol**: Model Context Protocol (MCP) - Standardizes the connection between the agent and data source.

## Prerequisites

1.  **Docker Desktop** (Must be running)
2.  **Ollama** installed and running locally (`ollama serve`).
3.  **Python 3.10+**

## Quick Start

### 1. Clone & Setup
```bash
git clone <repository-url>
cd Agentic-RAG-System
```

### 2. Infrastructure (Docker)
Start the Milvus and Redis services:
```bash
docker-compose up -d milvus redis
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull LLM Model
Download the required models to your local machine:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 5. Run the System
Start the application (Starts both the MCP Server and Streamlit UI):
```bash
python main.py
```
Access the UI at: `http://localhost:8501`

## Architecture

I split the backend into two distinct processes to mimic a microservices architecture:

1.  **MCP Server (`/mcp_server`)**: This is the "Knowledge Layer". It handles all document ingestion and Milvus interactions. It exposes a standardized `query_knowledge_base` tool.
2.  **Agent App (`/agent_app`)**: This is the "Reasoning Layer". It manages the LangGraph state machine, stores history in Redis, and queries the MCP server when it needs information.

## Project Structure

```
├── agent_app/          # Streamlit & LangGraph Logic
├── mcp_server/         # Knowledge Base & Milvus Logic
├── docker-compose.yml  # Infrastructure definition
├── requirements.txt    # Python dependencies
├── SYSTEM_DESIGN.md    # Detailed Architecture Doc
└── README.md           # This file
```

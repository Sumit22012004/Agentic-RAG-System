# System Design Document: Agentic RAG System

## 1. System Architecture

I designed the system using a **Microservices-inspired Client-Server Architecture**. I explicitly chose to decouple the data layer (Ingestion/Retrieval) from the reasoning layer (Agent) using the **Model Context Protocol (MCP)**. This separation ensures that the Knowledge Base can be scaled or accessed by other applications independently of the main chat agent.

### High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            Reasoning Layer (Docker)                               |
|                                                                                   |
|      +-----------+           +-------------+           +------------------+       |
|      | Streamlit | <-------> |  LangGraph  | <-------> |   Redis Cache    |       |
|      |    UI     |           |    Agent    |           | (Session Memory) |       |
|      +-----------+           +-------------+           +------------------+       |
|                                     |  |                                          |
|                 +-------------------+  +--------------------+                     |
|                 v                                           v                     |
|      +-------------------+                         +-----------------+            |
|      |    Ollama LLM     |                         |    Logging      |            |
|      |  (Inference API)  |                         | (Observability) |            |
|      +-------------------+                         +-----------------+            |
+-----------------------------------------------------------------------------------+
                                      ^
                                      | (MCP Protocol over HTTP)
                                      v
+-----------------------------------------------------------------------------------+
|                            Knowledge Layer (MCP Server)                           |
|                                                                                   |
|      +--------------+          +--------------+          +--------------+         |
|      |  MCP Server  | <------> | Milvus (DB)  | <------- |  Documents   |         |
|      |   Process    |          | Vector Search|          |(PDF,DOCX,PPT)|         |
|      +--------------+          +--------------+          +--------------+         |
+-----------------------------------------------------------------------------------+
```

### Component Breakdown
1.  **Streamlit UI**: A lightweight, clean interface for user interaction.
2.  **LangGraph Agent**: The central brain. It maintains state, critiques its own retrieval results, and decides when to search.
3.  **Redis**: used for **Checkpointing**. It stores the serialized state of the conversation graph, enabling long-running threading and persistence.
4.  **MCP Server**: A standalone process I implemented to expose RAG capabilities.
5.  **Milvus**: I selected Milvus for its high-performance **Vector Search** capability, which is crucial for handling semantic queries.
6.  **Ollama**: Handles local inference, ensuring all data stays on the machine.

---

## 2. Agentic Workflow Design

I implemented a **Cyclic Graph** pattern for the agent. Unlike a standard RAG chain that blindly answers, this agent has a "Self-Correction" phase.

### Workflow Logic

```
   [ START ]
       |
       v
  ( Retrieve ) <-------------------------------------+
       |                                             |
       v                                             |
[ Grade Documents ]                                  |
       |                                             |
       +---( Irrelevant/Noise )---> ( Rewrite Query )+
       |
       v
 ( All Relevant )
       |
       v
   ( Generate ) <-----------------------------+
       |                                      |
       v                                      |
[ Hallucination Check ]                       |
       |                                      |
       +---> ( Hallucinated ) ----------------+
       |
       v
  ( Grounded )
       |
       v
    [ END ]
```

**Key Behaviors:**
*   **Self-Correction Loop**: If the retrieved documents do not contain useful information, the agent does not give up. It reformulates the query and searches again.
*   **Groundedness Check**: Before showing the answer to the user, the agent performs a final pass to ensure the answer is supported by the context, minimizing hallucinations.

---

## 3. Context Construction Strategy

To achieve the "Maximum Performance" requirement, I focused on how data is retrieved and fed to the LLM.

### A. Retrieval Strategy
I implemented a **Dense Vector Search** approach in Milvus:
1.  **Dense Retrieval**: Catches semantic meaning (e.g., "The network is down" matches "Connection timeout").
2.  **Semantic Matching**: Ensures the agent can find conceptually relevant info even when phrasing differs.

### B. Smart Chunking
I used a **Recursive Character Splitter** (1000 characters with 200 overlap). This ensures that sentences are not cut in the middle, preserving the semantic context for the embedding model.

### C. Prompt Orchestration
I use a structured prompt template that explicitly separates the **system instruction**, the **retrieved context**, and the **user's query**. This ensures the LLM understands that it must prioritize the retrieved data over its internal weights, preventing it from ignoring the "Knowledge Base".

---

## 4. Technology Choices & Rationale

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **LLM** | **Ollama** | The assignment offered clear bonus points for self-hosted LLMs. It also prevents data leakage to cloud APIs. |
| **Vector DB** | **Milvus** | Chosen for its scalability and native support for high-scale vector search, which distinguishes it from simpler stores like Chroma. |
| **Orchestrator** | **LangGraph** | I needed a graph-based framework to implement loops (cycles). Linear chains (LangChain) cannot easily handle "Retry" logic. |
| **Interface** | **Streamlit** | Allowed me to build a clean UI rapidly, keeping the focus on the backend complexity. |
| **Protocol** | **MCP** | This was a strategic choice to decouple the architecture. It proves I can build systems that interoperate with the broader AI ecosystem. |
| **Memory** | **Redis** | In-memory storage is volatile. redis ensures that if the service restarts, the user's conversation history is preserved. |

---

## 5. Key Design Decisions

1.  **Async I/O**: I implemented the entire pipeline using Python's `asyncio`. This ensures that while the system is waiting for Milvus or the LLM, the server remains responsive to other requests.
2.  **Dockerized Infrastructure**: To ensure the system is portable and easy to set up, I containerized the complex dependencies (Milvus, Redis).
3.  **Observability**: I integrated structured logging to make the agent's thought process visible. This turns the "Black Box" into a "Glass Box".

## 6. Limitations

1.  **Hardware Requirements**: Running local LLMs is resource-intensive. The system performance is directly tied to the available GPU VRAM.
2.  **Context Window**: While RAG helps, the local model (Llama 3) has a fixed context window. Extremely long summaries of 100+ documents would require a Map-Reduce approach which I did not implement in this version.
3.  **Response Latency**: Since I am running the LLM locally on consumer hardware, I've observed that response times can be slower compared to cloud-hosted APIs. This leads to a visible delay before the user sees the answer.

## 7. Future Improvements & Mitigation Strategies

To address the **Response Latency** limitation, I can implement the following improvements:

1.  **Cloud LLM Hosting**: I could migrate the inference layer to a hosted provider (e.g., Groq, AWS Bedrock, or OpenAI) to leverage enterprise-grade GPUs.
2.  **Streaming Responses**: I can implement Token Streaming. Instead of waiting for the full response to generate, I would stream tokens to the UI as they are produced.
    *   **Benefit**: This would reduce the **Time to First Token (TTFT)** to under **1 second**, making the application feel instant even if the full answer takes longer to complete.

To address the **Hardware Requirements** limitation:

1.  **Quantization**: I can use 4-bit or 8-bit quantized models to reduce VRAM usage significantly without major loss in quality.
2.  **Model Distillation**: I could experiment with smaller, distilled models (like Llama 3 8B or Phi-3) that are optimized for consumer hardware.

To address the **Context Window** limitation:

1.  **Map-Reduce Summarization**: For extremely large document sets, I can implement a Map-Reduce agent that summarizes chunks in parallel before feeding them to the final answer generation step.
2.  **Sliding Window Context**: I can implement a sliding window approach to process long documents in segments, ensuring the most relevant parts are always within the context limit.

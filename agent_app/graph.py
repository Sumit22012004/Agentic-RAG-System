"""
LangGraph workflow definition.
Builds the cyclic RAG graph with self-correction.
"""
import logging
from langgraph.graph import StateGraph, END

from .state import AgentState
from .memory import ConversationMemory, save_history_background
from .nodes import (
    retrieve,
    grade_documents,
    rewrite_query,
    generate,
    check_hallucination,
    should_retrieve_again,
    should_regenerate,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and return the compiled RAG agent graph."""
    
    # Create graph with our state type
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_check", check_hallucination)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Edges
    workflow.add_edge("retrieve", "grade")
    
    # After grading, decide: generate or rewrite
    workflow.add_conditional_edges(
        "grade",
        should_retrieve_again,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        }
    )
    
    # After rewrite, try retrieval again
    workflow.add_edge("rewrite", "retrieve")
    
    # After generate, check for hallucination
    workflow.add_edge("generate", "hallucination_check")
    
    # After hallucination check, end
    workflow.add_conditional_edges(
        "hallucination_check",
        should_regenerate,
        {
            "end": END,
        }
    )
    
    logger.info("RAG graph built successfully")
    return workflow.compile()


# Pre-compiled graph for reuse
rag_graph = None


def get_graph():
    """Get or create the compiled graph."""
    global rag_graph
    if rag_graph is None:
        rag_graph = build_graph()
    return rag_graph


async def run_agent(question: str, session_id: str) -> str:
    """
    Run the RAG agent on a question.
    Returns the generated answer.
    """
    graph = get_graph()
    # Get chat history from Redis (sync call)
    memory = ConversationMemory(session_id)
    history = memory.get_history_sync()
    
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "retrieval_count": 0,
        "generation_count": 0,
        "chat_history": history,
    }
    
    logger.info(f"[Agent] Processing: '{question[:50]}...' with history len: {len(history)}")
    
    # Run the graph
    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(initial_state, config)
    
    answer = result.get("answer", "I couldn't find an answer to your question.")
    
    # Save new turn to history in background
    save_history_background(session_id, question, answer)
    
    logger.info(f"[Agent] Completed, answer length: {len(answer)}")
    
    return answer

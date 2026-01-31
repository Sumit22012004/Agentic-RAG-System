"""
Graph state definition for the RAG agent.
"""
from typing import List, TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    """State that flows through the graph."""
    
    # User's question
    question: str
    
    # Retrieved documents (accumulated across retries)
    documents: Annotated[List[str], add]
    
    # Generated answer
    answer: str
    
    # Number of retrieval attempts (to prevent infinite loops)
    retrieval_count: int
    
    # Number of generation attempts
    generation_count: int

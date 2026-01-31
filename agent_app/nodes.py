"""
Graph nodes - the actual logic of each step in the workflow.
"""
import logging
from .state import AgentState
from .tools import call_llm, search_knowledge_base

logger = logging.getLogger(__name__)

# Limits to prevent infinite loops
MAX_RETRIEVAL_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 2


async def retrieve(state: AgentState) -> dict:
    """Retrieve documents from the knowledge base."""
    question = state["question"]
    count = state.get("retrieval_count", 0)
    
    logger.info(f"[Retrieve] Attempt {count + 1}, query: '{question[:50]}...'")
    
    docs = await search_knowledge_base(question)
    
    return {
        "documents": [docs] if docs else [],
        "retrieval_count": count + 1,
    }


async def grade_documents(state: AgentState) -> dict:
    """Check if retrieved documents are relevant."""
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        logger.info("[Grade] No documents to grade")
        return {"documents": []}
    
    # Combine all docs
    context = "\n".join(documents)
    
    prompt = f"""Given the question and context below, determine if the context contains relevant information to answer the question.

Question: {question}

Context: {context[:2000]}

Respond with only 'yes' or 'no'."""

    response = await call_llm(prompt)
    is_relevant = "yes" in response.lower()
    
    logger.info(f"[Grade] Documents relevant: {is_relevant}")
    
    if is_relevant:
        return {}  # Keep documents as-is
    else:
        return {"documents": []}  # Clear irrelevant docs


async def rewrite_query(state: AgentState) -> dict:
    """Rewrite the query to improve retrieval."""
    question = state["question"]
    
    logger.info(f"[Rewrite] Improving query: '{question[:50]}...'")
    
    prompt = f"""The following question did not return good results from the knowledge base. 
Rewrite it to be more specific and searchable.

Original question: {question}

Rewritten question:"""

    new_question = await call_llm(prompt)
    new_question = new_question.strip()
    
    logger.info(f"[Rewrite] New query: '{new_question[:50]}...'")
    
    return {"question": new_question}


async def generate(state: AgentState) -> dict:
    """Generate an answer using the retrieved context."""
    question = state["question"]
    documents = state.get("documents", [])
    gen_count = state.get("generation_count", 0)
    
    logger.info(f"[Generate] Attempt {gen_count + 1}")
    
    context = "\n".join(documents) if documents else "No context available."
    
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, say so honestly.
Keep answers clear and concise."""

    prompt = f"""Context:
{context[:4000]}

Question: {question}

Answer:"""

    answer = await call_llm(prompt, system_prompt)
    
    return {
        "answer": answer,
        "generation_count": gen_count + 1,
    }


async def check_hallucination(state: AgentState) -> dict:
    """Check if the answer is grounded in the context."""
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    
    if not documents or not answer:
        return {}
    
    context = "\n".join(documents)
    
    prompt = f"""Check if this answer is supported by the context. 

Context: {context[:2000]}

Answer: {answer}

Is the answer fully supported by the context? Reply only 'yes' or 'no'."""

    response = await call_llm(prompt)
    is_grounded = "yes" in response.lower()
    
    logger.info(f"[Hallucination Check] Grounded: {is_grounded}")
    
    # We keep the answer either way, but log if it's not grounded
    if not is_grounded:
        logger.warning("[Hallucination Check] Answer may not be fully grounded")
    
    return {}


def should_retrieve_again(state: AgentState) -> str:
    """Decide whether to retry retrieval or give up."""
    documents = state.get("documents", [])
    retrieval_count = state.get("retrieval_count", 0)
    
    # If we have docs, move to generate
    if documents:
        return "generate"
    
    # If we've tried too many times, give up
    if retrieval_count >= MAX_RETRIEVAL_ATTEMPTS:
        logger.warning(f"[Router] Max retrieval attempts reached")
        return "generate"  # Generate with whatever we have
    
    # Otherwise, rewrite and retry
    return "rewrite"


def should_regenerate(state: AgentState) -> str:
    """Decide whether to regenerate or finish."""
    gen_count = state.get("generation_count", 0)
    
    # For now, just generate once. Could add hallucination-based retry here.
    if gen_count >= MAX_GENERATION_ATTEMPTS:
        return "end"
    
    return "end"

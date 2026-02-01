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
    history = state.get("chat_history", [])
    
    # Contextualize query if we have history
    final_query = question
    if history and count == 0:
        logger.info("[Retrieve] Contextualizing query with history...")
        
        # Format history string
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]])
        
        prompt = f"""Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{history_str}

Latest Question: {question}

Standalone Question:"""
        
        search_query = await call_llm(prompt)
        final_query = search_query.strip()
        logger.info(f"[Retrieve] Contextualized query: '{final_query}'")
    
    logger.info(f"[Retrieve] Attempt {count + 1}, query: '{final_query[:50]}...'")
    
    docs = await search_knowledge_base(final_query)
    
    return {
        "documents": [docs] if docs else [],
        "retrieval_count": count + 1,
        # Update question in state to the comprehensive one for future steps
        "question": final_query 
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
    
    system_prompt = """You are a helpful and knowledgeable AI assistant. Your goal is to provide concise yet complete answers based strictly on the provided context.

Rules:
1. Be Concise: Provide short, focused answers that directly address the user's question. Avoid unnecessary elaboration or verbosity.
2. Balance: Strike a balance between brevity and clarity—answers should be short but informative enough to be useful.
3. Structure: Use bullet points or lists only when necessary to clarify multiple points. Prefer brief paragraphs for single topics.
4. Tone: Maintain a professional, friendly, and conversational tone. Be direct but not blunt.
5. Accuracy: Use ONLY the provided context. Do not invent information.
6. Long Answers: Only provide comprehensive, detailed answers if the user explicitly asks for a "detailed answer", "explain thoroughly", "long answer", or similar.
7. Missing Info: If the context is insufficient, briefly explain what is missing."""

    prompt = f"""Context:
{context[:4000]}

User's Question: {question}

Answer (be helpful and conversational):"""

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

"""
Redis-based memory for conversation persistence.
Uses LangGraph's checkpointer interface.
"""
import logging
from typing import Optional
import redis.asyncio as redis
from langgraph.checkpoint.memory import MemorySaver

from .config import REDIS_URL

logger = logging.getLogger(__name__)

# Redis connection pool
_redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get or create Redis connection."""
    global _redis_client
    if _redis_client is None:
        logger.info(f"Connecting to Redis: {REDIS_URL}")
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        # Test connection
        await _redis_client.ping()
        logger.info("Connected to Redis")
    return _redis_client


async def close_redis():
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


def get_checkpointer():
    """
    Get a checkpointer for LangGraph.
    
    Note: LangGraph has built-in Redis checkpointer in newer versions.
    For now, using MemorySaver (in-memory) as fallback.
    In production, swap this for RedisSaver when available.
    """
    # Using in-memory for simplicity
    # The Redis connection above is used for manual state storage if needed
    return MemorySaver()


class ConversationMemory:
    """Simple conversation history stored in Redis."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.key = f"chat:{session_id}"
    
    async def add_message(self, role: str, content: str):
        """Add a message to history."""
        client = await get_redis()
        message = f"{role}:{content}"
        await client.rpush(self.key, message)
        # Keep only last 20 messages
        await client.ltrim(self.key, -20, -1)
    
    async def get_history(self) -> list:
        """Get conversation history."""
        client = await get_redis()
        messages = await client.lrange(self.key, 0, -1)
        history = []
        for msg in messages:
            if ":" in msg:
                role, content = msg.split(":", 1)
                history.append({"role": role, "content": content})
        return history
    
    async def clear(self):
        """Clear conversation history."""
        client = await get_redis()
        await client.delete(self.key)

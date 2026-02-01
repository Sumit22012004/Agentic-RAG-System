"""
Redis-based memory for conversation persistence.
Uses synchronous Redis to avoid event loop conflicts with Streamlit.
"""
import logging
import threading
from typing import Optional, List, Dict
import redis

from .config import REDIS_URL

logger = logging.getLogger(__name__)

# Sync Redis client (thread-safe)
_redis_client: Optional[redis.Redis] = None
_lock = threading.Lock()


def get_redis_sync() -> redis.Redis:
    """Get or create sync Redis connection."""
    global _redis_client
    with _lock:
        if _redis_client is None:
            logger.info(f"Connecting to Redis (sync): {REDIS_URL}")
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            # Test connection
            _redis_client.ping()
            logger.info("Connected to Redis")
    return _redis_client


class ConversationMemory:
    """Simple conversation history stored in Redis (sync)."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.key = f"chat:{session_id}"
    
    def add_message_sync(self, role: str, content: str):
        """Add a message to history (sync)."""
        client = get_redis_sync()
        message = f"{role}:{content}"
        client.rpush(self.key, message)
        # Keep only last 20 messages
        client.ltrim(self.key, -20, -1)
    
    def get_history_sync(self) -> List[Dict[str, str]]:
        """Get conversation history (sync)."""
        client = get_redis_sync()
        messages = client.lrange(self.key, 0, -1)
        history = []
        for msg in messages:
            if ":" in msg:
                role, content = msg.split(":", 1)
                history.append({"role": role, "content": content})
        return history
    
    def clear_sync(self):
        """Clear conversation history (sync)."""
        client = get_redis_sync()
        client.delete(self.key)

    # Async wrappers for compatibility (just call sync versions)
    async def add_message(self, role: str, content: str):
        """Async wrapper - runs sync code."""
        self.add_message_sync(role, content)
    
    async def get_history(self) -> List[Dict[str, str]]:
        """Async wrapper - runs sync code."""
        return self.get_history_sync()
    
    async def clear(self):
        """Async wrapper - runs sync code."""
        self.clear_sync()


def save_history_background(session_id: str, question: str, answer: str):
    """Save chat history in a background thread."""
    def _save():
        try:
            memory = ConversationMemory(session_id)
            memory.add_message_sync("user", question)
            memory.add_message_sync("assistant", answer)
            logger.info(f"[Background] Saved chat history for session {session_id[:8]}")
        except Exception as e:
            logger.error(f"[Background] Failed to save history: {e}")
    
    t = threading.Thread(target=_save, daemon=True)
    t.start()

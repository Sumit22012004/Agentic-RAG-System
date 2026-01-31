"""
Milvus database layer.
Handles connection, indexing, and vector search.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from pymilvus import MilvusClient, DataType
from pymilvus.exceptions import MilvusException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, TOP_K, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Thread pool for running sync Milvus ops in async context
_executor = ThreadPoolExecutor(max_workers=4)


class MilvusDB:
    """Milvus wrapper with vector search."""

    def __init__(self):
        self.client: Optional[MilvusClient] = None
        self.collection_name = COLLECTION_NAME

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(MilvusException),
    )
    def connect(self) -> None:
        """Connect to Milvus."""
        uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        logger.info(f"Connecting to Milvus at {uri}")
        self.client = MilvusClient(uri=uri)
        logger.info("Connected to Milvus")

    def _check_connection(self):
        if self.client is None:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        self._check_connection()

        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' exists")
            return

        logger.info(f"Creating collection '{self.collection_name}'")

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )

        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        )

        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=512,
        )

        # Index for dense vectors
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{self.collection_name}' created")

    def _insert_sync(self, texts: List[str], embeddings: List[List[float]], sources: List[str]) -> int:
        """Sync insert."""
        self._check_connection()

        if not texts:
            return 0

        data = [
            {"text": t, "dense_vector": e, "source": s}
            for t, e, s in zip(texts, embeddings, sources)
        ]

        result = self.client.insert(collection_name=self.collection_name, data=data)
        count = len(result["ids"]) if isinstance(result, dict) else len(texts)
        logger.info(f"Inserted {count} documents")
        return count

    async def insert(self, texts: List[str], embeddings: List[List[float]], sources: List[str]) -> int:
        """Async insert wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._insert_sync, texts, embeddings, sources)

    def _search_sync(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Sync search."""
        self._check_connection()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "source"],
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "text": hit["entity"].get("text", ""),
                    "source": hit["entity"].get("source", "unknown"),
                    "score": hit["distance"],
                })

        return docs

    async def search(self, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Async search wrapper."""
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(_executor, self._search_sync, query_embedding, top_k)
        
        if docs:
            logger.info(f"Found {len(docs)} docs, top score: {docs[0]['score']:.3f}")
        else:
            logger.info("No results found")
        
        return docs

    def drop_collection(self) -> None:
        """Drop the collection (useful for resetting)."""
        self._check_connection()
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Dropped collection '{self.collection_name}'")

    def close(self) -> None:
        """Close connection."""
        if self.client:
            self.client.close()
            logger.info("Milvus connection closed")


# Singleton
db = MilvusDB()

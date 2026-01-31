"""
Document ingestion pipeline.
Loads files, chunks them, generates embeddings, stores in Milvus.
"""
import logging
import asyncio
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
)

from .config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=2)

# Load embedding model once (lazy init)
_embedding_model: SentenceTransformer = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _embedding_model


# File type -> loader mapping
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def _load_document_sync(file_path: str) -> List[str]:
    """Load document - runs in thread pool."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    if ext not in LOADERS:
        supported = ", ".join(LOADERS.keys())
        raise ValueError(f"Unsupported file type: {ext}. Supported: {supported}")
    
    logger.info(f"Loading: {path.name}")
    
    loader = LOADERS[ext](str(path))
    docs = loader.load()
    texts = [doc.page_content for doc in docs if doc.page_content.strip()]
    
    logger.info(f"Loaded {len(texts)} pages from {path.name}")
    return texts


async def load_document(file_path: str) -> List[str]:
    """Async document loader."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _load_document_sync, file_path)


def chunk_text(texts: List[str], source: str) -> Tuple[List[str], List[str]]:
    """Split texts into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    sources = []
    
    for text in texts:
        if text.strip():
            text_chunks = splitter.split_text(text)
            chunks.extend(text_chunks)
            sources.extend([source] * len(text_chunks))
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks, sources


def _generate_embeddings_sync(texts: List[str]) -> List[List[float]]:
    """Generate embeddings - runs in thread pool."""
    if not texts:
        return []
    
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Async embedding generation."""
    if not texts:
        return []
    
    logger.info(f"Generating embeddings for {len(texts)} chunks")
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(_executor, _generate_embeddings_sync, texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


async def ingest_file(file_path: str, db) -> int:
    """
    Full pipeline: load -> chunk -> embed -> store.
    Returns number of chunks stored.
    """
    logger.info(f"Starting ingestion: {file_path}")
    
    # Load
    texts = await load_document(file_path)
    if not texts:
        logger.warning(f"No content in {file_path}")
        return 0
    
    # Chunk
    chunks, sources = chunk_text(texts, file_path)
    if not chunks:
        logger.warning(f"No chunks from {file_path}")
        return 0
    
    # Embed
    embeddings = await generate_embeddings(chunks)
    
    # Make sure collection exists
    db.ensure_collection()
    
    # Store
    count = await db.insert(chunks, embeddings, sources)
    
    logger.info(f"Ingested {count} chunks from {file_path}")
    return count

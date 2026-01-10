"""
RAG Manager for MarketLens Agent

Handles vector database operations for document indexing and retrieval.
Uses ChromaDB with local HuggingFace embeddings for efficient semantic search.

Features:
- Automatic chunk size optimization
- Source attribution in retrieved context
- Safe database operations with error handling
- GPU acceleration support

Author: Research Team
Version: 2.0.0
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    PERSIST_DIR: Path = Path("./chroma_db")
    COLLECTION_NAME: str = "market_research"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DEFAULT_K: int = 20
    MAX_K: int = 50


config = RAGConfig()

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_device() -> str:
    """
    Detect the best available device for embeddings.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS backend available")
        return "mps"
    else:
        logger.info("No GPU detected, using CPU")
        return "cpu"


def build_cited_context(docs: List[Document]) -> str:
    """
    Build a context string with numbered source citations.

    Args:
        docs: Retrieved documents

    Returns:
        Formatted context string with citations
    """
    if not docs:
        return ""

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")

        if source.startswith("data/"):
            source = Path(source).name

        context_parts.append(f"[{i}] Source: {source}")
        context_parts.append(doc.page_content)
        context_parts.append("")

    return "\n".join(context_parts)


# =============================================================================
# RAG MANAGER
# =============================================================================

class RAGManager:
    """
    Manages vector database operations for document retrieval.

    Responsibilities:
    - Document chunking and embedding
    - Vector storage using ChromaDB
    - Semantic retrieval
    - Database lifecycle management
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        persist_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.persist_dir = persist_dir or config.PERSIST_DIR
        self.device = device or detect_device()

        logger.info("Initializing RAG Manager")
        logger.info(f"Embedding model: {self.embedding_model}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Persistence directory: {self.persist_dir}")

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._init_embeddings()
        self._init_db()

        logger.info("RAG Manager initialized successfully")

    def _init_embeddings(self) -> None:
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model}")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32
                }
            )

            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embedding dimension: {len(test_embedding)}")

        except Exception as e:
            logger.error(f"Embedding initialization failed: {e}")
            raise RuntimeError("Failed to initialize embeddings")

    def _init_db(self) -> None:
        """Initialize the ChromaDB vector store."""
        try:
            logger.info("Initializing ChromaDB")

            self.vector_store = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir)
            )

            try:
                doc_count = self.vector_store._collection.count()
                logger.info(f"ChromaDB ready with {doc_count} documents")
            except Exception:
                logger.info("ChromaDB ready with new collection")

        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise RuntimeError("Failed to initialize vector store")

    def clear(self) -> None:
        """
        Safely clear all documents from the vector store.
        """
        logger.info("Clearing vector database")

        try:
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                logger.warning(f"Collection deletion warning: {e}")

            self._init_db()
            logger.info("Vector database cleared and reinitialized")

        except Exception as e:
            logger.error(f"Database clear failed: {e}")
            raise RuntimeError("Failed to clear vector database")

    def add_documents(
        self,
        text_content: str,
        source_url: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> int:
        """
        Add text content to the vector store.

        Returns:
            Number of chunks added
        """
        if not text_content.strip():
            raise ValueError("Text content is empty")

        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        logger.info(f"Indexing documents from source: {source_url}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text_content)

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": source_url,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        self.vector_store.add_documents(docs)
        logger.info(f"Added {len(docs)} chunks")

        return len(docs)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        """
        k = min(k or config.DEFAULT_K, config.MAX_K)

        logger.info(f"Retrieving documents for query: {query[:50]}")

        try:
            search_kwargs = {"k": k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict

            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            results = retriever.invoke(query)

            logger.info(f"Retrieved {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the vector store.
        """
        try:
            collection = self.vector_store._collection
            doc_count = collection.count()

            results = collection.get(include=["metadatas"])
            sources = {
                m.get("source")
                for m in results.get("metadatas", [])
                if m.get("source")
            }

            return {
                "total_documents": doc_count,
                "unique_sources": len(sources),
                "sources": list(sources),
                "embedding_model": self.embedding_model,
                "device": self.device
            }

        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}")
            return {"error": str(e)}
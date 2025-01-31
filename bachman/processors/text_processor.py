"""
Text processing module for handling text content.
"""

import hashlib
from typing import Optional, List, Dict
import logging

# import datetime
import uuid

# from datetime import UTC
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from bachman.processors.chunking import ChunkingConfig, ChunkingStrategy
from bachman.processors.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles text processing operations including chunking and hashing.

    Provides consistent handling of text content across different input sources
    with support for chunk tracking and content hashing.
    """

    def __init__(
        self, vector_store: VectorStore, chunking_config: Optional[dict] = None
    ):
        self.vector_store = vector_store
        self.default_chunking_config = chunking_config or {
            "strategy": "recursive",
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "separators": ["\n\n", "\n", ". ", " ", ""],
            "min_chunk_size": 50,
        }
        self.logger = logging.getLogger(__name__)

    def get_text_splitter(
        self, config: Optional[ChunkingConfig] = None, doc_type: Optional[str] = None
    ):
        """Get appropriate text splitter based on strategy and document type."""
        if not config:
            # If no config provided, try to get from chunking_config based on doc_type
            if doc_type and self.chunking_config and doc_type in self.chunking_config:
                cfg = self.chunking_config[doc_type]
                config = ChunkingConfig(
                    strategy=cfg.get("strategy", ChunkingStrategy.RECURSIVE),
                    chunk_size=cfg.get("chunk_size", 4096),
                    chunk_overlap=cfg.get("chunk_overlap", 200),
                    separators=cfg.get("separators", ["\n\n", "\n", " ", ""]),
                    min_chunk_size=cfg.get("min_chunk_size", 100),
                )
            else:
                # Use default config
                config = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE)

        if config.strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators or ["\n\n", "\n", " ", ""],
            )
        elif config.strategy == ChunkingStrategy.SENTENCE:
            return SentenceTransformersTokenTextSplitter(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )

    def _generate_text_hashes(self, text: str, chunks: List[str]) -> Dict:
        """Generate consistent hashes for original text and chunks"""
        parent_id = hashlib.sha256(text.encode()).hexdigest()
        chunk_hashes = [
            {
                "chunk_id": hashlib.sha256(chunk.encode()).hexdigest(),
                "parent_id": parent_id,
                "sequence": idx,
            }
            for idx, chunk in enumerate(chunks)
        ]

        return {"parent_id": parent_id, "chunks": chunk_hashes}

    async def process_text(
        self,
        text: str,
        collection_name: str,
        metadata: dict = None,
        skip_if_exists: bool = False,
        chunking_config: dict = None,
    ):
        """Process text with specified chunking configuration."""
        try:
            # Use provided config or fall back to default
            config = chunking_config or self.default_chunking_config
            self.logger.info(f"Processing text with chunking config: {config}")
            self.vector_store.collection_name = collection_name

            # Create text splitter based on config
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                separators=config["separators"],
                length_function=len,
            )

            # Split text into chunks
            chunks = text_splitter.split_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks")

            # Generate embeddings and store in vector store
            doc_id = str(uuid.uuid4())
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "parent_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {}),
                }

                # Store documents in vector store
                self.logger.info("Adding %d chunks to vector store", len(chunks))
                self.logger.info(
                    f"About to add chunks to collection: {collection_name}"
                )

                print(f"collection_name: {collection_name}")
                # Store chunk with metadata
                await self.vector_store.add_text(
                    text=chunk,
                    metadata=chunk_metadata,
                    cid=f"{doc_id}_chunk_{i}",
                    collection_name=collection_name,
                )

            self.logger.info(f"Successfully processed text with ID {doc_id}")
            return {
                "status": "success",
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "chunking_config": config,
            }

        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            raise

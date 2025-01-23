"""
Text processing module for handling text content.
"""

import hashlib
from typing import Optional, List, Dict
import logging
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from bachman.processors.chunking import ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles text processing operations including chunking and hashing.

    Provides consistent handling of text content across different input sources
    with support for chunk tracking and content hashing.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_text_splitter(self, config: Optional[ChunkingConfig] = None):
        """Get appropriate text splitter based on strategy."""
        if not config:
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
        # Add other strategies as needed

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
        metadata: Optional[Dict] = None,
        skip_if_exists: bool = False,
        chunking_config: Optional[ChunkingConfig] = None,
    ) -> dict:
        """Process single text into vector store."""
        try:
            if skip_if_exists:
                # Check if similar text exists
                if self.vector_store.text_exists(text, collection_name):
                    return {
                        "status": "skipped",
                        "message": "Similar text already exists",
                    }

            # Get text splitter and split text
            text_splitter = self.get_text_splitter(chunking_config)
            texts = text_splitter.split_text(text)

            # Prepare metadata for each chunk
            if metadata:
                metadatas = [metadata.copy() for _ in texts]
                # Add chunk information to metadata
                for i, meta in enumerate(metadatas):
                    meta.update({"chunk_index": i, "total_chunks": len(texts)})
            else:
                metadatas = None

            # Generate hashes
            hash_info = self._generate_text_hashes(text, texts)

            # Add hash info to metadata
            if metadata is None:
                metadata = {}
            metadata.update(hash_info)

            # Store text chunks in vector store
            result = self.vector_store.store_vectors(
                collection_name=collection_name, texts=texts, metadatas=metadatas
            )

            return {
                "status": "success",
                "storage": result,
                "chunks_processed": len(texts),
                "hash_info": hash_info,
            }
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

"""
Vector store module for document storage and retrieval.

This module provides a VectorStore class that interfaces with Qdrant vector database
for storing and retrieving document embeddings. It handles:
- Collection management (creation, deletion, recreation)
- Vector storage and retrieval
- Metadata management
- Similarity search

The VectorStore class provides a high-level interface that abstracts away the 
complexity of working directly with Qdrant, while maintaining flexibility for
different embedding models and search configurations.

Example:
    store = VectorStore(
        host="localhost", 
        port=6333,
        embedding_function=get_embeddings()
    )
    store.store_vectors(
        collection_name="financial_docs",
        texts=["document text"],
        metadatas=[{"source": "10K"}]
    )
"""

# Standard library imports
import os
import sys
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional  # , Union

# import traceback
# import asyncio
import psutil
import torch

# Third-party imports
from langchain.schema import Document
from qdrant_client import QdrantClient  # , AsyncQdrantClient
from qdrant_client.http import models

# from qdrant_client.models import Distance, VectorParams, CollectionStatus

# from tqdm import tqdm
import requests
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from tqdm.auto import tqdm

# import urllib.parse

# Local application imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

logger = logging.getLogger(__name__)


class CollectionDescription(BaseModel):
    """Model for a single collection."""

    name: str
    # Add other fields that come from Qdrant collection description


class CollectionResponse(BaseModel):
    """Model for collection response."""

    result: Dict[
        str, List[Dict[str, str]]
    ]  # {"collections": [{"name": "test_collection"}]}
    status: str
    time: float


class CollectionInfo(BaseModel):
    """Model for collection information."""

    name: str


class VectorStore:
    """Vector store interface for Qdrant database.

    This class provides methods to interact with a Qdrant vector database for storing
    and retrieving document embeddings. It handles collection management, vector storage,
    metadata management, and similarity search operations.

    Attributes:
        host (str): Hostname of the Qdrant server
        port (int): Port number of the Qdrant server
        client (QdrantClient): Initialized Qdrant client instance
        embedding_function: Function to generate embeddings for documents

    Example:
        store = VectorStore(
            host="localhost",
            port=6333,
            embedding_function=get_embeddings()
        )
        store.store_vectors(
            collection_name="financial_docs",
            texts=["document text"],
            metadatas=[{"source": "10K"}]
        )
    """

    # BGE embedding model output dimension
    VECTOR_SIZE = 1024

    def __init__(
        self, host: str, port: int, embedding_function, collection_name: str = None
    ):
        logger.info(
            f"Initializing VectorStore with host={host}, port={port}, collection_name={collection_name}"
        )
        self.host = host
        self.port = port
        self.embedding_function = embedding_function
        self.collection_name = collection_name

        # Log embedding model device
        if hasattr(self.embedding_function, "model"):
            device = next(self.embedding_function.model.parameters()).device
            logger.info(f"Embedding model is on device: {device}")

        self.client = QdrantClient(url=f"http://{host}:8714", timeout=60.0)
        logger.info("Successfully initialized Qdrant client")
        self.vectorstore = None

    def _create_collection(self, collection_name: str) -> bool:
        """Create a new collection using direct Qdrant client."""
        try:
            logger.info(
                f"Creating collection: {collection_name} with vector size {self.VECTOR_SIZE}"
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )
            logger.info(f"Successfully created collection: {collection_name}")
            return True
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Collection {collection_name} already exists")
                return True
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    async def add_documents(
        self, documents: List[Document], batch_size: int = 100
    ) -> None:
        """Add documents to the vector store in batches."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return

            if not self.collection_name:
                raise ValueError("Collection name not set")

            logger.info(
                f"Creating collection {self.collection_name} if it doesn't exist"
            )
            self._create_collection(self.collection_name)

            # Log CPU usage before vectorstore initialization
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"CPU Usage before vectorstore init: {cpu_percent}%")

            # Initialize vectorstore
            logger.info("Initializing vectorstore")
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_function,
            )

            # Log embedding model device again
            if hasattr(self.embedding_function, "model"):
                device = next(self.embedding_function.model.parameters()).device
                logger.info(f"Embedding model device before processing: {device}")

            # Process documents in batches
            total_added = 0
            for i in tqdm(
                range(0, len(documents), batch_size), desc="Adding documents"
            ):
                batch = documents[i : i + batch_size]
                try:
                    # Log CPU and GPU usage before batch processing
                    cpu_percent = psutil.cpu_percent(interval=1)
                    logger.debug(
                        f"CPU Usage before batch {i//batch_size + 1}: {cpu_percent}%"
                    )
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        logger.debug(f"GPU Memory allocated: {gpu_memory:.2f} MB")

                    logger.debug(f"Processing batch {i//batch_size + 1}")
                    self.vectorstore.add_documents(documents=batch)
                    total_added += len(batch)

                    # Log resource usage after batch
                    cpu_percent = psutil.cpu_percent(interval=1)
                    logger.debug(f"CPU Usage after batch: {cpu_percent}%")
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        logger.debug(f"GPU Memory after batch: {gpu_memory:.2f} MB")

                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                    raise

            logger.info(
                f"Successfully added {total_added} documents to collection '{self.collection_name}'"
            )

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    async def add_text(
        self,
        collection_name: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a single text document to the vector store."""
        try:
            if not self.collection_name:
                raise ValueError("Collection name not set")

            logger.info(
                f"Creating collection {self.collection_name} if it doesn't exist"
            )
            self._create_collection(self.collection_name)

            qdrant_id = str(uuid.uuid4())

            # Get embeddings for the text
            vector = await self.get_embeddings(text)

            # Prepare the payload
            payload = {
                "text": text,
                "doc_id": doc_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata if metadata else {}),
            }

            point_data = {
                "points": [
                    {
                        "id": qdrant_id,
                        "vector": vector,
                        "payload": payload,
                    }
                ]
            }

            # Use the upsert endpoint to store the vector
            response = requests.post(
                f"http://{self.host}:8716/collections/{collection_name}/points",
                json=point_data,
                timeout=10,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to store vector: {response.text}")

            result = {"id": qdrant_id, "payload": payload}  # Removed vector from result

            logger.info("\nSuccessfully created new document:")
            logger.info(json.dumps(result, indent=2))

            return {
                **result,
                "vector": vector,
            }  # Include vector in return but not in print

        except Exception as e:
            logger.error(f"Error storing text document: {str(e)}")
            raise

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        try:
            if not self.collection_name:
                raise ValueError("Collection name not set")

            if not self.vectorstore:
                logger.info("Initializing vectorstore for search")
                self.vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embedding_function,  # Changed here too
                )

            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Search using a pre-computed embedding vector.

        Args:
            embedding: Pre-computed embedding vector
            k: Number of results
            metadata_filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        return self.similarity_search(embedding, k, metadata_filter)

    def similarity_search_by_ticker(
        self,
        query: str,
        ticker: str,
        k: int = 4,
        doc_type: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Search documents for a specific ticker.

        Args:
            query: Search query
            ticker: Stock ticker
            k: Number of results
            doc_type: Optional document type filter
            date_range: Optional date range filter {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}

        Returns:
            List of similar documents
        """
        filter_dict = {"ticker": ticker}
        if doc_type:
            filter_dict["doc_type"] = doc_type

        if date_range:
            filter_dict["date"] = {
                "$gte": date_range["start"],
                "$lte": date_range["end"],
            }

        return self.similarity_search(query, k, filter_dict)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            stats = {
                "total_documents": collection_info.points_count,
                "collection_name": self.collection_name,
                "distance_metric": self.distance_metric,
                "vector_size": collection_info.config.params.vectors.size,
            }

            # Get unique values for key metadata fields
            docs = self.get_documents_by_metadata({})
            if docs:
                stats.update(
                    {
                        "unique_tickers": len(
                            set(d.metadata.get("ticker") for d in docs)
                        ),
                        "doc_types": list(
                            set(d.metadata.get("doc_type") for d in docs)
                        ),
                        "date_range": {
                            "earliest": min(
                                d.metadata.get("date")
                                for d in docs
                                if "date" in d.metadata
                            ),
                            "latest": max(
                                d.metadata.get("date")
                                for d in docs
                                if "date" in d.metadata
                            ),
                        },
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def clear_collection(self, confirm: bool = True) -> None:
        """
        Clear all documents from the collection.

        Args:
            confirm: Whether to require confirmation
        """
        try:
            if confirm:
                count = self.client.count(self.collection_name).count
                response = input(
                    f"Are you sure you want to delete {count} documents? (y/N): "
                )
                if response.lower() != "y":
                    logger.info("Collection deletion cancelled")
                    return

            self.client.delete_collection(self.collection_name)
            logger.info(f"Cleared collection {self.collection_name}")

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise

    def get_documents_by_metadata(
        self, metadata_filter: Dict[str, Any]
    ) -> List[Document]:
        """
        Retrieve documents matching metadata criteria.

        Args:
            metadata_filter: Dictionary of metadata filters

        Returns:
            List of matching documents
        """
        try:
            # Convert metadata filter to Qdrant filter format
            filter_conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, dict):  # Handle range queries
                    for op, val in value.items():
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=val),
                                range=models.Range(
                                    gte=val if op == "$gte" else None,
                                    lte=val if op == "$lte" else None,
                                ),
                            )
                        )
                else:
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}", match=models.MatchValue(value=value)
                        )
                    )

            return self.vectorstore.get_matches(
                where_document=models.Filter(must=filter_conditions)
            )

        except Exception as e:
            logger.error(f"Error retrieving documents by metadata: {str(e)}")
            raise

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text."""
        if self.embedding_function is None:
            raise ValueError("Embedding function not initialized")
        # Don't await the result since embed_query already returns the vector
        return self.embedding_function.embed_query(text)

    async def text_exists(self, collection_name: str, doc_id: str) -> bool:
        """Check if a document with given doc_id exists in collection."""
        try:
            print(
                f"\n=== Checking for doc_id: {doc_id} in collection: {collection_name} ==="
            )

            # Ensure collection exists first
            if not await self.ensure_collection_exists(collection_name):
                print("Collection check/creation failed")
                return False

            print("\nSearching for existing document...")
            matching_docs = await self.search_by_doc_id(collection_name, doc_id)

            if matching_docs:
                # Create a clean version of the docs without vectors for printing
                clean_docs = [
                    {"id": doc["id"], "payload": doc["payload"]}
                    for doc in matching_docs
                ]
                print("\nFound existing document(s):")
                print(json.dumps(clean_docs, indent=2))
                return True

            print(f"\nNo existing documents found with doc_id: {doc_id}")
            return False

        except Exception as e:
            print(f"Error checking text existence: {str(e)}")
            return False

    async def store_vectors(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        skip_if_exists: bool = False,
    ) -> Dict[str, Any]:
        """Store vectors with complete metadata."""
        try:
            if skip_if_exists and metadatas and metadatas[0].get("doc_id"):
                doc_id = metadatas[0]["doc_id"]
                exists = await self.text_exists(collection_name, doc_id)
                if exists:
                    return {
                        "status": "skipped",
                        "reason": "document already exists",
                        "doc_id": doc_id,
                    }

            vector = await self.get_embeddings(texts[0])
            point_id = str(uuid.uuid4())

            payload = {
                "text": texts[0],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadatas[0] if metadatas else {}),
            }

            point_data = {
                "points": [{"id": point_id, "vector": vector, "payload": payload}]
            }

            # Use the upsert endpoint
            response = requests.post(
                f"http://{self.host}:8716/collections/{collection_name}/points",
                json=point_data,
                timeout=10,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to store vector: {response.text}")

            result = {"id": point_id, "payload": payload}  # Removed vector from result

            print("\nSuccessfully created new document:")
            print(json.dumps(result, indent=2))

            return {
                **result,
                "vector": vector,
            }  # Include vector in return but not in print

        except Exception as e:
            print(f"Error storing vectors: {str(e)}")
            raise

    def delete_collection(self, collection_name: str):
        """Delete a collection if it exists."""
        try:
            response = requests.delete(
                f"http://{self.host}:{self.port}/collections/{collection_name}",
                timeout=10,  # Add 10 second timeout
            )
            if response.status_code == 200:
                logger.info(f"Successfully deleted collection: {collection_name}")
            else:
                logger.warning(
                    f"Failed to delete collection {collection_name}: {response.text}"
                )

        except requests.Timeout:
            logger.error(
                f"Timeout while deleting collection at {self.host}:{self.port}"
            )
            raise
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            raise

    def get_collections(self) -> Optional[List[str]]:
        """Get list of all collections."""
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/collections", timeout=10
            )
            if response.status_code == 200:
                # Add debug logging to see the actual response
                logger.debug(f"Collections response: {response.text}")
                data = response.json()
                # Modify the response structure to match what Qdrant returns
                collections_data = {
                    "collections": [
                        {"name": c["name"]} for c in data.get("collections", [])
                    ],
                    "status": "ok",
                }
                validated = CollectionResponse.model_validate(collections_data)
                return [c.name for c in validated.collections]
            return None
        except Exception as e:
            logger.warning(f"Could not fetch collections, but continuing: {str(e)}")
            return None

    async def search_by_doc_id(self, collection_name: str, doc_id: str) -> list:
        """Search for documents by doc_id."""
        try:
            response = requests.post(
                f"http://{self.host}:8716/collections/{collection_name}/search",
                json={
                    "query_vector": [0] * 1024,
                    "filter": {
                        "must": [
                            {"key": "doc_id", "match": {"value": doc_id, "exact": True}}
                        ]
                    },
                    "limit": 100,
                },
                timeout=10,
            )

            if response.status_code == 200:
                results = response.json()
                matching_docs = [
                    doc
                    for doc in results
                    if doc.get("payload", {}).get("doc_id") == doc_id
                ]
                if matching_docs:
                    return matching_docs

            return []

        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []

    def search(
        self, collection_name: str, metadata_filter: dict, limit: int = 1
    ) -> list:
        """
        Search for documents in the vector store using metadata filters.

        Args:
            collection_name (str): Name of the collection to search in
            metadata_filter (dict): Metadata filter conditions
            limit (int): Maximum number of results to return

        Returns:
            list: List of matching documents with their metadata
        """
        try:
            # Convert metadata filter to Qdrant filter format
            conditions = {
                "must": [
                    {"key": key, "match": {"value": value}}
                    for key, value in metadata_filter.items()
                ]
            }

            # Perform the search using Qdrant's scroll
            results = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                query_filter=conditions,  # Changed from filter to query_filter
                with_payload=True,
                with_vectors=False,
            )[0]

            # Format the results
            formatted_results = []
            for hit in results:
                formatted_results.append({"metadata": hit.payload, "score": 1.0})

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

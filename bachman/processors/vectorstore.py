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
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Third-party imports
from langchain.schema import Document
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import requests
from pydantic import BaseModel

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

    collections: List[CollectionDescription]
    status: str = "ok"


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

    def __init__(self, host: str, port: int, embedding_function=None):
        """Initialize VectorStore with Qdrant client."""
        self.host = host
        self.port = port
        self.client = QdrantClient(
            host=host,
            port=8716,  # Change to use the API port instead of direct Qdrant port
            check_compatibility=False,
        )
        self.embedding_function = embedding_function
        logger.info(f"Initialized Qdrant client for {host}:{port}")

        try:
            collections = self.client.get_collections()
            if collections is None:
                logger.warning("No collections found, but connection successful")
            else:
                logger.info(
                    f"Found {len(collections.collections)} existing collections"
                )
        except Exception as e:
            logger.warning(f"Could not fetch collections, but continuing: {str(e)}")

    def ensure_collection(self, collection_name: str, force_recreate: bool = False):
        """Ensure collection exists with correct dimensions."""
        try:
            # Check current collection
            response = requests.get(
                f"http://{self.host}:{self.port}/collections/{collection_name}",
                timeout=10,  # Add 10 second timeout
            )

            if response.status_code == 200:
                if force_recreate:
                    logger.info(f"Force recreating collection {collection_name}")
                    self.delete_collection(collection_name)
                else:
                    logger.info(f"Collection {collection_name} exists")
                    return

            # Create collection with correct dimensions
            response = requests.post(
                f"http://{self.host}:{self.port}/collections/{collection_name}",
                json={"size": 1024, "distance": "Cosine"},
                timeout=10,  # Add 10 second timeout
            )
            if response.status_code == 200:
                logger.info(f"Successfully created collection: {collection_name}")
            else:
                raise Exception(f"Failed to create collection: {response.text}")

        except requests.Timeout:
            logger.error(
                f"Timeout while connecting to Qdrant at {self.host}:{self.port}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_name}: {str(e)}")
            raise

    def get_collection(self, collection_name: str):
        """Get collection, creating it if it doesn't exist."""
        self.ensure_collection(collection_name)
        # Verify collection exists after creation
        response = self.client.get_collection(collection_name=collection_name)
        if response is None:
            raise ValueError(f"Failed to verify collection {collection_name} exists")
        return collection_name

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            batch_size: Number of documents to process at once
            show_progress: Whether to show progress bar
        """
        try:
            # Process documents in batches
            batches = range(0, len(documents), batch_size)
            if show_progress:
                batches = tqdm(batches, desc="Adding documents")

            for i in batches:
                batch = documents[i : i + batch_size]
                self.vectordb.add_documents(documents=batch)

            logger.info(f"Added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query or embedding vector
            k: Number of results to return
            metadata_filter: Optional metadata filter
            score_threshold: Minimum similarity score threshold

        Returns:
            List of similar documents
        """
        try:
            search_params = {}
            if score_threshold:
                search_params["score_threshold"] = score_threshold

            return self.vectordb.similarity_search(
                query=query, k=k, filter=metadata_filter, search_params=search_params
            )
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
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

            return self.vectordb.get_matches(
                where_document=models.Filter(must=filter_conditions)
            )

        except Exception as e:
            logger.error(f"Error retrieving documents by metadata: {str(e)}")
            raise

    def store_vectors(
        self,
        collection_name: str,
        texts: Union[str, list],
        metadatas: list = None,
        force_recreate: bool = False,
    ):
        """Store vectors in the specified collection."""
        try:
            # Convert single text to list if necessary
            if isinstance(texts, str):
                texts = [texts]
                if metadatas:
                    metadatas = [metadatas]

            # Ensure collection exists with optional recreation
            self.ensure_collection(collection_name, force_recreate=force_recreate)

            # Generate embeddings
            vectors = [self.embedding_function.embed_query(text) for text in texts]

            # Prepare points for insertion using models.PointStruct format
            points = []
            for i, (vector, text) in enumerate(zip(vectors, texts)):
                point = {
                    "id": str(uuid.uuid4()),
                    "vector": list(vector),  # Convert numpy array to list
                    "payload": {
                        "text": text,
                        "ticker": metadatas[i].get("ticker")
                        if metadatas and metadatas[i]
                        else None,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
                points.append(point)

                # Log the payload (excluding the vector for readability)
                log_point = point.copy()
                log_point["vector"] = f"<vector with {len(vector)} dimensions>"
                logger.info(f"Storing point: {json.dumps(log_point, indent=2)}")

            # Use the server's HTTP API directly
            response = requests.post(
                f"http://{self.host}:{self.port}/collections/{collection_name}/points",
                json={"points": points},
                timeout=30,  # Add 30 second timeout for vector storage
            )

            if response.status_code == 200:
                logger.info(
                    f"Successfully stored {len(texts)} vectors in collection {collection_name}"
                )
                return response.json()
            else:
                logger.error(
                    f"Failed to store vectors. Status: {response.status_code}, Response: {response.text}"
                )
                raise Exception(f"Failed to store vectors: {response.text}")

        except requests.Timeout:
            logger.error(
                f"Timeout while storing vectors in Qdrant at {self.host}:{self.port}"
            )
            raise
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
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

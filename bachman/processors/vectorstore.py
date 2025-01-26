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
from typing import List, Dict, Any, Optional, Union
import traceback
import asyncio

# Third-party imports
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import requests
from pydantic import BaseModel

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

    def __init__(self, host: str, port: int, embedding_function: Any = None):
        """Initialize VectorStore with Qdrant client."""
        self.host = host
        self.port = port
        base_url = f"http://{host}:{port}"
        self.client = QdrantClient(
            url=base_url,
            timeout=60.0,
            prefer_grpc=False,
            check_compatibility=False,
        )
        self.embedding_function = embedding_function

        try:
            collections_url = f"http://{host}:8714/collections"
            logger.debug(f"Attempting to fetch collections from: {collections_url}")

            response = requests.get(collections_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Received response: {data}")

            if isinstance(data, dict) and "result" in data:
                collections = data["result"]["collections"]
                collection_names = [coll["name"] for coll in collections]
                logger.info(f"Found collections via HTTP: {collection_names}")
            else:
                logger.warning(f"Unexpected response format: {data}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"HTTP request failed: {e}")
        except ValueError as e:
            logger.warning(f"JSON parsing failed: {e}")
        except Exception as e:
            logger.warning(f"Could not fetch collections during init: {str(e)}")
            logger.debug("Exception details:", exc_info=True)

        logger.info(f"Initialized Qdrant client for {host}:{port}")

    async def ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists using HTTP API."""
        try:
            # Check if collection exists
            response = requests.get(
                f"http://{self.host}:{self.port}/collections/{collection_name}",
                timeout=10,
            )

            if response.status_code != 200:
                # Create collection if it doesn't exist
                create_response = requests.post(
                    f"http://{self.host}:{self.port}/collections/{collection_name}",
                    json={"vectors": {"size": 1024, "distance": "Cosine"}},
                    timeout=10,
                )
                if create_response.status_code != 200:
                    raise Exception(
                        f"Failed to create collection: {create_response.text}"
                    )
                logger.info(f"Created collection: {collection_name}")

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
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

    async def ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure collection exists, create if it doesn't."""
        try:
            # Check if collection exists
            collections_response = requests.get(
                f"http://{self.host}:8716/collections", timeout=10
            )
            if collections_response.status_code == 200:
                collections_data = collections_response.json()
                print("\nCollections response:", json.dumps(collections_data, indent=2))

                # Parse the collections string looking for the collection name
                collections_str = str(collections_data)
                exists = f"name='{collection_name}'" in collections_str

                if not exists:
                    print(f"\nCreating collection: {collection_name}")
                    create_response = requests.post(
                        f"http://{self.host}:8716/collections/{collection_name}",
                        json={"size": 1024},
                        timeout=10,
                    )

                    if create_response.status_code == 200:
                        print(f"Successfully created collection {collection_name}")
                        # Wait a moment for the collection to be ready
                        await asyncio.sleep(1)
                        return True
                    else:
                        print(
                            f"Failed to create collection: {create_response.status_code}"
                        )
                        print(f"Response: {create_response.text}")
                        return False

                print(f"Collection {collection_name} already exists")
                return True

            print(f"Failed to get collections: {collections_response.status_code}")
            return False

        except Exception as e:
            print(f"Error ensuring collection exists: {str(e)}")
            print(f"Exception type: {type(e)}")
            print(f"Full exception details: {traceback.format_exc()}")
            return False

    async def store_text(
        self, collection_name: str, text: str, metadata: dict = None
    ) -> dict:
        """Store text in vector database."""
        try:
            # Ensure collection exists
            if not await self.ensure_collection_exists(collection_name):
                raise Exception(f"Failed to create collection: {collection_name}")

            # Get embeddings
            vector = await self.get_embeddings(text)
            point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                payload.update(metadata)

            # Create point
            point = {"id": point_id, "vector": vector, "payload": payload}

            # Store the point
            response = await self.client.upsert(
                collection_name=collection_name, points=[point]
            )

            # Wait for storage operation to complete and verify
            if response.status_code == 200:
                # Wait a moment for consistency
                await asyncio.sleep(1)

                # Verify storage
                search_response = await self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=metadata.get("doc_id")),
                            )
                        ]
                    ),
                    limit=1,
                )

                if search_response and search_response.points:
                    stored_point = search_response.points[0]
                    print("\nDocument stored and verified:")
                    print(f"ID: {stored_point.id}")
                    print(f"Doc ID: {stored_point.payload.get('doc_id')}")
                    print(f"Text: {stored_point.payload.get('text')[:100]}...")
                    return {
                        "id": point_id,
                        "vector": vector,
                        "payload": stored_point.payload,
                    }
                else:
                    raise Exception("Document storage could not be verified")

            raise Exception(f"Storage failed with status code: {response.status_code}")

        except Exception as e:
            print(f"\nError storing vectors: {str(e)}")
            print(f"Exception type: {type(e)}")
            print(f"Full exception details: {traceback.format_exc()}")
            raise

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

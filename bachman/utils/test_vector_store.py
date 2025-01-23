"""
Test vector store functionality.
"""
# pylint: disable=wrong-import-position
import sys
import os
from unittest.mock import Mock
import pytest
import numpy as np
from langchain.schema import Document


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from bachman.processors.vectorstore import (
    VectorStore,
)  # pylint: disable=wrong-import-position

# from bachman.models.embeddings import (
#     get_embeddings_model,
# )  # pylint: disable=wrong-import-position


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings model."""
    embeddings = Mock()
    embeddings.embed_query.return_value = np.random.rand(1536).tolist()
    embeddings.embed_documents.return_value = [
        np.random.rand(1536).tolist() for _ in range(3)
    ]
    return embeddings


@pytest.fixture
def vector_store(mock_embeddings):
    """Create test vector store instance."""
    return VectorStore(
        host="0.0.0.0",
        port="8713",
        embedding_function=mock_embeddings,
    )


def test_vector_store_initialization(vector_store):
    """Test vector store initialization."""
    assert vector_store.collection_name == "test_collection"
    assert vector_store.distance_metric == "Cosine"
    assert not vector_store.is_remote


def test_add_documents(vector_store):
    """Test adding documents to vector store."""
    documents = [
        Document(
            page_content="Test document 1",
            metadata={"source": "test", "ticker": "TEST", "doc_type": "10K"},
        ),
        Document(
            page_content="Test document 2",
            metadata={"source": "test", "ticker": "TEST", "doc_type": "10Q"},
        ),
    ]

    vector_store.add_documents(documents, show_progress=False)
    stats = vector_store.get_collection_stats()
    assert stats["total_documents"] == 2


def test_similarity_search(vector_store):
    """Test similarity search functionality."""
    # Add test documents
    documents = [
        Document(
            page_content="Financial report showing strong growth",
            metadata={"ticker": "TEST", "doc_type": "10K"},
        ),
        Document(
            page_content="Quarterly earnings exceeded expectations",
            metadata={"ticker": "TEST", "doc_type": "10Q"},
        ),
    ]
    vector_store.add_documents(documents, show_progress=False)

    # Test search
    query = "financial performance"
    results = vector_store.similarity_search(
        query=query, k=2, filter={"ticker": "TEST"}
    )

    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)


def test_metadata_filtering(vector_store):
    """Test document retrieval by metadata."""
    documents = [
        Document(
            page_content="Annual report 2023",
            metadata={"ticker": "TEST", "doc_type": "10K", "date": "2023-12-31"},
        ),
        Document(
            page_content="Q3 report 2023",
            metadata={"ticker": "TEST", "doc_type": "10Q", "date": "2023-09-30"},
        ),
    ]
    vector_store.add_documents(documents, show_progress=False)

    # Test metadata filtering
    filtered_docs = vector_store.get_documents_by_metadata({"doc_type": "10K"})
    assert len(filtered_docs) == 1
    assert filtered_docs[0].metadata["doc_type"] == "10K"


def test_collection_stats(vector_store):
    """Test collection statistics."""
    documents = [
        Document(
            page_content="Test document",
            metadata={"ticker": "TEST", "doc_type": "10K", "date": "2023-12-31"},
        )
    ]
    vector_store.add_documents(documents, show_progress=False)

    stats = vector_store.get_collection_stats()
    assert isinstance(stats, dict)
    assert "total_documents" in stats
    assert "doc_types" in stats
    assert "unique_tickers" in stats


def test_clear_collection(vector_store, monkeypatch):
    """Test collection clearing with confirmation."""
    # Mock input function to simulate 'y' response
    monkeypatch.setattr("builtins.input", lambda _: "y")

    # Add test document
    documents = [Document(page_content="Test document", metadata={"ticker": "TEST"})]
    vector_store.add_documents(documents, show_progress=False)

    # Clear collection
    vector_store.clear_collection(confirm=True)

    # Verify collection is empty
    stats = vector_store.get_collection_stats()
    assert stats["total_documents"] == 0


if __name__ == "__main__":
    pytest.main([__file__])

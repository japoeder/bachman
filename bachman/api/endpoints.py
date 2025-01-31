"""
This module defines the endpoints for the Bachman API.
"""

import logging
import asyncio
import json

# import uuid
from datetime import datetime
import requests

# import os
from flask import Flask, request, jsonify

# from langchain.schema import Document

from bachman.processors.chunking import ChunkingConfig

# from bachman.processors.document_types import DocumentType
from bachman.api.middleware import requires_api_key
from bachman.core.components import Components
from bachman.utils.format_results import format_analysis_results
from bachman.config.prompt_config import prompt_config
from bachman.config.app_config import (
    QDRANT_HOST,
    # CHUNKING_CONFIGS,
    # EMBEDDING_CONFIGS,
    # COLLECTION_CONFIGS,
)

logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.route("/bachman/analyze", methods=["POST"])
    @requires_api_key
    def analyze():
        """Text analysis endpoint using RAG with Groq LLM for financial documents."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Extract required parameters
            collection_name = data.get("collection_name")
            qdrant_id = data.get("id")

            if not collection_name or not qdrant_id:
                return jsonify({"error": "collection_name and id are required"}), 400

            logger.info(
                f"Analyzing document with Qdrant ID {qdrant_id} from collection {collection_name}"
            )

            if not Components.vector_store:
                return jsonify({"error": "Vector store not initialized"}), 500

            try:
                # First, get the main document metadata using scroll
                main_doc_response = Components.vector_store.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [{"key": "id", "match": {"value": qdrant_id}}]
                    },
                    with_payload=True,
                    limit=1,
                )

                if not main_doc_response[0]:
                    return (
                        jsonify(
                            {
                                "error": f"Document not found in collection {collection_name}"
                            }
                        ),
                        404,
                    )

                document = main_doc_response[0][0].payload
                doc_type = document.get("doc_type")

                # Get all related chunks using parent_id
                chunks_response = Components.vector_store.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [{"key": "parent_id", "match": {"value": qdrant_id}}]
                    },
                    with_payload=True,
                    limit=100,  # Adjust based on your needs
                )

                # Combine relevant sections for analysis
                chunks_text = []
                for chunk in chunks_response[0]:
                    chunks_text.append(chunk.payload.get("text", ""))

                # Create financial report analysis prompt
                analysis_prompt = prompt_config(
                    doc_type=doc_type or "financial_report",
                    ticker=document.get("ticker", "Unknown"),
                    chunks_text=" ".join(chunks_text[:5]),
                )

                # Get analysis from Groq
                analysis_result = Components.sentiment_analyzer.llm(analysis_prompt)

                # Parse the JSON response
                try:
                    parsed_result = json.loads(analysis_result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM response as JSON")
                    parsed_result = {
                        "dominant_sentiment": "UNKNOWN",
                        "confidence_level": "0",
                        "time_horizon": "UNKNOWN",
                        "trading_signal": "UNKNOWN",
                        "key_themes": ["Failed to parse analysis"],
                    }

                # Structure the response
                response = {
                    "analysis": parsed_result,
                    "metadata": {
                        "doc_id": qdrant_id,
                        "collection": collection_name,
                        "doc_type": doc_type,
                        "ticker": document.get("ticker"),
                        "type": "sentiment",
                        "timestamp": datetime.utcnow().isoformat(),
                        "chunks_analyzed": len(chunks_text),
                    },
                }

                # Format the results for logging
                format_analysis_results(
                    analysis_type="sentiment",
                    result=parsed_result,
                    metadata=response["metadata"],
                )

                return jsonify(response), 200

            except Exception as e:
                logger.error(f"Error performing analysis: {str(e)}")
                return jsonify({"error": f"Error performing analysis: {str(e)}"}), 500

        except Exception as e:
            logger.error(f"Error processing analysis request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/process_text", methods=["POST"])
    @requires_api_key
    def process_text():
        """Text processing endpoint without sentiment analysis."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()

            # Extract and validate collection name
            collection_name = data.get("collection_name")
            if not collection_name:
                return jsonify({"error": "collection_name is required"}), 400

            # Get text from request
            text = data.get("text")
            if not text:
                return jsonify({"error": "No text provided"}), 400

            # Extract chunking configuration if provided
            chunking_config = None
            if "chunking_config" in data:
                try:
                    chunking_config = ChunkingConfig(
                        strategy=data["chunking_config"].get("strategy", "recursive"),
                        chunk_size=data["chunking_config"].get("chunk_size", 4096),
                        chunk_overlap=data["chunking_config"].get("chunk_overlap", 200),
                        separators=data["chunking_config"].get("separators"),
                        min_chunk_size=data["chunking_config"].get(
                            "min_chunk_size", 100
                        ),
                    )
                    logger.info(
                        f"Using custom chunking configuration: {chunking_config}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Invalid chunking configuration provided: {e}. Using defaults."
                    )

            # Run the coroutine in the event loop
            result = asyncio.run(
                components.text_processor.process_text(
                    text=text,
                    collection_name=collection_name,
                    metadata=data.get("metadata"),
                    skip_if_exists=data.get("skip_if_exists", False),
                    chunking_config=chunking_config,
                )
            )

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/process_file", methods=["POST"])
    @requires_api_key
    def process_file():
        """Process a file and store its contents."""
        try:
            # Create new components instance for this request
            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()

            data = request.get_json()
            # Extract required parameters
            file_path = data.get("file_path")
            collection_name = data.get("collection_name")
            metadata = data.get("metadata", {})
            temp_dir = data.get("temp_dir")
            cleanup = data.get("cleanup", True)

            if not file_path or not collection_name:
                return jsonify({"error": "Missing required parameters"}), 400

            # Get doc_type specific chunking config
            doc_type = metadata.get("doc_type", "default")
            chunking_config = components.get_chunking_config(doc_type)
            logger.info(
                f"Using chunking config for doc_type '{doc_type}': {chunking_config}"
            )

            result = asyncio.run(
                components.file_processor.process_file(
                    file_path=file_path,
                    collection_name=collection_name,
                    metadata=metadata,
                    chunking_config=chunking_config,
                    temp_dir=temp_dir,
                    cleanup=cleanup,
                )
            )

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/get_sentiment", methods=["POST"])
    @requires_api_key
    def get_sentiment():
        """Retrieve or generate sentiment for previously processed document."""
        try:
            data = request.json
            doc_id = data.get("doc_id")
            if not doc_id:
                return jsonify({"error": "doc_id is required"}), 400

            result = Components.sentiment_analyzer.process_document(
                doc_id=doc_id, collection_name=data.get("collection_name")
            )

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error processing sentiment request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/search", methods=["POST"])
    @requires_api_key
    def search():
        """Search endpoint to query existing documents in the vector store."""
        try:
            data = request.json
            limit = data.get("limit", 100)
            if not data:
                return jsonify({"error": "No data provided"}), 400

            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()

            # Log just the search request details
            logger.info("=" * 80)
            logger.info("SEARCH REQUEST DETAILS")
            logger.info("=" * 80)
            logger.info(f"Request data: {json.dumps(data, indent=2)}")

            # Extract and validate collection name
            collection_name = data.get("collection_name")
            if not collection_name:
                return jsonify({"error": "collection_name is required"}), 400

            # Get query parameters
            metadata_filter = data.get("metadata", {})
            if not metadata_filter:
                return jsonify({"error": "Query metadata is required"}), 400

            # Prepare filter conditions
            filter_conditions = []
            for key, value in metadata_filter.items():
                filter_conditions.append({"key": key, "match": {"value": value}})

            # Make the search request
            search_url = f"http://{components.vector_store.host}:8716/collections/{collection_name}/search"
            search_payload = {
                "query_vector": [0] * 1024,
                "filter": {"must": filter_conditions},
                "limit": limit,
            }

            response = requests.post(
                search_url,
                json=search_payload,
                timeout=10,
            )

            if response.status_code == 200:
                docs = response.json()

                # Format the results
                formatted_results = []
                # for hit in raw_results:
                #     metadata = hit.get("payload", {})
                #     formatted_result = {
                #         "id": hit.get("id"),
                #         "metadata": {
                #             "doc_id": metadata.get("doc_id"),
                #             "ticker": metadata.get("ticker"),
                #             "source": metadata.get("source"),
                #             "doc_type": metadata.get("doc_type"),
                #             "timestamp": metadata.get("timestamp"),
                #         },
                #         "chunks": metadata.get("chunks", []),
                #         "text_preview": metadata.get("text", "")[:100] + "..."
                #         if metadata.get("text")
                #         else None,
                #     }
                #     formatted_results.append(formatted_result)

                # Log just the formatted results
                logger.info("=" * 80)
                logger.info("RESULTS")
                logger.info("=" * 80)

                for doc in docs:
                    doc.pop("vector")
                    formatted_results.append(doc)
                    logger.info(json.dumps(doc, indent=2))

                logger.info("=" * 80)

                return (
                    jsonify(
                        {"count": len(formatted_results), "results": formatted_results}
                    ),
                    200,
                )
            else:
                logger.error(f"Search failed with status {response.status_code}")
                return (
                    jsonify(
                        {"error": f"Search failed with status {response.status_code}"}
                    ),
                    500,
                )

        except Exception as e:
            logger.error(f"Error processing search request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/delete", methods=["PUT"])
    @requires_api_key
    def delete():
        """Delete endpoint to remove either a specific point or an entire collection."""
        try:
            # Create new components instance for this request
            components = Components(qdrant_host=QDRANT_HOST)

            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            collection_name = data.get("collection_name")
            if not collection_name:
                return jsonify({"error": "collection_name is required"}), 400

            # Check if this is a collection deletion request
            if data.get("confirm_coll_delete") is True:
                # Delete collection endpoint
                delete_url = f"http://{components.qdrant_host}:8716/collections/{collection_name}"

                response = requests.delete(
                    delete_url,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                if response.status_code == 200:
                    logger.info(f"Successfully deleted collection: {collection_name}")
                    return (
                        jsonify(
                            {
                                "status": "success",
                                "message": f"Successfully deleted collection {collection_name}",
                                "deleted_collection": collection_name,
                            }
                        ),
                        200,
                    )
                else:
                    error_msg = f"Collection delete operation failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), response.status_code

            # If not deleting collection, we need a qdrant_id for point deletion
            qdrant_id = data.get("qdrant_id")
            if not qdrant_id:
                return (
                    jsonify(
                        {
                            "error": "qdrant_id is required for point deletion. If you want to delete the entire collection, set confirm_coll_delete to true"
                        }
                    ),
                    400,
                )

            # Point deletion logic
            delete_url = f"http://{components.vector_store.host}:8716/collections/{collection_name}/points"
            delete_payload = {
                "points": [
                    {
                        "id": qdrant_id,
                        "vector": [0.0] * 1024,  # Required for PUT
                        "_delete": True,
                    }
                ]
            }

            response = requests.put(
                delete_url,
                json=delete_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"Successfully deleted point with Qdrant ID: {qdrant_id}")
                return (
                    jsonify(
                        {
                            "status": "success",
                            "message": f"Successfully deleted point {qdrant_id}",
                            "deleted_id": qdrant_id,
                        }
                    ),
                    200,
                )
            else:
                error_msg = f"Delete operation failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), response.status_code

        except Exception as e:
            logger.error(f"Error processing delete request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/status/<task_id>", methods=["GET"])
    @requires_api_key
    async def get_task_status(task_id):
        """Get the status of a document processing task."""
        try:
            status = await Components.task_tracker.get_status(task_id)
            return jsonify(status), 200
        except Exception as e:
            logger.error(f"Error retrieving task status: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app

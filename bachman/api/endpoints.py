"""
This module defines the endpoints for the Bachman API.
"""

import logging
import asyncio
import json
import uuid
import requests
from flask import Flask, request, jsonify
from langchain.schema import Document

from bachman.processors.chunking import ChunkingConfig

# from bachman.processors.document_types import DocumentType
from bachman.api.middleware import requires_api_key
from bachman.core.components import Components
from bachman.utils.format_results import format_analysis_results

logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.route("/bachman/analyze", methods=["POST"])
    @requires_api_key
    def analyze():
        """Text analysis endpoint."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Extract and validate collection name
            collection_name = data.get("collection_name")
            if not collection_name:
                return jsonify({"error": "collection_name is required"}), 400

            logger.info(
                f"Processing analysis request for ticker: {data.get('ticker', 'UNKNOWN')} in collection: {collection_name}"
            )

            # Get text from request
            text = data.get("text")
            if not text:
                return jsonify({"error": "No text provided"}), 400

            # Ensure collection exists before processing
            Components.vector_store.ensure_collection(collection_name)

            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "ticker": data.get("ticker", "UNKNOWN"),
                    "type": data.get("analysis_type", "sentiment"),
                    "source": data.get("load_type", "live"),
                    "id": str(uuid.uuid4()),
                },
            )

            # Analyze sentiment first
            sentiment_result = Components.sentiment_analyzer.analyze_text([doc])

            # Format results using the utility function
            format_analysis_results(
                analysis_type=doc.metadata["type"],
                result=sentiment_result,
                metadata=doc.metadata,
            )

            # Store document with metadata in the specified collection
            storage_result = Components.vector_store.store_vectors(
                collection_name=collection_name,
                texts=[text],
                metadatas=[doc.metadata],
            )
            logger.info(
                f"Successfully stored document in collection: {collection_name}"
            )

            return (
                jsonify(
                    {"sentiment_analysis": sentiment_result, "storage": storage_result}
                ),
                200,
            )

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/bachman/process_text", methods=["POST"])
    @requires_api_key
    def process_text():
        """Text processing endpoint without sentiment analysis."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

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
                Components.text_processor.process_text(
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
            data = request.get_json()

            # Extract required parameters
            file_path = data.get("file_path")
            collection_name = data.get("collection_name")
            doc_type = data.get("doc_type")
            process_sentiment = data.get("process_sentiment", False)
            metadata = data.get("metadata", {})

            if not file_path or not collection_name:
                return (
                    jsonify(
                        {
                            "error": "Missing required parameters: file_path and collection_name"
                        }
                    ),
                    400,
                )

            # Process file using asyncio.run() like the process_text endpoint
            result = asyncio.run(
                Components.file_processor.process_file(
                    file_path=file_path,
                    collection_name=collection_name,
                    doc_type=doc_type,
                    process_sentiment=process_sentiment,
                    metadata=metadata,
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
            if not data:
                return jsonify({"error": "No data provided"}), 400

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
            search_url = f"http://{Components.vector_store.host}:8716/collections/{collection_name}/search"
            search_payload = {
                "query_vector": [0] * 1024,
                "filter": {"must": filter_conditions},
                "limit": 100,
            }

            response = requests.post(
                search_url,
                json=search_payload,
                timeout=10,
            )

            if response.status_code == 200:
                raw_results = response.json()

                # Format the results
                formatted_results = []
                for hit in raw_results:
                    metadata = hit.get("payload", {})
                    formatted_result = {
                        "id": hit.get("id"),
                        "metadata": {
                            "doc_id": metadata.get("doc_id"),
                            "ticker": metadata.get("ticker"),
                            "source": metadata.get("source"),
                            "doc_type": metadata.get("doc_type"),
                            "timestamp": metadata.get("timestamp"),
                        },
                        "chunks": metadata.get("chunks", []),
                        "text_preview": metadata.get("text", "")[:100] + "..."
                        if metadata.get("text")
                        else None,
                    }
                    formatted_results.append(formatted_result)

                # Log just the formatted results
                logger.info("\nRESULTS")
                logger.info(json.dumps(formatted_results, indent=2))
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
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            collection_name = data.get("collection_name")
            if not collection_name:
                return jsonify({"error": "collection_name is required"}), 400

            # Check if this is a collection deletion request
            if data.get("confirm_coll_delete") is True:
                # Delete collection endpoint
                delete_url = f"http://{Components.vector_store.host}:8716/collections/{collection_name}"

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
            delete_url = f"http://{Components.vector_store.host}:8716/collections/{collection_name}/points"
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

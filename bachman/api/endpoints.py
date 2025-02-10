"""
This module defines the endpoints for the Bachman API.
"""

import logging
import asyncio
import json

# import uuid
# from datetime import datetime
import requests

# import os
from flask import Flask, request, jsonify

# from langchain.schema import Document

# from bachman.processors.chunking import ChunkingConfig

# from bachman.processors.document_types import DocumentType
from bachman.processors.analysis_processor import AnalysisProcessor
from bachman.api.middleware import requires_api_key
from bachman.core.components import Components

# from bachman.utils.format_results import format_analysis_results
# from bachman.config.prompt_config import prompt_config
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
            metadata = data.get("metadata")
            doc_id = metadata.get("doc_id")
            inf_specs = data.get("inference")
            inference_type = inf_specs.get("i_type")
            entity_type = inf_specs.get("e_type")
            inference_model = inf_specs.get("model")
            inference_provider = inf_specs.get("provider", "groq")

            # Create new components instance for this request
            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()
            components.vector_store.create_collection(collection_name=collection_name)

            embedding_configs = components.load_embedding_config()
            emb_cfg = embedding_configs[inference_provider]["models"][inference_model]

            if not collection_name or not doc_id:
                return (
                    jsonify({"error": "collection_name and doc_id are required"}),
                    400,
                )

            logger.info(
                f"Analyzing document with ID {doc_id} from collection {collection_name}"
            )

            # Initialize analysis processor
            analysis_processor = AnalysisProcessor(components.vector_store)

            # Process document and generate prompt
            response_list = analysis_processor.prepare_analysis(
                doc_id=doc_id,
                collection_name=collection_name,
                llm_context_window=emb_cfg.get("context_window", 8000),
                inference_type=inference_type,
                entity_type=entity_type,
                inference_model=inference_model,
                inference_provider=inference_provider,
            )

            if not response_list:
                return jsonify({"error": "Document not found"}), 404

            # For now, just return the preview
            return response_list, 200

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

            metadata = data.get("metadata", {})
            doc_type = metadata.get("doc_type", "unspecified_content")
            skip_if_exists = data.get("skip_if_exists", True)
            doc_id = metadata.get("doc_id")
            collection_name = data.get("collection_name")

            # Create new components instance for this request
            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()
            components.vector_store.create_collection(collection_name=collection_name)

            # Check if the document already exists in the collection
            if skip_if_exists is True:
                docs_ct = 0
                response = requests.post(
                    f"http://{components.qdrant_host}:8716/collections/{collection_name}/search",
                    json={
                        "query_vector": [0] * 1024,
                        "filter": {
                            "must": [
                                {
                                    "key": "metadata.doc_id",
                                    "match": {"value": doc_id, "exact": True},
                                }
                            ]
                        },
                        "limit": 100,
                    },
                    timeout=10,
                )
                docs_ct = len(response.json())
                print(f"docs_ct: {docs_ct}")

                if docs_ct > 0:
                    result = {
                        "status": "skipped",
                        "reason": "document already exists",
                        "doc_id": doc_id,
                    }
                    logger.info("=" * 80)
                    logger.info("RESULT")
                    logger.info("=" * 80)
                    spacer = " " * 57
                    for k, v in result.items():
                        print(f"{spacer} {k}: {v}")
                    logger.info("=" * 80)
                    return jsonify(result), 200

            chunking_configs = components.load_chunking_config()
            chunk_cfg = chunking_configs.get(doc_type)

            # Extract and validate collection name
            collection_name = data.get("collection_name")
            if not collection_name:
                return jsonify({"error": "collection_name is required"}), 400

            # Get text from request
            text = data.get("text")
            if not text:
                return jsonify({"error": "No text provided"}), 400

            # Get doc_type specific chunking config
            doc_type = metadata.get("doc_type", "default")
            chunking_config = components.get_chunking_config(doc_type)
            logger.info(
                f"Using chunking config for doc_type '{doc_type}': {chunking_config}"
            )

            # Run the coroutine in the event loop
            result = asyncio.run(
                components.text_processor.process_text(
                    text=text,
                    collection_name=collection_name,
                    metadata=data.get("metadata"),
                    skip_if_exists=data.get("skip_if_exists", False),
                    chunking_config=chunk_cfg,
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
            metadata = data.get("metadata", {})
            doc_type = metadata.get("doc_type", "unspecified_content")
            temp_dir = data.get("temp_dir")
            cleanup = data.get("cleanup", True)
            skip_if_exists = data.get("skip_if_exists", True)
            doc_id = metadata.get("doc_id")

            print(f"This is the doc_id for reference: {doc_id}")

            # Create new components instance for this request
            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()
            components.vector_store.create_collection(collection_name=collection_name)

            # Check if the document already exists in the collection
            if skip_if_exists is True:
                docs_ct = 0
                response = requests.post(
                    f"http://{components.qdrant_host}:8716/collections/{collection_name}/search",
                    json={
                        "query_vector": [0] * 1024,
                        "filter": {
                            "must": [
                                {
                                    "key": "metadata.doc_id",
                                    "match": {"value": doc_id, "exact": True},
                                }
                            ]
                        },
                        "limit": 100,
                    },
                    timeout=10,
                )
                docs_ct = len(response.json())
                print(f"printing docs_ct: {docs_ct}")

                if docs_ct > 0:
                    result = {
                        "status": "skipped",
                        "reason": "document already exists",
                        "doc_id": doc_id,
                    }
                    logger.info("=" * 80)
                    logger.info("RESULT")
                    logger.info("=" * 80)
                    spacer = " " * 57
                    for k, v in result.items():
                        print(f"{spacer} {k}: {v}")
                    logger.info("=" * 80)
                    return jsonify(result), 200

            if not file_path or not collection_name:
                return jsonify({"error": "Missing required parameters"}), 400

            # Get doc_type specific chunking config
            chunking_configs = components.load_chunking_config()
            chunk_cfg = chunking_configs.get(doc_type)
            logger.info(f"Using chunking config for doc_type '{doc_type}': {chunk_cfg}")

            result = asyncio.run(
                components.file_processor.process_file(
                    file_path=file_path,
                    collection_name=collection_name,
                    metadata=metadata,
                    chunking_config=chunk_cfg,
                    temp_dir=temp_dir,
                    cleanup=cleanup,
                    skip_if_exists=skip_if_exists,
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

            # Get qdrant_id if it exists
            qdrant_id = data.get("qdrant_id")
            # print(f"qdrant_id: {qdrant_id}")

            # Prepare filter conditions
            if qdrant_id:
                filter_conditions = [
                    {"key": "qdrant_id", "match": {"value": qdrant_id}}
                ]
                # print(f"filter_conditions: {filter_conditions}")
            else:
                # Get query parameters
                metadata_filter = data.get("metadata", {})
                if not metadata_filter:
                    return jsonify({"error": "Query metadata is required"}), 400

                filter_conditions = []
                for key, value in metadata_filter.items():
                    key = f"metadata.{key}"
                    filter_conditions.append({"key": key, "match": {"value": value}})

            # print(f"filter_conditions: {filter_conditions}")

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
                # print(f"docs: {docs}")

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

                if qdrant_id:
                    for point in docs["points"]:
                        point.pop("vector")
                        formatted_results.append(point)
                        logger.info(json.dumps(point, indent=2))
                else:
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
        """Delete endpoint to remove either specific points or an entire collection based on metadata."""
        try:
            # Create new components instance for this request
            components = Components(qdrant_host=QDRANT_HOST)
            components.initialize_components()

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

            # If not deleting collection, we need either a qdrant_id or metadata
            qdrant_id = data.get("qdrant_id")
            metadata = data.get("metadata")

            if not qdrant_id and not metadata:
                return (
                    jsonify(
                        {
                            "error": "Either qdrant_id or metadata is required for point deletion. "
                            "If you want to delete the entire collection, set confirm_coll_delete to true"
                        }
                    ),
                    400,
                )

            # Point deletion logic
            if qdrant_id:
                # Single point deletion using PUT endpoint
                delete_url = f"http://{components.qdrant_host}:8716/collections/{collection_name}/points"
                delete_payload = {
                    "points": [
                        {
                            "id": qdrant_id,
                            "_delete": True,
                        }
                    ]
                }
                method = "PUT"
            else:
                # Metadata-based deletion using POST endpoint
                delete_url = f"http://{components.qdrant_host}:8716/collections/{collection_name}/points/delete"
                delete_payload = {
                    "filter": {
                        "must": [
                            {"key": f"metadata.{key}", "match": {"value": value}}
                            for key, value in metadata.items()
                        ]
                    }
                }
                method = "POST"

            # Add debug logging
            logger.info(f"Sending {method} request to: {delete_url}")
            logger.info(f"With payload: {json.dumps(delete_payload, indent=2)}")

            print(f"delete_payload: {delete_payload}")

            response = requests.request(
                method=method,
                url=delete_url,
                json=delete_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            # Add response logging
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response body: {response.text}")

            if response.status_code == 200:
                logger.info("Successfully deleted points matching criteria")
                return (
                    jsonify(
                        {
                            "status": "success",
                            "message": "Successfully deleted matching points",
                            "filter": metadata
                            if metadata
                            else {"qdrant_id": qdrant_id},
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

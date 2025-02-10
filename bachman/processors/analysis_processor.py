"""
Analysis processor for handling document reconstruction and prompt generation.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from bachman.config.prompt_config import prompt_config
from bachman.models.llm import get_groq_llm

logger = logging.getLogger(__name__)


@dataclass
class DocumentContent:
    """Container for reconstructed document content."""

    content: str
    tables: List[Dict[str, Any]]
    doc_type: str
    ticker: str
    metadata: Dict[str, Any]


class AnalysisProcessor:
    """Handles document reconstruction and analysis preparation."""

    def __init__(self, vector_store):
        """Initialize with vector store connection."""
        self.vector_store = vector_store

    def prepare_analysis(
        self,
        doc_id: str,
        collection_name: str,
        llm_context_window: int = 8000,
        inference_type: str = None,
        entity_type: str = None,
        inference_model: str = None,
        inference_provider: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare document for analysis by reconstructing content and generating prompt.
        """
        try:
            # Use scroll method to fetch documents with matching doc_id
            docs_response = self.vector_store.client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [{"key": "metadata.doc_id", "match": {"value": doc_id}}]
                },
                with_payload=True,
                limit=100,  # Adjust based on your needs
            )

            if not docs_response[0]:
                logger.warning(
                    f"No documents found for doc_id: {doc_id} in collection: {collection_name}"
                )
                return None

            # First element contains the points
            llm_context_window = 4000
            docs = docs_response[0]
            metadata = docs[0].payload["metadata"].copy()
            metadata.pop("chunk_index")
            doc_type = metadata["doc_type"]
            ticker = metadata["ticker"]
            total_chunks = metadata["total_chunks"]

            # Check chunk total against total number of docs
            if total_chunks != len(docs):
                logger.warning(f"Total chunks mismatch: {total_chunks} != {len(docs)}")

            base_prompt_len = len(
                prompt_config(
                    doc_type=doc_type,
                    ticker=ticker,
                    chunks_text="",
                    entity_type=entity_type,
                    inference_type=inference_type,
                )
            )

            start_chunk = 0
            end_chunk = 0
            total_chunks = len(docs)
            chunk_set_index = 0
            llm_response_structure = {}
            llm_response_structure["doc_type"] = doc_type
            llm_response_structure["metadata"] = metadata
            llm_response_structure["collection_name"] = collection_name
            llm_response_structure["chunk_set"] = {}
            while start_chunk < total_chunks:
                if doc_type == "financial_document":
                    recon_src_content, end_chunk = self._reconstruct_financial_document(
                        docs,
                        total_chunks=total_chunks,
                        start_chunk=start_chunk,
                        llm_context_window=llm_context_window,
                        base_prompt_len=base_prompt_len,
                        metadata=metadata,
                    )
                else:
                    # Sort docs by chunk_index
                    docs.sort(key=lambda x: x.payload["metadata"]["chunk_index"])

                    recon_src_content, end_chunk = self._reconstruct_generic_document(
                        docs,
                        total_chunks=total_chunks,
                        start_chunk=start_chunk,
                        llm_context_window=llm_context_window,
                        base_prompt_len=base_prompt_len,
                        metadata=metadata,
                    )

                # For preview purposes, just show first segment
                content = recon_src_content.content

                # Generate prompt for preview
                analysis_prompt = prompt_config(
                    doc_type=doc_type,
                    ticker=ticker,
                    entity_type=entity_type,
                    chunks_text=content,
                    inference_type=inference_type,
                )

                llm_response_structure["chunk_set"][chunk_set_index] = {}
                llm_response_structure["chunk_set"][chunk_set_index][
                    "chunk_begin"
                ] = start_chunk
                llm_response_structure["chunk_set"][chunk_set_index][
                    "chunk_end"
                ] = end_chunk
                llm_response_structure["chunk_set"][chunk_set_index][
                    "prompt"
                ] = analysis_prompt

                # Make LLM request here
                logger.info(f"Generated prompt for doc_id {doc_id}:")
                logger.info(analysis_prompt)

                # Use the Groq client from llm.py
                groq_client = get_groq_llm()
                response = groq_client.invoke(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    model=inference_model,
                )
                response_content = response.content
                llm_response_structure["chunk_set"][chunk_set_index][
                    "response"
                ] = response_content
                # Log response for debugging
                logger.info(f"Groq API response: {response_content}")

                start_chunk = end_chunk + 1
                chunk_set_index += 1

            return llm_response_structure

        except Exception as e:
            logger.error(f"Error preparing analysis: {str(e)}")
            raise

    def _reconstruct_financial_document(
        self,
        docs: List[Any],
        start_chunk: int = 0,
        total_chunks: int = 0,
        llm_context_window: int = 8000,
        base_prompt_len: int = 0,
        metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Reconstruct financial document page by page, respecting context window limits."""
        MAX_TOKENS = 8000  # Groq's context window limit
        segments = []
        current_segment = {"text": "", "tables": [], "pages": [], "token_count": 0}

        # First, organize all content by page
        pages: Dict[int, Dict[str, Any]] = {}
        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            content_type = doc.metadata.get("content_type", "text")

            if page_num not in pages:
                pages[page_num] = {"text": [], "tables": []}

            if content_type == "table":
                try:
                    table_data = json.loads(doc.page_content)
                    table_index = doc.metadata.get("table_index", 0)
                    pages[page_num]["tables"].append(
                        {"index": table_index, "data": table_data}
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse table data on page {page_num}")
            else:
                pages[page_num]["text"].append(doc.page_content)

        # Process pages sequentially
        for page_num in sorted(pages.keys()):
            page = pages[page_num]
            page_text = " ".join(page["text"])

            # Rough token estimation - should be replaced with proper token counter
            estimated_tokens = len(page_text.split())

            if current_segment["token_count"] + estimated_tokens > MAX_TOKENS:
                # Current segment is full, store it and start new segment
                if current_segment["text"]:
                    segments.append(
                        {
                            "content": current_segment["text"],
                            "tables": current_segment["tables"],
                            "metadata": {
                                "pages": current_segment["pages"],
                                "token_count": current_segment["token_count"],
                                "doc_type": docs[0].metadata.get("doc_type"),
                                "ticker": docs[0].metadata.get("ticker"),
                                "segment_index": len(segments),
                            },
                        }
                    )

                # Start new segment with current page
                current_segment = {
                    "text": f"Page {page_num}:\n{page_text}\n",
                    "tables": page["tables"],
                    "pages": [page_num],
                    "token_count": estimated_tokens,
                }
            else:
                # Add to current segment
                current_segment["text"] += f"Page {page_num}:\n{page_text}\n"
                current_segment["tables"].extend(page["tables"])
                current_segment["pages"].append(page_num)
                current_segment["token_count"] += estimated_tokens

        # Add final segment if it exists
        if current_segment["text"]:
            segments.append(
                {
                    "content": current_segment["text"],
                    "tables": current_segment["tables"],
                    "metadata": {
                        "pages": current_segment["pages"],
                        "token_count": current_segment["token_count"],
                        "doc_type": docs[0].metadata.get("doc_type"),
                        "ticker": docs[0].metadata.get("ticker"),
                        "segment_index": len(segments),
                    },
                }
            )

        logger.info(f"Document split into {len(segments)} segments")
        for segment in segments:
            logger.info(
                f"Segment {segment['metadata']['segment_index']}: "
                f"Pages {segment['metadata']['pages']}, "
                f"Token count: {segment['metadata']['token_count']}"
            )

        dc = "test"
        end_chunk = 0

        return dc, end_chunk

    def _reconstruct_generic_document(
        self,
        sorted_docs: List[Any],
        start_chunk: int = 0,
        total_chunks: int = 0,
        llm_context_window: int = 8000,
        base_prompt_len: int = 0,
        metadata: Dict[str, Any] = None,
    ) -> DocumentContent:
        """Reconstruct generic document using chunk indices."""

        docs_to_process = []
        for chunk_index in range(start_chunk, total_chunks):
            loop_content = sorted_docs[chunk_index].payload["text"]
            # loop_content_len = len(loop_content)
            docs_to_process.append(loop_content)
            l_chk = " ".join(docs_to_process)
            p_len = len(l_chk) + base_prompt_len
            if p_len > llm_context_window:
                if base_prompt_len > llm_context_window:
                    logger.warning(
                        f"Base prompt length exceeds context window: {base_prompt_len} > {llm_context_window}"
                    )
                    return None
                elif chunk_index == 0:
                    logger.warning(
                        "First chunk + base prompt exceeds context window. Need smaller chunks or base prompt."
                    )
                    return None
                docs_to_process.pop()
                break
            else:
                end_chunk = chunk_index

        combined_text = " ".join(docs_to_process)

        dc = DocumentContent(
            content=combined_text,
            tables=[],  # Generic documents don't have tables
            doc_type=sorted_docs[0].payload["metadata"]["doc_type"],
            ticker=sorted_docs[0].payload["metadata"]["ticker"],
            metadata=sorted_docs[0].payload["metadata"],
        )

        return dc, end_chunk

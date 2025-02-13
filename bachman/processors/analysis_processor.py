"""
Analysis processor for handling document reconstruction and prompt generation.
"""
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import pandas as pd
import numpy as np
from quantum_trade_utilities.data.load_credentials import load_credentials
from quantum_trade_utilities.core.get_path import get_path
from openai import OpenAI

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
        creds_path = get_path("creds")
        vllm_details = load_credentials(creds_path, "vllm_ds")
        self.vllm_host = vllm_details[0]
        self.vllm_port = vllm_details[1]

    def prepare_analysis(
        self,
        doc_id: str,
        collection_name: str,
        llm_context_window: int = 8000,
        inference_type: str = None,
        entity_type: str = None,
        inference_model: str = None,
        inference_provider: str = None,
        paginated_doctypes: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare document for analysis by reconstructing content and generating prompt.
        """
        try:
            response = requests.post(
                f"http://192.168.1.10:8716/collections/{collection_name}/search",
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
                    "limit": 1000,
                },
                timeout=10,
            )

            docs = response.json()

            if len(docs) == 0:
                logger.warning(
                    f"No documents found for doc_id: {doc_id} in collection: {collection_name}"
                )
                return None

            # First element contains the points
            # llm_context_window = 7500
            metadata = docs[0]["payload"]["metadata"].copy()
            doc_type = metadata["doc_type"]
            ticker = metadata["ticker"]

            # If paginated, grab some additional metadata
            doc_list = []
            if doc_type in paginated_doctypes:
                total_pages = 0
                total_tables = 0
                total_chunks = 0
                for doc in docs:
                    try:
                        total_pages = max(
                            total_pages, doc["payload"]["metadata"]["page"]
                        )
                        doc_list.append(doc)
                    except:
                        continue

                for doc in docs:
                    try:
                        tid = int(doc["payload"]["metadata"]["table_index"]) / int(
                            doc["payload"]["metadata"]["table_index"]
                        )
                        total_tables += tid
                    except:
                        continue

                for doc in docs:
                    try:
                        total_chunks = doc["payload"]["metadata"]["total_chunks"]
                    except:
                        continue

                # Check chunk total against total number of docs
                if total_tables + total_chunks != len(docs):
                    logger.warning(
                        f"Total chunks mismatch: {total_tables + total_chunks} != {len(docs)}"
                    )

            else:
                metadata.pop("chunk_index")
                total_chunks = metadata["total_chunks"]

                # Check chunk total against total number of docs
                if total_chunks != len(docs):
                    logger.warning(
                        f"Total chunks mismatch: {total_chunks} != {len(docs)}"
                    )

            base_prompt_len = len(
                prompt_config(
                    doc_type=doc_type,
                    ticker=ticker,
                    chunks_text="",
                    entity_type=entity_type,
                    inference_type=inference_type,
                )
            )

            llm_response_structure = {}
            llm_response_structure["doc_type"] = doc_type
            llm_response_structure["metadata"] = metadata
            llm_response_structure["collection_name"] = collection_name
            llm_response_structure["inference"] = {
                "inference_type": inference_type,
                "entity_type": entity_type,
                "inference_model": inference_model,
                "inference_provider": inference_provider,
            }
            llm_response_structure["doc_grp"] = {}
            doc_grp_index = 0

            if doc_type == "financial_document":
                recon_src_content_df = self._reconstruct_financial_document(
                    doc_list,
                    llm_context_window=llm_context_window,
                    base_prompt_len=base_prompt_len,
                )

                doc_grp_max = recon_src_content_df["doc_grp"].max()
                for doc_grp in range(0, doc_grp_max + 1):
                    # for doc_grp in range(0, 4):
                    loop_df = recon_src_content_df[
                        recon_src_content_df["doc_grp"] == doc_grp
                    ]

                    table_index_begin = loop_df["metadata.table_index"].min()
                    table_index_end = loop_df["metadata.table_index"].max()
                    start_chunk = loop_df["metadata.chunk_index"].min()
                    end_chunk = loop_df["metadata.chunk_index"].max()
                    if table_index_begin is np.nan:
                        table_index_begin = 999999
                    if table_index_end is np.nan:
                        table_index_end = 999999
                    if start_chunk is np.nan:
                        start_chunk = 999999
                    if end_chunk is np.nan:
                        end_chunk = 999999
                    # Concatenate the page_content for the current doc_grp
                    content = loop_df["page_content"].str.cat(sep=" ")

                    # Generate prompt for preview
                    analysis_prompt = prompt_config(
                        doc_type=doc_type,
                        ticker=ticker,
                        entity_type=entity_type,
                        chunks_text=content,
                        inference_type=inference_type,
                    )

                    llm_response_structure["doc_grp"][doc_grp_index] = {}
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "chunk_begin"
                    ] = start_chunk
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "chunk_end"
                    ] = end_chunk
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "table_index_begin"
                    ] = table_index_begin
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "table_index_end"
                    ] = table_index_end
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "prompt"
                    ] = analysis_prompt

                    # Make LLM request here
                    logger.info(f"Generated prompt for doc_id {doc_id}:")
                    logger.info(analysis_prompt)

                    # Use specified inference_provider
                    response = self._get_inference_response(
                        analysis_prompt,
                        inference_provider,
                        inference_model,
                    )

                    cleaned_response = json.loads(response)
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "response"
                    ] = cleaned_response
                    # Log response for debugging
                    logger.info(
                        f"{inference_provider} API response: {cleaned_response}"
                    )

                    doc_grp_index += 1

            else:
                start_chunk = 0
                end_chunk = 0
                while start_chunk < total_chunks:
                    # Sort docs by chunk_index
                    docs.sort(key=lambda x: x["payload"]["metadata"]["chunk_index"])

                    recon_src_content, end_chunk = self._reconstruct_generic_document(
                        docs,
                        total_chunks=total_chunks,
                        start_chunk=start_chunk,
                        llm_context_window=llm_context_window,
                        base_prompt_len=base_prompt_len,
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

                    llm_response_structure["doc_grp"][doc_grp_index] = {}
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "chunk_begin"
                    ] = start_chunk
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "chunk_end"
                    ] = end_chunk
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "table_index_begin"
                    ] = 0
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "table_index_end"
                    ] = 0
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "prompt"
                    ] = analysis_prompt

                    # Make LLM request here
                    logger.info(f"Generated prompt for doc_id {doc_id}:")
                    logger.info(analysis_prompt)

                    # Use specified inference_provider
                    response = self._get_inference_response(
                        analysis_prompt,
                        inference_provider,
                        inference_model,
                    )

                    cleaned_response = json.loads(response)
                    llm_response_structure["doc_grp"][doc_grp_index][
                        "response"
                    ] = cleaned_response
                    # Log response for debugging
                    logger.info(
                        f"{inference_provider} API response: {cleaned_response}"
                    )

                    start_chunk = end_chunk + 1
                    doc_grp_index += 1

            return llm_response_structure

        except Exception as e:
            logger.error(f"Error preparing analysis: {str(e)}")
            raise

    def _reconstruct_financial_document(
        self,
        doc_list: List[Any],
        llm_context_window: int = 8000,
        base_prompt_len: int = 0,
    ) -> List[Dict[str, Any]]:
        """Reconstruct financial document page by page, respecting context window limits."""

        df_tmp = pd.DataFrame(doc_list)
        df_tmp["metadata"] = df_tmp["payload"].apply(lambda x: x["metadata"])
        # Flatten metadata
        metadata_df = pd.json_normalize(df_tmp["metadata"])
        content_df = pd.json_normalize(df_tmp["payload"])

        # Combine everything
        result_df = pd.concat([df_tmp, content_df, metadata_df], axis=1).drop(
            ["payload", "metadata"], axis=1
        )

        # Sort the data by page, then text chunks, then tables
        sorted_df = result_df.sort_values(by=["page", "chunk_index", "table_index"])

        # Add length of chunked content
        sorted_df["content_length"] = sorted_df["page_content"].str.len()

        # Apply cumulative sum with reset and group ct.
        sorted_df = self._create_reset_cumsum(
            sorted_df, "content_length", llm_context_window, base_prompt_len
        )

        return sorted_df

    def _reconstruct_generic_document(
        self,
        sorted_docs: List[Any],
        start_chunk: int = 0,
        total_chunks: int = 0,
        llm_context_window: int = 8000,
        base_prompt_len: int = 0,
    ) -> DocumentContent:
        """Reconstruct generic document using chunk indices."""

        docs_to_process = []
        for chunk_index in range(start_chunk, total_chunks):
            loop_content = sorted_docs[chunk_index]["payload"]["text"]
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
            doc_type=sorted_docs[0]["payload"]["metadata"]["doc_type"],
            ticker=sorted_docs[0]["payload"]["metadata"]["ticker"],
            metadata=sorted_docs[0]["payload"]["metadata"],
        )

        return dc, end_chunk

    def _create_reset_cumsum(
        self, df, value_column, llm_context_window, base_prompt_len
    ):
        # Create a copy to avoid modifying the original
        # Reset index to avoid issues with cumulative sum
        df = df.copy().reset_index(drop=False)

        # Initialize the new columns
        df["cumsum"] = 0
        df["doc_grp"] = 0

        cumsum = 0
        doc_grp = 0

        # Iterate through the rows in their current order
        for idx in df.index:
            value = df.at[idx, value_column]
            cumsum += value
            if cumsum + base_prompt_len > llm_context_window:
                if base_prompt_len > llm_context_window:
                    logger.warning(
                        f"Base prompt length exceeds context window: {base_prompt_len} > {llm_context_window}"
                    )
                    return None
                elif idx == 0:
                    logger.warning(
                        f"First chunk:{cumsum} + base prompt:{base_prompt_len} exceeds context window. Need smaller chunks or base prompt."
                    )
                    return None
                cumsum = value  # reset to current value
                doc_grp += 1  # increment group
            df.at[idx, "cumsum"] = cumsum
            df.at[idx, "doc_grp"] = doc_grp

        return df

    def _get_inference_response(
        self, analysis_prompt, inference_provider, inference_model
    ):
        response_content = None
        if inference_provider == "groq":
            groq_client = get_groq_llm()
            response = groq_client.invoke(
                messages=[{"role": "user", "content": analysis_prompt}],
                model=inference_model,
            )
            response_content = response.content

        elif inference_provider == "vllm":
            # Modify OpenAI's API key and API base to use VLLM's API server.
            openai_api_key = "EMPTY"
            openai_api_base = f"http://{self.vllm_host}:{self.vllm_port}/v1"
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )

            models = client.models.list()
            model = models.data[0].id

            messages = [
                {
                    "role": "user",
                    "content": "Return ONLY a JSON response with no explanation or thinking process. "
                    + analysis_prompt,
                }
            ]
            response = client.chat.completions.create(model=model, messages=messages)

            raw_response = response.choices[0].message.content
            response_content = self._extract_vllm_json(raw_response)

        return response_content

    def _extract_vllm_json(self, text):
        # Convert to lowercase for case-insensitive split
        lower_text = text.lower()
        parts = lower_text.split("</think>")

        json_part = parts[1]
        # Find first { and last }
        start = json_part.find("{")
        end = json_part.rfind("}") + 1
        if start >= 0 and end > 0:
            # Use these indices on the original text
            original_start = text.lower().find("{")
            original_end = text.lower().rfind("}") + 1
            return text[original_start:original_end]
        return None

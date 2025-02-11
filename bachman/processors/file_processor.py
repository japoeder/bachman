"""File processing module for handling different document types."""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import subprocess

# import uuid
import shutil
import tabula
import pymupdf
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# from bachman.processors.chunking import get_chunking_config


# from langchain.document_loaders import PyPDFLoader, TextLoader
# from bachman.processors.chunking import ChunkingConfig

# from bachman.processors.types import ChunkInfo, HashInfo, ProcessingResult
from bachman.processors.document_types import DocumentType

# from bachman.core.interfaces import TaskTracker
from bachman.processors.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class ProcessingStatus:
    """Status tracking for document processing."""

    def __init__(self, doc_id: str, chunking_config: dict):
        self.doc_id = doc_id
        self.status = "pending"
        self.components = {
            "text": {"status": "pending", "pages": []},
            "tables": {"status": "pending", "pages": []},
            "metadata": {"status": "pending"},
        }
        self.chunking_config = chunking_config
        self.errors = []
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.default_chunking_config = {
            "strategy": "recursive",
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "separators": ["\n\n", "\n", ". ", " ", ""],
            "min_chunk_size": 50,
        }

    def to_dict(self) -> Dict:
        """Convert status to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "status": self.status,
            "components": self.components,
            "errors": self.errors,
            "last_updated": self.last_updated,
        }

    def update(self, component: str, status: str, error: Optional[str] = None):
        """Update component status."""
        if component in self.components:
            self.components[component]["status"] = status
        if error:
            self.errors.append(error)
        self.last_updated = datetime.utcnow().isoformat()


class FileProcessor:
    """
    Handles processing of different file types with configurable strategies.

    Supports PDF and text files with customizable chunking and metadata options.
    Provides tracking of processed documents and their chunks.
    """

    def __init__(
        self,
        text_processor,
        # , task_tracker: TaskTracker
        vector_store: VectorStore,
    ):
        """Initialize with required dependencies."""
        self.text_processor = text_processor
        # self.task_tracker = task_tracker
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        self.has_java = self._check_java_dependencies()
        self._temp_files = set()  # Track all temp files created

    def _check_java_dependencies(self) -> bool:
        """Check if Java is installed and accessible."""
        try:
            result = subprocess.run(
                ["java", "-version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                self.logger.info("Java is available for table extraction")
                return True
            else:
                self.logger.warning(
                    "Java is not available. Table extraction will be disabled."
                )
                return False
        except Exception as e:
            self.logger.warning(
                f"Error checking Java: {str(e)}. Table extraction will be disabled."
            )
            return False

    def read_pdf(self, file_path: str):
        """Read text content from a PDF file and return text and word count."""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                word_count = len(text.split())
                return text, word_count
        except Exception as e:
            logger.error(f"Error reading PDF file: {str(e)}")
            raise

    def cleanup_temp_files(self, temp_dir: Optional[str] = None):
        """Clean up temporary files and directories."""
        try:
            if temp_dir and os.path.exists(temp_dir):
                self.logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                self._temp_files.clear()
            elif self._temp_files:
                self.logger.info("Cleaning up individual temporary files")
                for file_path in self._temp_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                self._temp_files.clear()

            self.logger.info("Temporary file cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    def validate_chunking(
        self, documents: List[Document], chunking_config: dict
    ) -> dict:
        """
        Validate that chunks match the expected configuration.

        Args:
            documents: List of processed documents
            chunking_config: Original chunking configuration

        Returns:
            dict: Validation statistics and any warnings
        """
        stats = {
            "config": chunking_config,
            "total_chunks": len(documents),
            "chunk_sizes": [],
            "avg_chunk_size": 0,
            "min_chunk_size": float("inf"),
            "max_chunk_size": 0,
            "warnings": [],
            "content_types": {},
        }

        for doc in documents:
            # Get chunk size in tokens and characters
            content_len = len(doc.page_content)
            stats["chunk_sizes"].append(content_len)

            # Track min/max
            stats["min_chunk_size"] = min(stats["min_chunk_size"], content_len)
            stats["max_chunk_size"] = max(stats["max_chunk_size"], content_len)

            # Track content types
            content_type = doc.metadata.get("content_type", "unknown")
            stats["content_types"][content_type] = (
                stats["content_types"].get(content_type, 0) + 1
            )

        # Calculate average
        if stats["chunk_sizes"]:
            stats["avg_chunk_size"] = sum(stats["chunk_sizes"]) / len(
                stats["chunk_sizes"]
            )

        # Add warnings for potential issues
        target_size = chunking_config.get("chunk_size", 512)
        if stats["max_chunk_size"] > target_size * 1.5:
            stats["warnings"].append(
                f"Some chunks exceed target size by 50%+ (max: {stats['max_chunk_size']} vs target: {target_size})"
            )
        if stats["min_chunk_size"] < target_size * 0.5:
            stats["warnings"].append(
                f"Some chunks are less than 50% of target size (min: {stats['min_chunk_size']} vs target: {target_size})"
            )

        logger.info("Chunking validation results:")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Average chunk size: {stats['avg_chunk_size']:.2f} characters")
        logger.info(
            f"Size range: {stats['min_chunk_size']} to {stats['max_chunk_size']} characters"
        )
        logger.info(f"Content types: {stats['content_types']}")
        if stats["warnings"]:
            for warning in stats["warnings"]:
                logger.warning(warning)

        return stats

    async def process_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict = None,
        skip_if_exists: bool = False,
        chunking_config: dict = None,
        temp_dir: Optional[str] = None,
        cleanup: bool = True,
    ):
        """Process a file and store its contents with proper chunking."""
        try:
            self.logger.info("Processing file: %s", file_path)
            doc_type = metadata.get("doc_type", "default")
            doc_id = metadata.get("doc_id")
            config = chunking_config or self.default_chunking_config
            metadata["collection_name"] = collection_name

            # Setup temporary file paths
            temp_files = {}
            if temp_dir:
                os.makedirs(temp_dir, exist_ok=True)
                base_name = Path(file_path).stem
                temp_files = {
                    "text": os.path.join(temp_dir, f"{base_name}_text.txt"),
                    "tables": os.path.join(temp_dir, f"{base_name}_tables.json"),
                    "chunks": os.path.join(temp_dir, f"{base_name}_chunks.json"),
                }
                # Track temp files for cleanup
                self._temp_files.update(temp_files.values())
                self.logger.info(f"Will save intermediate files to: {temp_dir}")

            # Process PDF and get both text and tables
            documents = []
            all_tables = []
            full_text = []
            page_boundaries = []  # Track where each page starts

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                current_position = 0

                for page_num in range(len(pdf_reader.pages)):
                    # Extract text
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    full_text.append(text)
                    # Track page boundary
                    page_boundaries.append(
                        {"start": current_position, "text": text, "page": page_num + 1}
                    )
                    current_position += len(text)

                    # Extract tables if Java is available
                    if self.has_java:
                        try:
                            tables = self._extract_tables(file_path, page_num)
                            for table_idx, table in enumerate(tables):
                                table_dict = table.to_dict()
                                # Convert any non-serializable objects to strings
                                table_dict = {k: str(v) for k, v in table_dict.items()}
                                all_tables.append(
                                    {
                                        "page": page_num + 1,
                                        "table_idx": table_idx,
                                        "content": table_dict,
                                    }
                                )
                                # Create a document for the table
                                table_text = json.dumps(table_dict, indent=2)
                                documents.append(
                                    Document(
                                        page_content=table_text,
                                        metadata={
                                            "source": file_path,
                                            "doc_id": doc_id,
                                            "doc_type": doc_type,
                                            "content_type": "table",
                                            "page": page_num + 1,
                                            "table_index": table_idx,
                                            **metadata,
                                        },
                                    )
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Error extracting tables from page {page_num + 1}: {str(e)}"
                            )

            # Save intermediate files if temp_dir is provided
            if temp_dir:
                # Save full text
                with open(temp_files["text"], "w", encoding="utf-8") as f:
                    f.write("\n".join(full_text))
                self.logger.info(f"Saved text to: {temp_files['text']}")

                # Save tables
                with open(temp_files["tables"], "w", encoding="utf-8") as f:
                    json.dump(all_tables, f, indent=2)
                self.logger.info(f"Saved tables to: {temp_files['tables']}")

            # Process text chunks with page tracking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size"),
                chunk_overlap=config.get("chunk_overlap"),
            )

            combined_text = "\n".join(full_text)
            text_chunks = text_splitter.split_text(combined_text)

            # Save chunks if temp_dir provided
            if temp_dir:
                with open(temp_files["chunks"], "w", encoding="utf-8") as f:
                    json.dump(text_chunks, f, indent=2)
                self.logger.info(f"Saved chunks to: {temp_files['chunks']}")

            # Find page number for each chunk
            for i, chunk in enumerate(text_chunks):
                chunk_start = combined_text.index(chunk)
                chunk_page = 1  # default to first page

                # Find which page this chunk starts on
                for boundary in page_boundaries:
                    if chunk_start >= boundary["start"]:
                        chunk_page = boundary["page"]
                    else:
                        break

                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "content_type": "text",
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            "page": chunk_page,  # Add page number to metadata
                            **metadata,
                        },
                    )
                )

            # Add validation with prominent logging
            self.logger.info("\n" + "=" * 50)
            self.logger.info("Starting Chunking Validation")
            self.logger.info("=" * 50)
            chunking_stats = self.validate_chunking(documents, chunking_config)

            # Store documents in vector store
            self.logger.info("Adding %d documents to vector store", len(documents))
            self.logger.info(f"About to add documents to collection: {collection_name}")
            self.vector_store.collection_name = collection_name
            await self.vector_store.add_documents(documents)

            result = {
                "status": "success",
                "doc_id": doc_id,
                "num_documents": len(documents),
                "num_text_chunks": len(text_chunks),
                "num_tables": len(all_tables),
                "file_path": file_path,
                "temp_files": temp_files if temp_dir else None,
                "doc_type": doc_type,
                "chunking_config": chunking_config,
                "chunking_stats": chunking_stats,
            }

            # Log the final results
            self.logger.info("\n" + "=" * 50)
            self.logger.info("Processing Results")
            self.logger.info("=" * 50)
            self.logger.info("Document ID: %s", doc_id)
            self.logger.info("Total documents: %d", len(documents))
            self.logger.info("Text chunks: %d", len(text_chunks))
            self.logger.info("Tables: %d", len(all_tables))
            self.logger.info("=" * 50 + "\n")

            # Cleanup if requested
            if cleanup:
                self.logger.info("Cleaning up temporary files")
                self.cleanup_temp_files(temp_dir)
                result["temp_files"] = None  # Files no longer exist

            return result

        except Exception as e:
            self.logger.error("Error processing file %s: %s", file_path, str(e))
            raise

    # def __del__(self):
    #     """Cleanup any remaining temporary files on object destruction."""
    #     try:
    #         print("TESTING Cleaning up temp files")
    #         #self.cleanup_temp_files()
    #     except Exception as e:
    #         self.logger.error(f"Error during final cleanup: {str(e)}")

    async def _process_text_components(
        self, file_path: str, status: ProcessingStatus
    ) -> dict:
        """Process text components of the file."""
        try:
            # Read text content and get word count
            text_content, word_count = self.read_pdf(file_path)

            status.update("text", "completed")
            return {"text": text_content, "word_count": word_count}

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            status.update("text", "failed", str(e))
            return {"text": "", "word_count": 0}

    async def _process_table_components(
        self, file_path: str, status: ProcessingStatus
    ) -> dict:
        """Process table components of the file."""
        try:
            # Read PDF document
            pdf = pymupdf.open(file_path)
            tables = []

            # Process each page
            for page_num in range(len(pdf)):
                page_tables = self._extract_tables_from_pdf(file_path, page_num)
                if page_tables:
                    tables.extend(page_tables)
                    status.components["tables"]["pages"].append(
                        {"page": page_num, "count": len(page_tables)}
                    )

            status.update("tables", "completed")
            return {"tables": tables}

        except Exception as e:
            logger.error(f"Error processing tables: {str(e)}")
            status.update("tables", "failed", str(e))
            return {"tables": []}

    def _apply_document_type_processing(
        self, result: Dict, document_type: DocumentType
    ) -> Dict:
        """Apply document-type-specific processing to extracted content."""
        if document_type == DocumentType.FINANCIAL_STATEMENT:
            return self._process_as_financial_statement(result)
        elif document_type == DocumentType.EARNINGS_CALL:
            return self._process_as_earnings_call(result)
        else:
            return self._process_as_generic(result)

    def _process_as_financial_statement(self, result: Dict) -> Dict:
        """Process content as a financial statement."""
        sections = {
            "balance_sheet": [],
            "income_statement": [],
            "cash_flow": [],
            "notes": [],
        }

        # Process tables and identify financial sections
        processed_tables = []
        for table in result["tables"]:
            table_type = self._identify_financial_table_type(table["content"])
            if table_type:
                processed_tables.append({**table, "type": table_type})
                sections[table_type].append(table["content"])

        return {
            "processed_text": result["text"],
            "tables": processed_tables,
            "sections": sections,
        }

    def _process_as_earnings_call(self, result: Dict) -> Dict:
        """Process content as an earnings call transcript."""
        sections = {
            "introduction": [],
            "prepared_remarks": [],
            "qa_session": [],
            "closing_remarks": [],
        }

        # Identify speakers and segment Q&A
        # content = result["raw_content"]
        # Add logic to segment earnings call sections

        return {
            "processed_text": result["text"],
            "tables": result["tables"],
            "sections": sections,
            "speakers": [],  # Add speaker identification
        }

    def _process_as_generic(self, result: Dict) -> Dict:
        """Process content as a generic document."""
        return {
            "processed_text": result["text"],
            "tables": result["tables"],
            "sections": {},
        }

    def _identify_financial_table_type(self, content: str) -> Optional[str]:
        """Identify type of financial table."""
        content_lower = content.lower()
        if "balance sheet" in content_lower:
            return "balance_sheet"
        elif "income statement" in content_lower:
            return "income_statement"
        elif "cash flow" in content_lower:
            return "cash_flow"
        return None

    def _extract_tables_from_pdf(self, file_path: str, page_num: int) -> List[Dict]:
        """Extract tables from PDF page using tabula."""
        tables = []
        try:
            pdf_tables = tabula.read_pdf(
                file_path,
                pages=page_num + 1,  # tabula uses 1-based page numbers
                multiple_tables=True,
                force_subprocess=True,
            )

            if pdf_tables:
                for table_idx, table in enumerate(pdf_tables):
                    # Convert table to string format
                    table_text = "\n".join(
                        [" | ".join(map(str, row)) for row in table.values]
                    )
                    tables.append(
                        {"page": page_num, "index": table_idx, "content": table_text}
                    )

        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {str(e)}")

        return tables

    def _extract_tables(self, file_path: str, page_num: int) -> List[Dict]:
        """Extract tables from a specific page of the PDF."""
        try:
            # Use tabula-py to extract tables
            tables = tabula.read_pdf(
                file_path,
                pages=page_num + 1,  # tabula uses 1-based page numbers
                multiple_tables=True,
                guess=True,
                lattice=True,  # For tables with lines/borders
                stream=True,  # For tables without clear borders
                pandas_options={"header": None},  # Don't assume first row is header
            )

            self.logger.info(f"Found {len(tables)} tables on page {page_num + 1}")
            return tables
        except Exception as e:
            self.logger.error(
                f"Error extracting tables from page {page_num + 1}: {str(e)}"
            )
            return []

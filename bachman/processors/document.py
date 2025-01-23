"""
Document processor module for handling text documents.

This module provides functionality for processing and managing text documents,
including chunking, metadata handling, and document transformations.
"""

# Standard library imports
import os
import logging
from typing import List, Dict, Any, Optional
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    A class for processing and managing text documents.

    This class provides methods for chunking text documents, handling metadata,
    and performing document transformations. It uses the RecursiveCharacterTextSplitter
    for intelligent text chunking based on content structure.

    Attributes:
        chunk_size (int): Maximum size of text chunks
        chunk_overlap (int): Number of characters to overlap between chunks
        text_splitter (RecursiveCharacterTextSplitter): The text splitter instance

    Example:
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
        chunks = processor.split_document(document)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """Initialize document processor with configurable chunking."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
        )

    def process_pdf(
        self, pdf_path: str, save_dir: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Process a PDF file and extract text and images.

        Args:
            pdf_path: Path to PDF file
            save_dir: Directory to save processed files
            metadata: Additional metadata to attach to documents
        """
        try:
            doc = pymupdf.open(pdf_path)
            filename = os.path.basename(pdf_path)

            # Setup directories
            subdirs = {
                "text": os.path.join(save_dir, "processed_text"),
                "images": os.path.join(save_dir, "processed_images"),
                "pages": os.path.join(save_dir, "processed_pages"),
                "tables": os.path.join(save_dir, "processed_tables"),
            }

            for dir_path in subdirs.values():
                os.makedirs(dir_path, exist_ok=True)

            results = {
                "text_files": [],
                "image_files": [],
                "page_files": [],
                "table_files": [],
            }

            # Process each page
            for page_num in tqdm(range(len(doc)), desc=f"Processing {filename}"):
                page = doc[page_num]

                # Extract and split text
                text = page.get_text()
                chunks = self.text_splitter.split_text(text)

                # Save text chunks
                for i, chunk in enumerate(chunks):
                    text_file = f"{filename}_text_{page_num}_{i}.txt"
                    text_path = os.path.join(subdirs["text"], text_file)

                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(chunk)
                    results["text_files"].append(text_path)

                # Extract and save images
                images = page.get_images()
                for idx, image in enumerate(images):
                    try:
                        xref = image[0]
                        pix = pymupdf.Pixmap(doc, xref)
                        image_file = f"{filename}_image_{page_num}_{idx}_{xref}.png"
                        image_path = os.path.join(subdirs["images"], image_file)
                        pix.save(image_path)
                        results["image_files"].append(image_path)
                    except Exception as e:
                        logger.warning(
                            f"Error saving image {idx} from page {page_num}: {str(e)}"
                        )

                # Save page as image
                try:
                    pix = page.get_pixmap()
                    page_file = f"page_{page_num:03d}.png"
                    page_path = os.path.join(subdirs["pages"], page_file)
                    pix.save(page_path)
                    results["page_files"].append(page_path)
                except Exception as e:
                    logger.warning(f"Error saving page {page_num} as image: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def load_processed_documents(
        self, base_path: str, ticker: str, doc_type: str = "text", batch_size: int = 100
    ) -> List[Document]:
        """
        Load processed documents with batching support.

        Args:
            base_path: Base directory path
            ticker: Stock ticker symbol
            doc_type: Type of document to load ('text' or 'tables')
            batch_size: Number of documents to load at once
        """
        try:
            path = f"{base_path}/{ticker}/annual_reports/processed_{doc_type}"
            if not os.path.exists(path):
                os.makedirs(path)

            loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)

            # Load and process documents in batches
            processed_docs = []
            raw_docs = loader.load()

            for i in range(0, len(raw_docs), batch_size):
                batch = raw_docs[i : i + batch_size]

                for doc in batch:
                    chunks = self.text_splitter.split_text(doc.page_content)

                    for chunk in chunks:
                        processed_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": doc.metadata["source"],
                                    "doc_type": doc_type,
                                    "ticker": ticker,
                                    "chunk_size": self.chunk_size,
                                    "chunk_overlap": self.chunk_overlap,
                                },
                            )
                        )

            logger.info(f"Loaded {len(processed_docs)} documents for {ticker}")
            return processed_docs

        except Exception as e:
            logger.error(f"Error loading documents for {ticker}: {str(e)}")
            raise

    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get detailed statistics about processed documents."""
        try:
            chunk_lengths = [len(doc.page_content) for doc in documents]
            doc_types = set(doc.metadata.get("doc_type") for doc in documents)
            tickers = set(doc.metadata.get("ticker") for doc in documents)

            return {
                "total_documents": len(documents),
                "average_chunk_length": sum(chunk_lengths) / len(chunk_lengths)
                if chunk_lengths
                else 0,
                "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                "doc_types": list(doc_types),
                "tickers": list(tickers),
                "chunk_size": documents[0].metadata.get("chunk_size")
                if documents
                else None,
                "chunk_overlap": documents[0].metadata.get("chunk_overlap")
                if documents
                else None,
            }

        except Exception as e:
            logger.error(f"Error calculating document stats: {str(e)}")
            raise

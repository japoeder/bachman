"""
Document loader for processing and loading documents into a vector store.
"""

import os
import logging
from typing import List, Dict, Any
import pymupdf
import tabula
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


class DocumentLoader:
    """
    Document loader for processing and loading documents into a vector store.
    """

    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize document loader.

        Args:
            raw_dir: Directory containing raw PDF files
            processed_dir: Directory to store processed files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_pdf(self, filepath: str, ticker: str, doc_type: str) -> None:
        """
        Process a PDF file and extract text, tables, and images.

        Args:
            filepath: Path to PDF file
            ticker: Stock ticker symbol
            doc_type: Type of report (e.g., 'annual_reports')
        """
        try:
            # Create output directories
            subdirs = {
                "images": f"{self.processed_dir}/{ticker}/{doc_type}/processed_images",
                "text": f"{self.processed_dir}/{ticker}/{doc_type}/processed_text",
                "tables": f"{self.processed_dir}/{ticker}/{doc_type}/processed_tables",
                "pages": f"{self.processed_dir}/{ticker}/{doc_type}/processed_page_images",
            }

            for dir_path in subdirs.values():
                os.makedirs(dir_path, exist_ok=True)

            # Open PDF
            doc = pymupdf.open(filepath)
            filename = os.path.basename(filepath)

            # Process each page
            for page_num in tqdm(range(len(doc)), desc="Processing PDF pages"):
                page = doc[page_num]

                # Extract and save text chunks
                text = page.get_text()
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    text_path = f"{subdirs['text']}/{filename}_text_{page_num}_{i}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(chunk)

                # Extract and save tables
                try:
                    tables = tabula.read_pdf(
                        filepath,
                        pages=page_num + 1,
                        multiple_tables=True,
                        force_subprocess=True,
                        encoding="utf-8",
                    )

                    for table_idx, table in enumerate(tables):
                        table_text = "\n".join(
                            [" | ".join(map(str, row)) for row in table.values]
                        )
                        table_path = f"{subdirs['tables']}/{filename}_table_{page_num}_{table_idx}.txt"
                        with open(table_path, "w", encoding="utf-8") as f:
                            f.write(table_text)

                except Exception as e:
                    logging.warning(
                        f"Error extracting tables from page {page_num}: {str(e)}"
                    )

                # Extract and save images
                images = page.get_images()
                for idx, image in enumerate(images):
                    try:
                        xref = image[0]
                        pix = pymupdf.Pixmap(doc, xref)
                        image_path = f"{subdirs['images']}/{filename}_image_{page_num}_{idx}_{xref}.png"
                        pix.save(image_path)
                    except Exception as e:
                        logging.warning(
                            f"Error saving image {idx} from page {page_num}: {str(e)}"
                        )

                # Save page as image
                try:
                    pix = page.get_pixmap()
                    page_path = f"{subdirs['pages']}/page_{page_num:03d}.png"
                    pix.save(page_path)
                except Exception as e:
                    logging.warning(f"Error saving page {page_num} as image: {str(e)}")

        except Exception as e:
            logging.error(f"Error processing PDF {filepath}: {str(e)}")
            raise

    def load_processed_documents(
        self,
        ticker: str,
        doc_type: str,
        file_type: str = "text",
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Load processed documents for a specific ticker and report type.

        Args:
            ticker: Stock ticker symbol
            doc_type: Type of document to load ('reddit_posts', 'financial_statements', 'earnings_calls', 'articles')
            file_type: Type of file to load ('text' or 'tables')
            batch_size: Number of documents to load at once

        Returns:
            List of documents with metadata
        """
        try:
            path = f"{self.processed_dir}/{ticker}/{doc_type}/processed_{file_type}"
            loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)

            # Load documents in batches
            all_documents = []
            for i in range(0, len(loader.load()), batch_size):
                batch = loader.load()[i : i + batch_size]

                # Add metadata
                for doc in batch:
                    doc.metadata.update(
                        {
                            "ticker": ticker,
                            "doc_type": doc_type,
                            "file_type": file_type,
                        }
                    )

                all_documents.extend(batch)

            return all_documents

        except Exception as e:
            logging.error(f"Error loading processed documents: {str(e)}")
            raise

    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        try:
            return {
                "total_documents": len(documents),
                "avg_length": sum(len(doc.page_content) for doc in documents)
                / len(documents)
                if documents
                else 0,
                "doc_types": set(doc.metadata.get("doc_type") for doc in documents),
            }
        except Exception as e:
            logging.error(f"Error calculating document stats: {str(e)}")
            raise

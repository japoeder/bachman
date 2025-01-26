![Bachman Banner](https://raw.githubusercontent.com/japoeder/bachman/main/bachman/_img/bachman_banner.jpg)

# Bachman 🚀

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

[![MongoDB](https://img.shields.io/badge/MongoDB-4.4%2B-green.svg)](https://www.mongodb.com/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust financial document processing service for sentiment analysis, content extraction, and semantic search powered by Groq LLM and vector storage.

## 📋 Table of Contents

- [Bachman 🚀](#bachman-)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [✨ Features](#-features)
  - [🛠 Installation](#-installation)
  - [⚙️ Configuration](#️-configuration)
    - [LLM Services](#llm-services)
    - [Vector Storage](#vector-storage)
    - [Processing Options](#processing-options)
  - [📖 API Usage](#-api-usage)
    - [Authentication](#authentication)
    - [Endpoints](#endpoints)
  - [📁 Project Structure](#-project-structure)
  - [🔧 Development](#-development)
    - [Service Management](#service-management)
    - [Code Quality](#code-quality)

## 🔍 Overview

Bachman API is a specialized service for processing and analyzing financial documents. It provides three primary functionalities:

1. **Document Processing**: Extracts and structures content from financial documents, including tables and sections
2. **Sentiment Analysis**: Leverages Groq LLM for advanced financial sentiment analysis
3. **Vector Storage**: Enables efficient storage and retrieval of processed documents using Qdrant

The service is designed to handle various financial document types including:
- Financial Statements
- Earnings Call Transcripts
- News Articles
- Research Reports

## ✨ Features

- Document Processing
  - PDF and text file support
  - Automated table extraction
  - Section recognition for financial documents
  - Configurable chunking strategies

- Sentiment Analysis
  - Groq LLM integration
  - Financial context awareness
  - Batch and real-time processing
  - Customizable analysis parameters

- Storage & Retrieval
  - Qdrant vector database integration
  - Efficient document embedding
  - Content-based similarity search
  - Metadata-rich storage

- System Features
  - REST API interface
  - Robust error handling and logging
  - Configurable processing pipelines
  - Document deduplication

## 🛠 Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your API keys (see Configuration section)

## ⚙️ Configuration

Required API keys and settings:

### LLM Services
- **Groq**: Primary LLM for sentiment analysis and document processing
  - Set `GROQ_API_KEY` in environment

### Vector Storage
- **Qdrant**: Document storage and retrieval
  - Set `QDRANT_HOST` and `QDRANT_PORT` in environment

### Processing Options
- **Chunking**: Configure text splitting parameters
  - Default chunk size: 512 tokens
  - Default overlap: 50 tokens

## 📖 API Usage

### Authentication

All endpoints require an API key passed in the header:
```bash
-H "x-api-key: your-api-key"
```

### Endpoints

#### 1. Process Text
Process and store text content with optional sentiment analysis.

```bash
POST /bachman/process_text
```

Request:
```json
{
    "text": "Your text content here",
    "collection_name": "financial_reports",
    "metadata": {
        "ticker": "AAPL",
        "source": "earnings_call"
    },
    "chunking_config": {
        "strategy": "recursive",
        "chunk_size": 4096,
        "chunk_overlap": 200
    },
    "skip_if_exists": false
}
```

#### 2. Process File
Process and store document content from a file.

```bash
POST /bachman/process_file
```

Request:
```json
{
    "file_path": "/path/to/document.pdf",
    "collection_name": "financial_reports",
    "document_type": "financial_statement",
    "process_sentiment": true,
    "metadata": {
        "ticker": "AAPL",
        "year": "2024"
    }
}
```

#### 3. Analyze Text
Perform sentiment analysis on text content.

```bash
POST /bachman/analyze
```

Request:
```json
{
    "text": "Your text content here",
    "collection_name": "sentiment_analysis",
    "ticker": "AAPL",
    "analysis_type": "sentiment",
    "load_type": "live"
}
```

#### 4. Get Sentiment
Retrieve sentiment analysis for a previously processed document.

```bash
POST /bachman/get_sentiment
```

Request:
```json
{
    "doc_id": "document-uuid",
    "collection_name": "financial_reports"
}
```

#### 5. Search
Search for documents in the vector store.

```bash
POST /bachman/search
```

Request:
```json
{
    "collection_name": "financial_reports",
    "metadata": {
        "ticker": "AAPL",
        "source": "earnings_call"
    }
}
```

#### 6. Delete
Delete a specific point or entire collection.

```bash
PUT /bachman/delete
```

Delete specific point:
```json
{
    "collection_name": "financial_reports",
    "qdrant_id": "point-id"
}
```

Delete entire collection:
```json
{
    "collection_name": "financial_reports",
    "confirm_coll_delete": true
}
```

## 📁 Project Structure

```
bachman/
├── README.md         # Project documentation
├── __init__.py       # Package initialization
├── processors/       # Core processing modules
│   ├── chunking.py      # Text chunking logic
│   ├── document_types.py# Document type definitions
│   ├── file_processor.py# File handling logic
│   ├── sentiment.py     # Sentiment analysis
│   ├── text_processor.py# Text processing
│   └── types.py         # Type definitions
├── models/          # ML models
│   ├── embeddings.py    # Document embedding
│   └── llm.py          # LLM integration
├── utils/           # Utility functions
│   └── vector_store.py  # Vector DB operations
├── .pre-commit-config.yaml
├── .pylintrc
├── pyproject.toml
└── requirements.txt  # Dependencies
```

## 🔧 Development

### Service Management

```bash
# Start the API service
python -m bachman

# Run tests
pytest tests/

# Generate documentation
pdoc --html bachman/
```
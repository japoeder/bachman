![Bachman Banner](https://raw.githubusercontent.com/japoeder/bachman/main/bachman/_img/bachman_banner.png)

# Bachman 🚀

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

[![MongoDB](https://img.shields.io/badge/MongoDB-4.4%2B-green.svg)](https://www.mongodb.com/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust financial data ingestion service for real-time and historical stock pricing, company news, social data, and more.

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
    - [📖 Usage](#-usage)
        - [API Endpoints](#api-endpoints)
            - [Process Document](#process-document)
            - [Parameters](#parameters)
            - [Analyze Text](#analyze-text)
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

## 📖 Usage

### API Endpoints

#### Process Document

```bash
curl -X POST http://localhost:5000/bachman/process_file \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "collection_name": "financial_reports",
    "document_type": "financial_statement"
  }'
```

#### Parameters

| Parameter           | Description             | Default     | Required |
| ------------------- | ----------------------- | ----------- | -------- |
| `file_path`         | Path to document        | -           | Yes      |
| `collection_name`   | Vector store collection | "documents" | No       |
| `document_type`     | Type of document        | "generic"   | No       |
| `process_sentiment` | Run sentiment analysis  | true        | No       |
| `skip_if_exists`    | Skip if hash exists     | false       | No       |

#### Analyze Text

```bash
curl -X POST http://localhost:5000/bachman/analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "text": "Your text content here",
    "ticker": "AAPL",
    "analysis_type": "sentiment"
  }'
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

### Code Quality

This project uses:

- Black for code formatting
- Pylint for code analysis
- Pre-commit hooks for consistency
- Type hints throughout
- Comprehensive docstrings
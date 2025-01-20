"""
Text processing utilities
"""

import re
from transformers import AutoTokenizer
from bs4 import BeautifulSoup


def extract_ticker(text):
    """
    Simple pattern matching/tokenization for ticker symbols
    """
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokens = tokenizer(text)

    # Simple pattern matching is enough here
    pattern = r"\$[A-Z]{1,5}"  # Matches $AAPL, $GOOGL, etc.
    return re.findall(pattern, text)


def parse_article(html_content):
    """
    Extract article body from HTML
    Uses basic tokenization for text cleanup
    """
    # Simple text extraction and cleanup
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = BeautifulSoup(html_content, "html.parser").get_text()
    tokens = tokenizer(text, truncation=True, max_length=512)
    return tokens

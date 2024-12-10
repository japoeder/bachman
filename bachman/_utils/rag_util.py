"""
RAG utilities for querying financial documents.
"""

import re
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def setup_rag_pipeline(persist_dir, model_name="meta-llama/Meta-Llama-3-8B"):
    """
    Setup RAG pipeline with ChromaDB and LLM
    """
    # Setup embedding function
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load vector database
    vectordb = Chroma(
        persist_directory=persist_dir, embedding_function=embedding_function
    )

    # Setup LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Setup retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    return qa_chain


def extract_amount(text):
    """Extract numerical amount from text"""
    pattern = r"\$?(\d+(?:\.\d+)?)\s*(?:billion|B|million|M|thousand|K)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        amount = float(match.group(1))
        multiplier = (
            {
                "billion": 1e9,
                "B": 1e9,
                "million": 1e6,
                "M": 1e6,
                "thousand": 1e3,
                "K": 1e3,
            }.get(match.group(2), 1)
            if match.group(2)
            else 1
        )
        return amount * multiplier
    return None


def is_reasonable_amount(amount, metric_type="revenue"):
    """Check if amount is within reasonable ranges"""
    ranges = {
        "revenue": (1e8, 1e12),  # $100M to $1T
        "profit": (1e7, 1e11),  # $10M to $100B
        "assets": (1e8, 1e13),  # $100M to $10T
    }
    min_val, max_val = ranges.get(metric_type, (0, float("inf")))
    return min_val <= amount <= max_val


def query_financial_data(query, qa_chain, vectordb):
    """
    Enhanced query with data validation and source checking
    """
    # First check if we have relevant data
    relevant_docs = vectordb.similarity_search(
        query, k=3, fetch_k=10  # Fetch more to check data availability
    )

    # Extract years from query
    year_pattern = r"\b20\d{2}\b"
    query_years = re.findall(year_pattern, query)

    if query_years:
        # Check if we have documents for the requested year
        year_found = any(
            str(year) in str(doc.page_content)
            for year in query_years
            for doc in relevant_docs
        )
        if not year_found:
            return {
                "answer": f"I don't have reliable data for the year(s) {', '.join(query_years)} in my database.",
                "confidence": "high",
                "reason": "Missing year data",
            }

    # Get response
    response = qa_chain.invoke(query)

    # Validate numerical responses
    if "$" in response["result"]:
        amount = extract_amount(response["result"])
        if amount and not is_reasonable_amount(amount, "revenue"):
            return {
                "answer": "The retrieved amount seems unreasonable. Please verify with source documents.",
                "confidence": "low",
                "sources": [doc.metadata for doc in response["source_documents"]],
            }

    # Include source documents in response
    return {
        "answer": response["result"],
        "confidence": "medium",
        "sources": [doc.metadata for doc in response["source_documents"]],
    }

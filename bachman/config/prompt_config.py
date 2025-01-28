"""
Prompt configuration for the RAG analysis.
"""


def prompt_config(doc_type: str, ticker: str, chunks_text: str) -> str:
    """
    Generate a prompt for the RAG analysis.
    """
    prompt = ""
    if doc_type == "financial_report":
        prompt = f"""You are a financial analyst tasked with analyzing a {doc_type}. 
                Please analyze the following document sections and provide a comprehensive analysis.

                Document Type: {doc_type}
                Company: {ticker}
                
                Document Sections:
                {chunks_text}

                Analyze the document and provide your analysis in the following JSON format:
                {{
                    "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                    "confidence_level": "1-10",
                    "time_horizon": "SHORT_TERM/MEDIUM_TERM/LONG_TERM",
                    "trading_signal": "BUY/HOLD/SELL",
                    "key_themes": [
                        "List of key themes identified",
                        "Each as a separate string"
                    ],
                    "supporting_evidence": {{
                        "revenue": "Quote about revenue",
                        "market_position": "Quote about market position",
                        "risks": "Quote about risks"
                    }},
                    "analysis_details": {{
                        "financial_metrics": {{
                            "revenue_trend": "description",
                            "profit_margins": "description",
                            "cash_flow": "description",
                            "debt_levels": "description"
                        }},
                        "business_performance": {{
                            "market_position": "description",
                            "competitive_advantages": "description",
                            "risk_factors": "description"
                        }}
                    }}
                }}

                Ensure your response is valid JSON and includes all the fields specified above.
                """
    return prompt

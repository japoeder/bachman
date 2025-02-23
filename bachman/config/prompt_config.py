"""
Prompt configuration for the RAG analysis.
"""


def prompt_config(
    doc_type: str,
    ticker: str,
    chunks_text: str,
    entity_type: str = None,
    inference_type: str = None,
) -> str:
    """
    Generate a prompt for the RAG analysis.
    """
    prompt = ""
    if inference_type == "sentiment":
        if entity_type == "org":
            if doc_type == "financial_document":
                prompt = f"""You are a financial analyst tasked with analyzing a {doc_type} for {ticker}.
                        Please provide a comprehensive analysis of the following document sections for the ticker.
                        If competitors discussed, please ensure ticker assessment is impacted.
                        
                        Document Sections:
                        {chunks_text}

                        Analyze the document and provide your analysis in the following JSON format:
                        {{
                            "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                            "confidence_level": "1, 2, 3, 4, 5, 6, 7, 8, 9, or 10",
                            "time_horizon": "IMMEDIATE/SHORT/MEDIUM/LONG",
                            "trading_signal": "BUY/HOLD/SELL",
                            "key_themes": [
                                "List of key themes identified",
                                "Each as a separate string"
                            ],
                            "supporting_evidence": {{
                                "revenue": "Revenue description",
                                "solvency": "Solvency description",
                                "liquidity": "Liquidity description",
                                "profitability": "Profitability description",
                                "growth": "Growth description",
                                "cash_flow": "Cash flow description",
                                "market_impact": "Market impact description",
                                "risks": "Risks description"
                            }}
                        }}

                        Please only return valid JSON that includes all the fields specified above. If not enough info
                        to fill in a field, please say "No data."
                        """
            elif doc_type == "news_article":
                prompt = f"""You are a financial analyst tasked with analyzing a {doc_type} for {ticker}.
                        Please provide a comprehensive analysis of the following document sections for the ticker.
                        If competitors discussed, please ensure ticker assessment is impacted.

                        Content:
                        {chunks_text}

                        Provide your analysis in the following JSON format:
                        {{
                            "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                            "confidence_level": "1, 2, 3, 4, 5, 6, 7, 8, 9, or 10",
                            "time_horizon": "IMMEDIATE/SHORT/MEDIUM/LONG",
                            "trading_signal": "BUY/HOLD/SELL",
                            "key_themes": [
                                "List of key themes identified",
                                "Each as a separate string"
                            ],
                            "supporting_evidence": {{
                                "revenue": "Revenue description",
                                "solvency": "Solvency description",
                                "liquidity": "Liquidity description",
                                "profitability": "Profitability description",
                                "growth": "Growth description",
                                "cash_flow": "Cash flow description",
                                "market_impact": "Market impact description",
                                "risks": "Risks description"
                            }}
                        }}                        

                        Please only return valid JSON that includes all the fields specified above. If not enough info
                        to fill in a field, please say "No data."
                        """
            elif doc_type == "reddit_content":
                prompt = f"""You are a financial analyst tasked with analyzing a {doc_type} for {ticker}.
                        Please provide a comprehensive analysis of the following document sections for the ticker.
                        If competitors discussed, please ensure ticker assessment is impacted.

                        Comment or Post Content:
                        {chunks_text}

                        Provide your analysis in the following JSON format:
                        {{
                            "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                            "confidence_level": "1, 2, 3, 4, 5, 6, 7, 8, 9, or 10",
                            "time_horizon": "IMMEDIATE/SHORT/MEDIUM/LONG",
                            "trading_signal": "BUY/HOLD/SELL",
                            "key_themes": [
                                "List of key themes identified",
                                "Each as a separate string"
                            ],
                            "supporting_evidence": {{
                                "revenue": "Revenue description",
                                "solvency": "Solvency description",
                                "liquidity": "Liquidity description",
                                "profitability": "Profitability description",
                                "growth": "Growth description",
                                "cash_flow": "Cash flow description",
                                "market_impact": "Market impact description",
                                "risks": "Risks description"
                            }}
                        }}

                        Please only return valid JSON that includes all the fields specified above. If not enough info
                        to fill in a field, please say "No data."
                        """
            elif doc_type == "earnings_transcript":
                prompt = f"""You are a financial analyst tasked with analyzing a {doc_type} for {ticker}.
                        Please provide a comprehensive analysis of the following document sections for the ticker.
                        If competitors discussed, please ensure ticker assessment is impacted.

                        Earnings Transcript:
                        {chunks_text}

                        Provide your analysis in the following JSON format:
                        {{
                            "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                            "confidence_level": "1, 2, 3, 4, 5, 6, 7, 8, 9, or 10",
                            "time_horizon": "IMMEDIATE/SHORT/MEDIUM/LONG",
                            "trading_signal": "BUY/HOLD/SELL",
                            "key_themes": [
                                "List of key themes identified",
                                "Each as a separate string"
                            ],
                            "supporting_evidence": {{
                                "revenue": "Revenue description",
                                "solvency": "Solvency description",
                                "liquidity": "Liquidity description",
                                "profitability": "Profitability description",
                                "growth": "Growth description",
                                "cash_flow": "Cash flow description",
                                "market_impact": "Market impact description",
                                "risks": "Risks description"
                            }}
                        }}

                        Please only return valid JSON that includes all the fields specified above. If not enough info
                        to fill in a field, please say "No data."
                        """
    return prompt

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
            if doc_type == "financial_report":
                prompt = f"""You are a financial analyst tasked with analyzing a {doc_type}. 
                        Please analyze the following document sections and provide a comprehensive analysis.

                        Document Type: {doc_type}
                        Ticker: {ticker}
                        
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

                        Ensure your response is valid JSON and includes all the fields specified above for the aforementioned ticker.
                        """
            elif doc_type == "news_article":
                prompt = f"""You are a financial analyst tasked with analyzing a {doc_type}. 
                        Please analyze the following article content and provide a sentiment analysis.
                        The goal is to rate the sentiment of the article as positive, negative, or neutral and
                        provide a buy, sell, or hold recommendation based on the sentiment.  You can also provide
                        confidence level for the recommendation.

                        Document Type: {doc_type}
                        Ticker: {ticker}

                        Content:
                        {chunks_text}

                        Provide your analysis in the following JSON format:
                        {{
                            "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                            "confidence_level": "1-10",
                            "trading_signal": "BUY/HOLD/SELL",
                            "key_themes": [
                                "List of key themes identified",
                                "Each as a separate string"
                            ],
                            "supporting_evidence": {{
                                "key_quotes": ["relevant quote 1", "relevant quote 2"],
                                "market_impact": "brief description"
                            }}
                        }}                        

                        Ensure your response is valid JSON and includes all the fields specified above for the aforementioned ticker.
                        """
            elif doc_type == "reddit_content":
                prompt = f"""You are a financial analyst tasked with analyzing {doc_type} for investing. 
                        Please analyze the following comment or post content and provide a sentiment analysis.
                        The goal is to rate the sentiment of the article as positive, negative, or neutral and
                        provide a buy, sell, or hold recommendation based on the sentiment.  You can also provide
                        confidence level for the recommendation.

                        Document Type: {doc_type}
                        Ticker: {ticker}

                        Comment or Post Content:
                        {chunks_text}

                        Provide your analysis in the following JSON format:
                        {{
                            "dominant_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
                            "confidence_level": "1-10",
                            "trading_signal": "BUY/HOLD/SELL",
                            "key_themes": [
                                "List of key themes identified",
                                "Each as a separate string"
                            ],
                            "supporting_evidence": {{
                                "key_quotes": ["relevant quote 1", "relevant quote 2"],
                                "market_impact": "brief description"
                            }}
                        }}

                        Ensure your response is valid JSON and includes all the fields specified above for the aforementioned ticker.
                        """
    return prompt

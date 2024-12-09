"""
Estimate the sentiment of a news article using a pre-trained model.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# from typing import Tuple

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# model="cardiffnlp/twitter-roberta-base-sentiment"
model = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
labels = ["positive", "negative", "neutral"]


def estimate_sentiment(news):
    """
    Estimate the sentiment of a news article.
    """
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        output_sentiment = labels[torch.argmax(result)]
        return probability, output_sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(
        [
            "markets responded positively to the news!",
            "traders were pleasantly surprised!",
        ]
    )
    print(tensor, sentiment)

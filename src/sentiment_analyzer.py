import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Load the FinBERT model and tokenizer
# This might download the model files the first time it's run
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using the FinBERT model.

    Args:
        text (str): The input text (e.g., news headline or description).

    Returns:
        tuple: A tuple containing:
            - sentiment_label (str): 'positive', 'negative', or 'neutral'.
            - sentiment_score (float): The probability score of the predicted sentiment.
            - scores (dict): Dictionary containing probabilities for all labels ('positive', 'negative', 'neutral').
        Returns (None, None, None) if the input is invalid or an error occurs.
    """
    if not isinstance(text, str) or not text.strip():
        return None, None, None # Return None for empty or invalid input

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(**inputs)

        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probabilities[0].numpy() # Get scores for the first (and only) input

        # Get the predicted sentiment label index
        predicted_class_id = np.argmax(scores)

        # Map index to label based on model config
        sentiment_label = model.config.id2label[predicted_class_id]
        sentiment_score = scores[predicted_class_id]

        all_scores = {model.config.id2label[i]: scores[i] for i in range(len(scores))}

        return sentiment_label, float(sentiment_score), {k: float(v) for k, v in all_scores.items()}

    except Exception as e:
        print(f"Error during sentiment analysis for text: '{text[:50]}...': {e}")
        return None, None, None

# Example usage (for testing the module directly)
if __name__ == '__main__':
    test_texts = [
        "Stocks rallied on positive economic news.",
        "The company reported a significant drop in profits.",
        "Market remains flat amid uncertainty.",
        "", # Empty string test
        None # None test
    ]

    print("--- Testing Sentiment Analysis ---")
    for t in test_texts:
        label, score, all_scores_dict = analyze_sentiment(t)
        if label:
            print(f"Text: '{t}'")
            print(f"  Sentiment: {label} (Score: {score:.4f})")
            print(f"  All Scores: {all_scores_dict}")
        else:
            print(f"Text: '{t}' -> Invalid input or error during analysis.")

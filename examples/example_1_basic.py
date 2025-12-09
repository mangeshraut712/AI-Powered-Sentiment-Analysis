"""
Example 1: Basic Sentiment Analysis
Demonstrates simple text classification
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sentiment_classifier import SentimentClassifier

# Initialize classifier
print("🤖 Initializing Sentiment Classifier...")
classifier = SentimentClassifier()

# Example texts
examples = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible product. Broke after one day. Very disappointed.",
    "The service was okay, nothing special but not bad either.",
    "Best purchase I've ever made! Highly recommend to everyone!",
    "Worst experience of my life. Never coming back here again."
]

print("\n" + "="*60)
print("SENTIMENT ANALYSIS EXAMPLES")
print("="*60 + "\n")

# Analyze each example
for i, text in enumerate(examples, 1):
    sentiment, confidence = classifier.classify(text)
    
    # Format output
    emoji = "😊" if sentiment == "positive" else "😞"
    color = "\033[92m" if sentiment == "positive" else "\033[91m"
    reset = "\033[0m"
    
    print(f"Example {i}:")
    print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"Result: {emoji} {color}{sentiment.upper()}{reset} ({confidence:.1%} confidence)")
    print()

print("="*60)
print("✅ Analysis Complete!")

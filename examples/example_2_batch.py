"""
Example 2: Batch Processing
Demonstrates analyzing multiple texts efficiently
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sentiment_classifier import SentimentClassifier
import time

# Initialize classifier
print("🤖 Initializing Sentiment Classifier...")
classifier = SentimentClassifier()

# Sample movie reviews
reviews = [
    "The cinematography was breathtaking and the story was compelling.",
    "Boring plot, terrible acting, waste of time.",
    "A masterpiece! One of the best films I've ever seen.",
    "Disappointing sequel. Nothing like the original.",
    "Absolutely loved it! Can't wait to watch it again.",
    "Poor quality, bad service, very unsatisfied.",
    "Exceeded all my expectations. Highly recommended!",
    "Not worth the money. Very disappointing experience.",
    "Fantastic product! Works perfectly and great value.",
    "Terrible quality. Broke immediately. Don't buy this."
]

print(f"\n📊 Analyzing {len(reviews)} reviews...\n")

# Track timing
start_time = time.time()

# Analyze all reviews
results = []
for review in reviews:
    sentiment, confidence = classifier.classify(review)
    results.append({
        'text': review,
        'sentiment': sentiment,
        'confidence': confidence
    })

# Calculate statistics
positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
negative_count = len(results) - positive_count
avg_confidence = sum(r['confidence'] for r in results) / len(results)
elapsed_time = time.time() - start_time

# Display results
print("="*70)
print("BATCH ANALYSIS RESULTS")
print("="*70 + "\n")

for i, result in enumerate(results, 1):
    emoji = "😊" if result['sentiment'] == 'positive' else "😞"
    print(f"{i}. {emoji} {result['sentiment'].upper()} ({result['confidence']:.1%})")
    print(f"   {result['text'][:60]}{'...' if len(result['text']) > 60 else ''}")
    print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Total Reviews:      {len(results)}")
print(f"Positive:           {positive_count} ({positive_count/len(results):.1%})")
print(f"Negative:           {negative_count} ({negative_count/len(results):.1%})")
print(f"Avg Confidence:     {avg_confidence:.1%}")
print(f"Processing Time:    {elapsed_time:.3f}s")
print(f"Speed:              {len(results)/elapsed_time:.1f} reviews/second")
print("="*70)

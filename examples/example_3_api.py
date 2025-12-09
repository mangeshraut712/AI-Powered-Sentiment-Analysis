"""
Example 3: API Integration
Demonstrates using the REST API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5001"

print("🌐 Sentiment Analysis API Demo")
print("="*60 + "\n")

# Example 1: Single text analysis
print("1️⃣  Single Text Analysis")
print("-" * 60)

text = "This product is absolutely amazing! Best purchase ever!"
response = requests.post(
    f"{BASE_URL}/api/analyze",
    json={"text": text}
)

if response.status_code == 200:
    result = response.json()
    if result['success']:
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Word Count: {result['word_count']}")
        print("✅ Success!\n")
else:
    print(f"❌ Error: {response.status_code}\n")

# Example 2: Batch analysis
print("2️⃣  Batch Analysis")
print("-" * 60)

texts = [
    "Excellent service and great quality!",
    "Terrible experience. Very disappointed.",
    "Amazing product! Highly recommend!"
]

response = requests.post(
    f"{BASE_URL}/api/batch-analyze",
    json={"texts": texts}
)

if response.status_code == 200:
    result = response.json()
    if result['success']:
        print(f"Analyzed {result['total_analyzed']} texts:\n")
        for i, item in enumerate(result['results'], 1):
            emoji = "😊" if item['sentiment'] == 'positive' else "😞"
            print(f"{i}. {emoji} {item['sentiment'].upper()} ({item['confidence']}%)")
            print(f"   {item['text']}\n")
        print("✅ Success!\n")
else:
    print(f"❌ Error: {response.status_code}\n")

# Example 3: Model statistics
print("3️⃣  Model Statistics")
print("-" * 60)

response = requests.get(f"{BASE_URL}/api/stats")

if response.status_code == 200:
    result = response.json()
    if result['success']:
        stats = result['stats']
        print(f"Positive Documents: {stats['positive_docs']:,}")
        print(f"Negative Documents: {stats['negative_docs']:,}")
        print(f"Total Words: {stats['total_positive_words'] + stats['total_negative_words']:,}")
        print(f"Unique Features: {stats['unique_positive_features'] + stats['unique_negative_features']:,}")
        
        print("\nTop 5 Positive Words:")
        for word in stats['top_positive_words'][:5]:
            print(f"  • {word['word']}: {word['count']:,}")
        
        print("\nTop 5 Negative Words:")
        for word in stats['top_negative_words'][:5]:
            print(f"  • {word['word']}: {word['count']:,}")
        
        print("\n✅ Success!\n")
else:
    print(f"❌ Error: {response.status_code}\n")

print("="*60)
print("🎉 API Demo Complete!")
print("\nNote: Make sure the web server is running:")
print("  python src/web/app.py")

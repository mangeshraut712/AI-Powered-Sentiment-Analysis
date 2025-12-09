# 📚 Examples

This directory contains example scripts demonstrating how to use the Sentiment Analysis system.

## Available Examples

### 1. Basic Sentiment Analysis
**File:** `example_1_basic.py`

Demonstrates simple text classification with multiple examples.

**Run:**
```bash
python examples/example_1_basic.py
```

**What it shows:**
- How to initialize the classifier
- Analyzing individual texts
- Interpreting results with confidence scores
- Formatted output with emojis

---

### 2. Batch Processing
**File:** `example_2_batch.py`

Shows how to efficiently analyze multiple texts at once.

**Run:**
```bash
python examples/example_2_batch.py
```

**What it shows:**
- Processing multiple texts
- Calculating statistics (positive/negative ratio)
- Performance metrics (speed, timing)
- Summary reports

---

### 3. API Integration
**File:** `example_3_api.py`

Demonstrates using the REST API for integration.

**Run:**
```bash
# First, start the web server
python src/web/app.py

# Then in another terminal
python examples/example_3_api.py
```

**What it shows:**
- Single text analysis via API
- Batch analysis via API
- Fetching model statistics
- Error handling

---

## Quick Test

Run all examples at once:

```bash
# Example 1
python examples/example_1_basic.py

# Example 2
python examples/example_2_batch.py

# Example 3 (requires web server)
python src/web/app.py &
sleep 2
python examples/example_3_api.py
```

## Creating Your Own Examples

Use these examples as templates for your own applications:

```python
from src.models.sentiment_classifier import SentimentClassifier

# Initialize
classifier = SentimentClassifier()

# Your code here
sentiment, confidence = classifier.classify("Your text")
print(f"{sentiment}: {confidence:.2%}")
```

## More Resources

- [Quick Start Guide](../docs/QUICKSTART.md)
- [API Documentation](../docs/API.md)
- [Main README](../README.md)

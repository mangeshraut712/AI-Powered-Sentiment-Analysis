from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import pickle
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import BinaryClassifier
from src.preprocessing import clean_text

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js

# Load Models (Global State)
binary_clf = None
emotion_model = None
emotion_vectorizer = None

def load_models():
    global binary_clf, emotion_model, emotion_vectorizer
    
    print("⏳ Loading Models...")
    
    # 1. Binary Sentiment
    try:
        binary_clf = BinaryClassifier()
        print("✅ Binary Sentiment Model Loaded")
    except Exception as e:
        print(f"❌ Binary Model Error: {e}")

    # 2. Emotion Detection
    try:
        with open('results/models/tfidf_vectorizer.pkl', 'rb') as f:
            emotion_vectorizer = pickle.load(f)
        with open('results/models/logistic_regression_model.pkl', 'rb') as f:
            emotion_model = pickle.load(f)
        print("✅ Emotion Detection Model Loaded")
    except Exception as e:
        print(f"⚠️  Emotion Model not found ({e}). Ensure scripts/run_analysis.py has been run.")

load_models()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "binary_loaded": binary_clf is not None, "emotion_loaded": emotion_model is not None})

@app.route('/api/stats', methods=['GET'])
def stats():
    stats_data = {}
    
    # Binary Stats
    if binary_clf and binary_clf.total_words > 0:
        stats_data['binary'] = {
            'positive_docs': binary_clf.num_positive_docs,
            'negative_docs': binary_clf.num_negative_docs,
            'total_words': binary_clf.total_words,
            'top_features': binary_clf.get_top_features(n=10)
        }
    
    # Emotion Stats (Mock or Real if we loaded full dataset info, but usually model doesn't store training dist)
    # We'll serve static emotion stats from the paper/dataset description if model doesn't have it.
    stats_data['emotions'] = {
        "Neutral": 8638, "Worry": 8459, "Happiness": 5209, 
        "Sadness": 5165, "Love": 3842, "Surprise": 2187, "Anger": 1433
    }
    
    return jsonify(stats_data)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    results = {}
    
    # Binary Analysis
    if binary_clf:
        try:
            sentiment, conf = binary_clf.classify(text)
            results['sentiment'] = sentiment.title() # "Positive" or "Negative"
            results['sentiment_confidence'] = float(conf)
        except Exception as e:
            print(f"Binary Error: {e}")
            results['sentiment'] = "Neutral"
            results['sentiment_confidence'] = 0.0
    else:
        results['sentiment'] = "Error"
        results['sentiment_confidence'] = 0.0

    # Emotion Analysis
    if emotion_model and emotion_vectorizer:
        try:
            cleaned = clean_text(text)
            feat = emotion_vectorizer.transform([cleaned])
            pred = emotion_model.predict(feat)[0]
            probs = emotion_model.predict_proba(feat)[0]
            conf = np.max(probs)
            
            results['emotion'] = pred
            results['emotion_confidence'] = float(conf)
        except Exception as e:
            print(f"Emotion Error: {e}")
            results['emotion'] = "Unknown"
            results['emotion_confidence'] = 0.0
    else:
        # Should not happen in production if setup is correct, but graceful fallback
        results['emotion'] = "System Not Ready"
        results['emotion_confidence'] = 0.0

    return jsonify(results)

if __name__ == '__main__':
    print("🚀 Starting Flask API Server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)

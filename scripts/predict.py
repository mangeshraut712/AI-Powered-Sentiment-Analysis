#!/usr/bin/env python3
"""
Interactive Sentiment Predictor
================================

This script allows you to test the trained model on custom text.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from pathlib import Path
from src.preprocessing import clean_text
from src.feature_engineering import create_tfidf_features
import pandas as pd

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    model_path = Path('../results/models/logistic_regression_model.pkl')
    
    if not model_path.exists():
        print("❌ Model not found! Please run 'python run_analysis.py' first.")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    return model

def predict_sentiment(text, model, vectorizer=None):
    """Predict sentiment for a given text."""
    # Clean the text
    cleaned = clean_text(text)
    
    # If we don't have a vectorizer, we need to create features
    # For now, we'll use a simple approach
    # In production, you'd save and load the vectorizer
    
    print(f"\n📝 Original text: {text}")
    print(f"🧹 Cleaned text: {cleaned}")
    
    # Note: This is a simplified version
    # In a real scenario, you'd need to save and load the TF-IDF vectorizer
    print("\n⚠️  Note: Full prediction requires the TF-IDF vectorizer.")
    print("   Run the complete analysis to see predictions.")
    
    return cleaned

def interactive_mode():
    """Run interactive prediction mode."""
    print("\n" + "="*70)
    print("  🤖 Interactive Sentiment Predictor")
    print("="*70)
    print("\nEnter text to analyze (or 'quit' to exit)")
    print("-"*70)
    
    model = load_model_and_vectorizer()
    
    while True:
        print("\n📝 Enter text: ", end='')
        text = input().strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text!")
            continue
        
        cleaned = predict_sentiment(text, model)

def demo_predictions():
    """Show demo predictions on sample texts."""
    print("\n" + "="*70)
    print("  🎯 Demo Predictions")
    print("="*70)
    
    sample_texts = [
        "I love this amazing day! Everything is perfect!",
        "I'm so sad and depressed. Nothing is going right.",
        "This is the worst thing ever. I hate it!",
        "Wow! I can't believe this happened!",
        "I'm worried about the exam tomorrow.",
        "Just another boring day at work.",
        "You are the best! I appreciate you so much!"
    ]
    
    print("\n📊 Analyzing sample texts...\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Original: {text}")
        cleaned = clean_text(text)
        print(f"   Cleaned:  {cleaned}")
        print(f"   Expected: [Run full analysis to see prediction]")

if __name__ == "__main__":
    print("\n🚀 DSCI-521 Sentiment Analysis - Interactive Demo")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_predictions()
    else:
        print("\nOptions:")
        print("  python predict.py          - Interactive mode")
        print("  python predict.py --demo   - Demo predictions")
        print("\nℹ️  Note: Run 'python run_analysis.py' first to train the model!")
        print("\nStarting demo mode...\n")
        demo_predictions()

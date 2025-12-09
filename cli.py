"""
Command Line Interface for Sentiment Analysis
Provides easy-to-use commands for training, testing, and analyzing text
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import BinaryClassifier as SentimentClassifier
from src.preprocessing import clean_text
import pickle
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(args):
    """Train a new model"""
    logger.info("Training new sentiment classifier...")
    
    classifier = SentimentClassifier(
        use_bigrams=args.bigrams,
        remove_stop_words=args.remove_stopwords,
        min_word_length=args.min_length,
        model_name=args.model_name
    )
    
    train_dir = args.train_dir or "./train"
    
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        return
    
    classifier.train(train_dir)
    logger.info("Training complete!")


def evaluate_model(args):
    """Evaluate model on test data"""
    logger.info("Loading model and evaluating...")
    
    classifier = SentimentClassifier(
        use_bigrams=args.bigrams,
        remove_stop_words=args.remove_stopwords,
        min_word_length=args.min_length,
        model_name=args.model_name
    )
    
    test_dir = args.test_dir or "./test"
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    metrics = classifier.evaluate(test_dir)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1_score']*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Positive:  {metrics['true_positive']}")
    print(f"  True Negative:  {metrics['true_negative']}")
    print(f"  False Positive: {metrics['false_positive']}")
    print(f"  False Negative: {metrics['false_negative']}")
    print(f"\nTotal Samples: {metrics['total_samples']}")
    print("="*50 + "\n")


def classify_text(args):
    """Classify a single text"""
    classifier = SentimentClassifier(
        use_bigrams=args.bigrams,
        remove_stop_words=args.remove_stopwords,
        min_word_length=args.min_length,
        model_name=args.model_name
    )
    
    if args.file:
        # Read from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return
    else:
        # Use provided text
        text = args.text
    
    if not text:
        logger.error("No text provided")
        return
    
    sentiment, confidence = classifier.classify(text)
    
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS")
    print("="*50)
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"\nSentiment:  {sentiment.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*50 + "\n")


def show_stats(args):
    """Show model statistics"""
    classifier = SentimentClassifier(
        use_bigrams=args.bigrams,
        remove_stop_words=args.remove_stopwords,
        min_word_length=args.min_length,
        model_name=args.model_name
    )
    
    top_features = classifier.get_top_features(n=args.top_n)
    
    print("\n" + "="*50)
    print("MODEL STATISTICS")
    print("="*50)
    print(f"Positive Documents: {classifier.num_positive_docs:,}")
    print(f"Negative Documents: {classifier.num_negative_docs:,}")
    print(f"Total Positive Words: {classifier.total_positive_words:,}")
    print(f"Total Negative Words: {classifier.total_negative_words:,}")
    print(f"Unique Positive Features: {len(classifier.positive_words):,}")
    print(f"Unique Negative Features: {len(classifier.negative_words):,}")
    
    print(f"\nTop {args.top_n} Positive Words:")
    for word, count in top_features['positive']:
        print(f"  {word:20s} {count:,}")
    
    print(f"\nTop {args.top_n} Negative Words:")
    for word, count in top_features['negative']:
        print(f"  {word:20s} {count:,}")
    
    print("="*50 + "\n")


def predict_emotion(args):
    """Predict emotion (7-class) using the complex pipeline"""
    text = args.text
    if not text:
        logger.error("No text provided")
        return

    try:
        # Paths
        base_path = Path('results/models')
        vec_path = base_path / 'tfidf_vectorizer.pkl'
        model_path = base_path / 'logistic_regression_model.pkl'
        
        if not vec_path.exists() or not model_path.exists():
            logger.error("Models not found. Please run 'python scripts/run_analysis.py' first.")
            return

        # Load artifacts
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Process
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = np.max(probs)
        
        print("\n" + "="*50)
        print("EMOTION DETECTION (7-Class)")
        print("="*50)
        print(f"Text:       {text[:100]}")
        print(f"Cleaned:    {cleaned[:100]}")
        print(f"\nEmotion:    {prediction.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error in emotion prediction: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis CLI - Advanced Naive Bayes Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python cli.py train --train-dir ./train
  
  # Evaluate on test set
  python cli.py evaluate --test-dir ./test
  
  # Classify text
  python cli.py classify --text "This movie was amazing!"
  
  # Classify from file
  python cli.py classify --file review.txt
  
  # Show model statistics
  python cli.py stats --top-n 20
        """
    )
    
    # Common arguments
    parser.add_argument('--model-name', default='enhanced', help='Model name for saving/loading')
    parser.add_argument('--bigrams', action='store_true', default=True, help='Use bigram features')
    parser.add_argument('--no-bigrams', dest='bigrams', action='store_false', help='Disable bigrams')
    parser.add_argument('--remove-stopwords', action='store_true', default=True, help='Remove stop words')
    parser.add_argument('--no-stopwords', dest='remove_stopwords', action='store_false', help='Keep stop words')
    parser.add_argument('--min-length', type=int, default=2, help='Minimum word length')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--train-dir', help='Training data directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--test-dir', help='Test data directory')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify text sentiment')
    classify_parser.add_argument('--text', help='Text to classify')
    classify_parser.add_argument('--file', help='File containing text to classify')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show model statistics')
    stats_parser.add_argument('--top-n', type=int, default=15, help='Number of top features to show')
    
    # Emotion command (New)
    emotion_parser = subparsers.add_parser('emotion', help='Detect specific emotion (7 classes)')
    emotion_parser.add_argument('--text', required=True, help='Text to analyze')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'classify':
        classify_text(args)
    elif args.command == 'stats':
        show_stats(args)
    elif args.command == 'emotion':
        predict_emotion(args)



if __name__ == '__main__':
    main()

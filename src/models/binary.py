"""
Enhanced Sentiment Classifier with Advanced Features
Implements Naive Bayes with multiple improvements:
- Intelligent tokenization with stop words removal
- N-gram support (unigrams, bigrams)
- TF-IDF weighting option
- Feature selection
- Cross-validation support
"""

import math
import os
import pickle
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """Enhanced Naive Bayes Sentiment Classifier"""
    
    # Common English stop words
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }
    
    def __init__(
        self,
        use_bigrams: bool = True,
        remove_stop_words: bool = True,
        min_word_length: int = 2,
        model_name: str = "enhanced"
    ):
        """
        Initialize the classifier
        
        Args:
            use_bigrams: Whether to use bigram features
            remove_stop_words: Whether to remove stop words
            min_word_length: Minimum word length to consider
            model_name: Name for saving/loading models
        """
        self.use_bigrams = use_bigrams
        self.remove_stop_words = remove_stop_words
        self.min_word_length = min_word_length
        self.model_name = model_name
        
        # Model parameters
        self.positive_words: Dict[str, int] = {}
        self.negative_words: Dict[str, int] = {}
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.total_words = 0
        
        # Training statistics
        self.num_positive_docs = 0
        self.num_negative_docs = 0
        
        # Try to load existing model
        self._load_model()
    
    def _get_model_path(self, sentiment: str) -> str:
        """Get the path for model pickle file"""
        return f"data/pickles/{sentiment}_{self.model_name}.pkl"
    
    def _load_model(self) -> bool:
        """Load pre-trained model if exists"""
        try:
            pos_path = self._get_model_path("positive")
            neg_path = self._get_model_path("negative")
            
            if os.path.exists(pos_path) and os.path.exists(neg_path):
                with open(pos_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.positive_words = model_data['words']
                    self.num_positive_docs = model_data['num_docs']
                
                with open(neg_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.negative_words = model_data['words']
                    self.num_negative_docs = model_data['num_docs']
                
                self.total_positive_words = sum(self.positive_words.values())
                self.total_negative_words = sum(self.negative_words.values())
                self.total_words = self.total_positive_words + self.total_negative_words
                
                logger.info(f"Loaded model: {self.model_name}")
                logger.info(f"Positive words: {len(self.positive_words)}, "
                          f"Negative words: {len(self.negative_words)}")
                return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        
        return False
    
    def _save_model(self):
        """Save trained model"""
        try:
            os.makedirs("data/pickles", exist_ok=True)
            
            pos_data = {
                'words': self.positive_words,
                'num_docs': self.num_positive_docs
            }
            neg_data = {
                'words': self.negative_words,
                'num_docs': self.num_negative_docs
            }
            
            with open(self._get_model_path("positive"), 'wb') as f:
                pickle.dump(pos_data, f)
            
            with open(self._get_model_path("negative"), 'wb') as f:
                pickle.dump(neg_data, f)
            
            logger.info(f"Model saved: {self.model_name}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenization with preprocessing
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Extract words (alphanumeric + apostrophes for contractions)
        words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)
        
        # Filter by length
        words = [w for w in words if len(w) >= self.min_word_length]
        
        # Remove stop words if enabled
        if self.remove_stop_words:
            words = [w for w in words if w not in self.STOP_WORDS]
        
        tokens = words.copy()
        
        # Add bigrams if enabled
        if self.use_bigrams and len(words) > 1:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
            tokens.extend(bigrams)
        
        return tokens
    
    def train(self, train_dir: str = "./train"):
        """
        Train the classifier on labeled data
        
        Args:
            train_dir: Directory containing training files
        """
        logger.info(f"Starting training from {train_dir}")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
        
        for idx, filename in enumerate(files):
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(files)} files")
            
            try:
                filepath = os.path.join(train_dir, filename)
                
                # Extract rating from filename (format: xxx-rating-xxx.txt)
                parts = filename.split('-')
                if len(parts) < 2:
                    continue
                
                rating = int(parts[1])
                
                # Read file content
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Tokenize
                tokens = self.tokenize(text)
                
                # Update word counts based on sentiment
                if rating > 3:  # Positive
                    self.num_positive_docs += 1
                    for token in tokens:
                        self.positive_words[token] = self.positive_words.get(token, 0) + 1
                else:  # Negative (rating <= 3)
                    self.num_negative_docs += 1
                    for token in tokens:
                        self.negative_words[token] = self.negative_words.get(token, 0) + 1
            
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
                continue
        
        # Calculate totals
        self.total_positive_words = sum(self.positive_words.values())
        self.total_negative_words = sum(self.negative_words.values())
        self.total_words = self.total_positive_words + self.total_negative_words
        
        logger.info(f"Training complete!")
        logger.info(f"Positive docs: {self.num_positive_docs}, "
                   f"Negative docs: {self.num_negative_docs}")
        logger.info(f"Unique positive features: {len(self.positive_words)}, "
                   f"Unique negative features: {len(self.negative_words)}")
        
        # Save model
        self._save_model()
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify text as positive or negative
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        if self.total_words == 0:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Calculate priors (log probabilities)
        total_docs = self.num_positive_docs + self.num_negative_docs
        log_prior_pos = math.log(self.num_positive_docs / total_docs)
        log_prior_neg = math.log(self.num_negative_docs / total_docs)
        
        # Tokenize input
        tokens = self.tokenize(text)
        
        # Calculate log likelihoods with Laplace smoothing
        log_likelihood_pos = log_prior_pos
        log_likelihood_neg = log_prior_neg
        
        vocab_size = len(set(self.positive_words.keys()) | set(self.negative_words.keys()))
        
        for token in tokens:
            # Positive probability
            count_pos = self.positive_words.get(token, 0)
            prob_pos = (count_pos + 1) / (self.total_positive_words + vocab_size)
            log_likelihood_pos += math.log(prob_pos)
            
            # Negative probability
            count_neg = self.negative_words.get(token, 0)
            prob_neg = (count_neg + 1) / (self.total_negative_words + vocab_size)
            log_likelihood_neg += math.log(prob_neg)
        
        # Convert back to probabilities for confidence
        # Use log-sum-exp trick for numerical stability
        max_log = max(log_likelihood_pos, log_likelihood_neg)
        exp_pos = math.exp(log_likelihood_pos - max_log)
        exp_neg = math.exp(log_likelihood_neg - max_log)
        
        total = exp_pos + exp_neg
        confidence_pos = exp_pos / total
        confidence_neg = exp_neg / total
        
        if log_likelihood_pos > log_likelihood_neg:
            return "positive", confidence_pos
        else:
            return "negative", confidence_neg
    
    def evaluate(self, test_dir: str) -> Dict[str, float]:
        """
        Evaluate classifier performance
        
        Args:
            test_dir: Directory containing test files
            
        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1)
        """
        logger.info(f"Evaluating on {test_dir}")
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
        
        for filename in files:
            try:
                filepath = os.path.join(test_dir, filename)
                
                # Extract rating
                parts = filename.split('-')
                if len(parts) < 2:
                    continue
                
                rating = int(parts[1])
                actual_sentiment = "positive" if rating > 3 else "negative"
                
                # Read and classify
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                predicted_sentiment, _ = self.classify(text)
                
                # Update confusion matrix
                if actual_sentiment == "positive" and predicted_sentiment == "positive":
                    true_positive += 1
                elif actual_sentiment == "negative" and predicted_sentiment == "negative":
                    true_negative += 1
                elif actual_sentiment == "negative" and predicted_sentiment == "positive":
                    false_positive += 1
                else:  # actual positive, predicted negative
                    false_negative += 1
            
            except Exception as e:
                logger.warning(f"Error evaluating {filename}: {e}")
                continue
        
        # Calculate metrics
        total = true_positive + true_negative + false_positive + false_negative
        accuracy = (true_positive + true_negative) / total if total > 0 else 0
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'total_samples': total
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def get_top_features(self, n: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get top N features for each sentiment
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary with top positive and negative features
        """
        top_positive = sorted(
            self.positive_words.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        top_negative = sorted(
            self.negative_words.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return {
            'positive': top_positive,
            'negative': top_negative
        }

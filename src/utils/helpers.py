"""
Utility functions for sentiment analysis
"""

import re
from typing import List, Dict
import os
import json


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def load_stopwords(filepath: str = None) -> set:
    """
    Load stop words from file
    
    Args:
        filepath: Path to stop words file
        
    Returns:
        Set of stop words
    """
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f)
    
    # Default stop words
    return {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }


def calculate_metrics(true_pos: int, true_neg: int, false_pos: int, false_neg: int) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        true_pos: True positives
        true_neg: True negatives
        false_pos: False positives
        false_neg: False negatives
        
    Returns:
        Dictionary with metrics
    """
    total = true_pos + true_neg + false_pos + false_neg
    
    accuracy = (true_pos + true_neg) / total if total > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_results(results: Dict, filepath: str):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

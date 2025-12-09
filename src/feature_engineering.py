"""
Feature Engineering Utilities
==============================

This module provides functions for extracting features from text data.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import re


def extract_keywords(df, text_column='cleaned_text', sentiment_column='sentiment', 
                    top_n=10, method='frequency'):
    """
    Extract top keywords for each sentiment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    sentiment_column : str
        Name of the sentiment column
    top_n : int
        Number of top keywords to extract
    method : str
        Method to use ('frequency' or 'tfidf')
    
    Returns:
    --------
    dict
        Dictionary mapping sentiments to their top keywords
    """
    keywords = {}
    
    for sentiment in df[sentiment_column].unique():
        # Get all text for this sentiment
        sentiment_texts = df[df[sentiment_column] == sentiment][text_column]
        
        if method == 'frequency':
            # Simple word frequency
            all_words = ' '.join(sentiment_texts).split()
            word_counts = Counter(all_words)
            keywords[sentiment] = [word for word, count in word_counts.most_common(top_n)]
        
        elif method == 'tfidf':
            # TF-IDF based keywords
            vectorizer = TfidfVectorizer(max_features=top_n)
            try:
                vectorizer.fit(sentiment_texts)
                keywords[sentiment] = vectorizer.get_feature_names_out().tolist()
            except:
                keywords[sentiment] = []
    
    return keywords


def create_tfidf_features(df, text_column='cleaned_text', max_features=1000, 
                         ngram_range=(1, 2), min_df=2):
    """
    Create TF-IDF features from text.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    max_features : int
        Maximum number of features
    ngram_range : tuple
        Range of n-grams to consider
    min_df : int
        Minimum document frequency
    
    Returns:
    --------
    tuple
        (feature_matrix, vectorizer, feature_names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'
    )
    
    feature_matrix = vectorizer.fit_transform(df[text_column])
    feature_names = vectorizer.get_feature_names_out()
    
    return feature_matrix, vectorizer, feature_names


def create_count_features(df, text_column='cleaned_text', max_features=1000,
                         ngram_range=(1, 1), min_df=2):
    """
    Create count-based features from text.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    max_features : int
        Maximum number of features
    ngram_range : tuple
        Range of n-grams to consider
    min_df : int
        Minimum document frequency
    
    Returns:
    --------
    tuple
        (feature_matrix, vectorizer, feature_names)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'
    )
    
    feature_matrix = vectorizer.fit_transform(df[text_column])
    feature_names = vectorizer.get_feature_names_out()
    
    return feature_matrix, vectorizer, feature_names


def add_text_statistics(df, text_column='cleaned_text'):
    """
    Add statistical features about the text.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added statistical features
    """
    df = df.copy()
    
    # Character count
    df['char_count'] = df[text_column].apply(len)
    
    # Word count
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    # Average word length
    df['avg_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    # Exclamation marks
    df['exclamation_count'] = df[text_column].apply(lambda x: str(x).count('!'))
    
    # Question marks
    df['question_count'] = df[text_column].apply(lambda x: str(x).count('?'))
    
    # Uppercase ratio
    df['uppercase_ratio'] = df[text_column].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    return df


def get_word_frequency_by_sentiment(df, text_column='cleaned_text', 
                                   sentiment_column='sentiment'):
    """
    Get word frequency distribution for each sentiment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    sentiment_column : str
        Name of the sentiment column
    
    Returns:
    --------
    dict
        Dictionary mapping sentiments to word frequency counters
    """
    word_freq = {}
    
    for sentiment in df[sentiment_column].unique():
        sentiment_texts = df[df[sentiment_column] == sentiment][text_column]
        all_words = ' '.join(sentiment_texts).split()
        word_freq[sentiment] = Counter(all_words)
    
    return word_freq


def create_feature_dataframe(feature_matrix, feature_names, index=None):
    """
    Convert feature matrix to DataFrame.
    
    Parameters:
    -----------
    feature_matrix : sparse matrix
        Feature matrix from vectorizer
    feature_names : array
        Feature names
    index : array-like, optional
        Index for the DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Feature DataFrame
    """
    df_features = pd.DataFrame(
        feature_matrix.toarray(),
        columns=feature_names,
        index=index
    )
    return df_features


if __name__ == "__main__":
    # Test feature engineering functions
    print("Feature engineering module loaded successfully!")
    print("Available functions:")
    print("- extract_keywords()")
    print("- create_tfidf_features()")
    print("- create_count_features()")
    print("- add_text_statistics()")
    print("- get_word_frequency_by_sentiment()")

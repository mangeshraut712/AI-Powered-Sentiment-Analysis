"""
Text Preprocessing Utilities
=============================

This module provides functions for cleaning and preprocessing text data.
"""

import re
import string
import pandas as pd
import neattext.functions as nfx
from textblob import TextBlob


def clean_text(text, remove_urls=True, remove_handles=True, remove_punctuation=True, 
               lowercase=True, remove_stopwords=False):
    """
    Clean and preprocess text data.
    
    Parameters:
    -----------
    text : str
        Input text to clean
    remove_urls : bool
        Remove URLs from text
    remove_handles : bool
        Remove Twitter handles (@username)
    remove_punctuation : bool
        Remove punctuation marks
    lowercase : bool
        Convert text to lowercase
    remove_stopwords : bool
        Remove common stopwords
    
    Returns:
    --------
    str
        Cleaned text
    
    Examples:
    ---------
    >>> text = "@user Check out https://example.com! #awesome"
    >>> clean_text(text)
    'check out awesome'
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = nfx.remove_urls(text)
    
    # Remove Twitter handles
    if remove_handles:
        text = nfx.remove_userhandles(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = nfx.remove_punctuations(text)
    
    # Remove stopwords
    if remove_stopwords:
        text = nfx.remove_stopwords(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def clean_dataframe(df, text_column='content', **kwargs):
    """
    Apply text cleaning to a DataFrame column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the column containing text
    **kwargs : dict
        Additional arguments to pass to clean_text()
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned text
    """
    df = df.copy()
    df['cleaned_text'] = df[text_column].apply(lambda x: clean_text(x, **kwargs))
    return df


def consolidate_sentiments(df, sentiment_column='sentiment', mapping=None):
    """
    Consolidate similar sentiment categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    sentiment_column : str
        Name of the sentiment column
    mapping : dict, optional
        Custom mapping for sentiment consolidation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with consolidated sentiments
    
    Examples:
    ---------
    >>> mapping = {'anger': 'anger', 'hate': 'anger', 'fun': 'happiness', 'happiness': 'happiness'}
    >>> df = consolidate_sentiments(df, mapping=mapping)
    """
    df = df.copy()
    
    if mapping is None:
        # Default mapping
        mapping = {
            'anger': 'anger',
            'hate': 'anger',
            'boredom': 'neutral',
            'empty': 'neutral',
            'neutral': 'neutral',
            'enthusiasm': 'happiness',
            'fun': 'happiness',
            'happiness': 'happiness',
            'relief': 'happiness',
            'love': 'love',
            'sadness': 'sadness',
            'surprise': 'surprise',
            'worry': 'worry'
        }
    
    df[sentiment_column] = df[sentiment_column].map(mapping)
    
    # Remove any rows with unmapped sentiments (NaN)
    df = df.dropna(subset=[sentiment_column])
    
    return df


def get_sentiment_polarity(text):
    """
    Get sentiment polarity using TextBlob.
    
    Parameters:
    -----------
    text : str
        Input text
    
    Returns:
    --------
    float
        Polarity score (-1 to 1)
    """
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except:
        return 0.0


def get_sentiment_subjectivity(text):
    """
    Get sentiment subjectivity using TextBlob.
    
    Parameters:
    -----------
    text : str
        Input text
    
    Returns:
    --------
    float
        Subjectivity score (0 to 1)
    """
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.subjectivity
    except:
        return 0.0


def add_textblob_features(df, text_column='cleaned_text'):
    """
    Add TextBlob sentiment features to DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added sentiment features
    """
    df = df.copy()
    df['polarity'] = df[text_column].apply(get_sentiment_polarity)
    df['subjectivity'] = df[text_column].apply(get_sentiment_subjectivity)
    return df


def preprocess_pipeline(df, text_column='content', sentiment_column='sentiment',
                       consolidate=True, add_textblob=True, verbose=True, **clean_kwargs):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    sentiment_column : str
        Name of the sentiment column
    consolidate : bool
        Whether to consolidate sentiments
    add_textblob : bool
        Whether to add TextBlob features
    verbose : bool
        Whether to print progress messages
    **clean_kwargs : dict
        Additional arguments for text cleaning
    
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed DataFrame
    """
    if verbose:
        print("Starting preprocessing pipeline...")
    
    # Clean text
    if verbose:
        print("1. Cleaning text...")
    df = clean_dataframe(df, text_column, **clean_kwargs)
    
    # Consolidate sentiments
    if consolidate:
        if verbose:
            print("2. Consolidating sentiments...")
        df = consolidate_sentiments(df, sentiment_column)
    
    # Add TextBlob features
    if add_textblob:
        if verbose:
            print("3. Adding TextBlob features...")
        df = add_textblob_features(df)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    if verbose:
        print(f"Preprocessing complete! Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Test preprocessing functions
    test_text = "@user Check out https://example.com! This is #awesome!!!"
    cleaned = clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    polarity = get_sentiment_polarity(cleaned)
    print(f"Polarity: {polarity}")

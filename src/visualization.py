"""
Visualization Utilities
=======================

This module provides functions for creating visualizations of sentiment analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_sentiment_distribution(df, sentiment_column='sentiment', save_path=None):
    """
    Plot the distribution of sentiments.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    sentiment_column : str
        Name of the sentiment column
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Count plot
    sentiment_counts = df[sentiment_column].value_counts()
    ax = sentiment_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    
    plt.title('Distribution of Sentiments', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(sentiment_counts):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_word_cloud(text_data, title='Word Cloud', save_path=None, max_words=100):
    """
    Generate and plot a word cloud.
    
    Parameters:
    -----------
    text_data : str or list
        Text data (single string or list of strings)
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the figure
    max_words : int
        Maximum number of words to display
    """
    if isinstance(text_data, list):
        text_data = ' '.join(text_data)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        relative_scaling=0.5
    ).generate(text_data)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_sentiment_word_clouds(df, text_column='cleaned_text', 
                               sentiment_column='sentiment', save_dir=None):
    """
    Generate word clouds for each sentiment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    text_column : str
        Name of the text column
    sentiment_column : str
        Name of the sentiment column
    save_dir : str or Path, optional
        Directory to save figures
    """
    sentiments = df[sentiment_column].unique()
    n_sentiments = len(sentiments)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_sentiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_sentiments > 1 else [axes]
    
    for idx, sentiment in enumerate(sentiments):
        sentiment_texts = df[df[sentiment_column] == sentiment][text_column]
        text_data = ' '.join(sentiment_texts)
        
        wordcloud = WordCloud(
            width=400,
            height=300,
            background_color='white',
            max_words=50,
            colormap='viridis'
        ).generate(text_data)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'{sentiment.capitalize()}', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_sentiments, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'sentiment_wordclouds.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Label names
    save_path : str or Path, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison of model metrics.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics as values
    save_path : str or Path, optional
        Path to save the figure
    """
    df_results = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    df_results.plot(kind='bar', ax=axes[0], edgecolor='black')
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].legend(title='Metrics')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Heatmap
    sns.heatmap(df_results, annot=True, fmt='.3f', cmap='YlGnBu', 
                ax=axes[1], cbar_kws={'label': 'Score'})
    axes[1].set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Metric', fontsize=12)
    axes[1].set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_text_statistics(df, save_path=None):
    """
    Plot text statistics distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with text statistics
    save_path : str or Path, optional
        Path to save the figure
    """
    stat_columns = ['char_count', 'word_count', 'avg_word_length']
    available_columns = [col for col in stat_columns if col in df.columns]
    
    if not available_columns:
        print("No text statistics found in DataFrame!")
        return
    
    fig, axes = plt.subplots(1, len(available_columns), figsize=(15, 4))
    if len(available_columns) == 1:
        axes = [axes]
    
    for idx, col in enumerate(available_columns):
        axes[idx].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_sentiment_polarity(df, sentiment_column='sentiment', save_path=None):
    """
    Plot sentiment polarity distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with polarity scores
    sentiment_column : str
        Name of the sentiment column
    save_path : str or Path, optional
        Path to save the figure
    """
    if 'polarity' not in df.columns:
        print("Polarity column not found in DataFrame!")
        return
    
    plt.figure(figsize=(12, 6))
    
    for sentiment in df[sentiment_column].unique():
        sentiment_data = df[df[sentiment_column] == sentiment]['polarity']
        plt.hist(sentiment_data, bins=30, alpha=0.5, label=sentiment, edgecolor='black')
    
    plt.title('Sentiment Polarity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Polarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Sentiment')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("- plot_sentiment_distribution()")
    print("- plot_word_cloud()")
    print("- plot_sentiment_word_clouds()")
    print("- plot_confusion_matrix()")
    print("- plot_model_comparison()")
    print("- plot_text_statistics()")
    print("- plot_sentiment_polarity()")

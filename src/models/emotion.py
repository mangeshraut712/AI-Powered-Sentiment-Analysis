"""
Machine Learning Models
=======================

This module provides machine learning model implementations for sentiment classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pickle
from pathlib import Path


class SentimentClassifier:
    """
    Base class for sentiment classification models.
    """
    
    def __init__(self, model_type='logistic_regression', **model_params):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('logistic_regression' or 'naive_bayes')
        **model_params : dict
            Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._create_model(**model_params)
        self.is_fitted = False
        
    def _create_model(self, **params):
        """Create the appropriate model based on model_type."""
        if self.model_type == 'logistic_regression':
            default_params = {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs',
                'multi_class': 'multinomial'
            }
            default_params.update(params)
            return LogisticRegression(**default_params)
        
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        verbose : bool
            Whether to print training information
        """
        if verbose:
            print(f"Training {self.model_type}...")
            print(f"Training set size: {X_train.shape}")
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        if verbose:
            print("Training complete!")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        
        Returns:
        --------
        array
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        
        Returns:
        --------
        array
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions!")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        verbose : bool
            Whether to print evaluation metrics
        
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if verbose:
            print(f"\n{self.model_type.upper()} Evaluation Metrics:")
            print("=" * 50)
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
        
        return metrics
    
    def cross_validate(self, X, y, cv=5, verbose=True):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels
        cv : int
            Number of cross-validation folds
        verbose : bool
            Whether to print results
        
        Returns:
        --------
        dict
            Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        if verbose:
            print(f"\nCross-Validation Results ({cv}-fold):")
            print("=" * 50)
            print(f"Scores: {scores}")
            print(f"Mean Accuracy: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an untrained model!")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_fitted = True
        print(f"Model loaded from: {filepath}")


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def compare_models(X_train, X_test, y_train, y_test, models=['logistic_regression', 'naive_bayes']):
    """
    Train and compare multiple models.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and testing features
    y_train, y_test : array-like
        Training and testing labels
    models : list
        List of model types to compare
    
    Returns:
    --------
    dict
        Dictionary of trained models and their metrics
    """
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print('='*60)
        
        classifier = SentimentClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics
        }
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)
    
    comparison_df = pd.DataFrame({
        model: result['metrics'] 
        for model, result in results.items()
    }).T
    
    print(comparison_df)
    
    return results


if __name__ == "__main__":
    print("Machine Learning Models module loaded successfully!")
    print("Available classes and functions:")
    print("- SentimentClassifier")
    print("- train_test_split_data()")
    print("- compare_models()")

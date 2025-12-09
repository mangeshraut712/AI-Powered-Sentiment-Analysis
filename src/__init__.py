"""
DSCI-521 Sentiment Analysis Package
====================================

This package provides utilities for sentiment analysis and emotion detection from text.

Modules:
    - data_loader: Functions for loading and managing datasets
    - preprocessing: Text cleaning and preprocessing utilities
    - feature_engineering: Feature extraction and transformation
    - models: Machine learning model implementations
    - visualization: Plotting and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "DSCI-521 Group (Mangesh Raut, Josh Clark, Will Wu, Mobin Rahimi)"

from . import data_loader
from . import preprocessing
from . import feature_engineering
from . import models
from . import visualization

__all__ = [
    'data_loader',
    'preprocessing',
    'feature_engineering',
    'models',
    'visualization'
]

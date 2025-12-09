# Project Methodology

## Overview

This document describes the methodology used in the DSCI-521 Sentiment Analysis project.

---

## 1. Data Collection

### Dataset Source
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text
- **Size:** 40,000 tweets
- **Format:** CSV file

### Dataset Structure
```
Columns:
- tweet_id: Unique identifier for each tweet
- sentiment: Emotion label (13 categories)
- content: Tweet text content
```

---

## 2. Data Preprocessing

### 2.1 Text Cleaning
The following preprocessing steps were applied:

1. **URL Removal:** Remove all URLs using regex patterns
2. **Handle Removal:** Remove Twitter handles (@username)
3. **Lowercase Conversion:** Convert all text to lowercase
4. **Punctuation Removal:** Remove punctuation marks
5. **Whitespace Normalization:** Remove extra spaces

### 2.2 Sentiment Consolidation
Original 13 emotion categories were consolidated into 6 main categories:

| Original Emotions | Consolidated Category |
|-------------------|----------------------|
| anger, hate | anger |
| boredom, empty, neutral | neutral |
| enthusiasm, fun, happiness, relief | happiness |
| love | love |
| sadness | sadness |
| surprise | surprise |
| worry | worry |

**Rationale:** Address class imbalance and group semantically similar emotions.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Distribution Analysis
- Analyzed frequency distribution of emotions
- Identified class imbalance issues
- Visualized distributions using bar charts

### 3.2 Text Statistics
Computed the following statistics:
- Character count per tweet
- Word count per tweet
- Average word length
- Special character usage (!, ?)

### 3.3 Keyword Extraction
- Extracted top keywords for each sentiment
- Used both frequency-based and TF-IDF methods
- Generated word clouds for visualization

---

## 4. Feature Engineering

### 4.1 TF-IDF Vectorization
```python
Parameters:
- max_features: 1000
- ngram_range: (1, 2)  # Unigrams and bigrams
- min_df: 2
- stop_words: 'english'
```

### 4.2 TextBlob Sentiment Features
Added two additional features:
- **Polarity:** Sentiment polarity score (-1 to 1)
- **Subjectivity:** Subjectivity score (0 to 1)

### 4.3 Statistical Features
- Text length metrics
- Punctuation counts
- Uppercase ratio

---

## 5. Model Development

### 5.1 Train-Test Split
```python
- Test size: 20%
- Random state: 42
- Stratified split: Yes (to maintain class distribution)
```

### 5.2 Models Implemented

#### Logistic Regression
```python
Parameters:
- solver: 'lbfgs'
- multi_class: 'multinomial'
- max_iter: 1000
- random_state: 42
```

**Advantages:**
- Fast training
- Interpretable coefficients
- Good baseline performance
- Handles multi-class naturally

#### Naive Bayes (Multinomial)
```python
Parameters:
- Default parameters
```

**Advantages:**
- Fast training and prediction
- Works well with text data
- Probabilistic interpretation
- Handles high-dimensional data

---

## 6. Model Evaluation

### 6.1 Metrics Used
1. **Accuracy:** Overall correctness
2. **Precision:** Positive predictive value (weighted average)
3. **Recall:** Sensitivity (weighted average)
4. **F1-Score:** Harmonic mean of precision and recall

### 6.2 Cross-Validation
- **Method:** 5-fold cross-validation
- **Purpose:** Assess model generalization
- **Metric:** Accuracy

### 6.3 Confusion Matrix
- Visualize classification performance
- Identify common misclassifications
- Analyze per-class performance

---

## 7. Keyword Spotting Method

The keyword spotting approach involves:

1. **Tokenization:** Split text into individual words
2. **Keyword Identification:** Extract significant words for each emotion
3. **Pattern Matching:** Identify emotion-specific vocabulary patterns
4. **Classification:** Use ML models to classify based on keyword presence

---

## 8. Challenges and Solutions

### Challenge 1: Class Imbalance
**Problem:** Some emotions have far more examples than others

**Solutions:**
- Sentiment consolidation
- Stratified train-test split
- Weighted metrics

### Challenge 2: Noisy Text Data
**Problem:** Tweets contain URLs, handles, special characters

**Solutions:**
- Comprehensive text cleaning pipeline
- Regular expression-based cleaning
- Standardization of text format

### Challenge 3: Context and Sarcasm
**Problem:** Difficult to detect sarcasm and context-dependent emotions

**Solutions:**
- Feature engineering (punctuation, capitalization)
- N-gram features to capture context
- Future work: Deep learning models

---

## 9. Validation Strategy

1. **Hold-out Validation:** 80-20 train-test split
2. **Cross-Validation:** 5-fold CV for robustness
3. **Stratification:** Maintain class distribution
4. **Multiple Metrics:** Comprehensive evaluation

---

## 10. Reproducibility

To ensure reproducibility:
- Fixed random seeds (random_state=42)
- Documented all parameters
- Version-controlled code
- Saved trained models
- Detailed requirements.txt

---

## References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. TextBlob Documentation: https://textblob.readthedocs.io/
3. Kaggle Dataset: https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text

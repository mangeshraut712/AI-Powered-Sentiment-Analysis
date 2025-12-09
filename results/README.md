# Results Directory

This directory stores outputs from model training and analysis.

---

## Structure

```
results/
├── figures/        # Generated plots and visualizations
├── models/         # Saved trained models
└── metrics/        # Performance metrics and reports
```

---

## Subdirectories

### figures/
**Purpose:** Store all generated visualizations

**Typical contents:**
- `sentiment_distribution.png` - Bar chart of emotion frequencies
- `wordcloud_*.png` - Word clouds for each sentiment
- `confusion_matrix_*.png` - Model confusion matrices
- `model_comparison.png` - Performance comparison charts
- `text_statistics.png` - Text feature distributions

**Naming convention:** `descriptive_name_YYYYMMDD.png`

---

### models/
**Purpose:** Store trained machine learning models

**Typical contents:**
- `logistic_regression_model.pkl` - Trained Logistic Regression
- `naive_bayes_model.pkl` - Trained Naive Bayes
- `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer
- `label_encoder.pkl` - Label encoder (if used)

**Naming convention:** `model_type_YYYYMMDD.pkl`

**Note:** Model files are gitignored due to size. Document model parameters separately.

---

### metrics/
**Purpose:** Store performance metrics and evaluation reports

**Typical contents:**
- `classification_report.txt` - Detailed classification metrics
- `cross_validation_scores.csv` - CV results
- `model_comparison.csv` - Comparison of different models
- `feature_importance.csv` - Important features

**Naming convention:** `metric_type_YYYYMMDD.csv`

---

## Usage Examples

### Saving Figures
```python
from src.visualization import plot_sentiment_distribution

plot_sentiment_distribution(
    df, 
    save_path='results/figures/sentiment_dist_20241209.png'
)
```

### Saving Models
```python
from src.models import SentimentClassifier

classifier = SentimentClassifier(model_type='logistic_regression')
classifier.train(X_train, y_train)
classifier.save_model('results/models/logistic_regression_20241209.pkl')
```

### Loading Models
```python
classifier = SentimentClassifier(model_type='logistic_regression')
classifier.load_model('results/models/logistic_regression_20241209.pkl')
predictions = classifier.predict(X_test)
```

### Saving Metrics
```python
import pandas as pd

metrics_df = pd.DataFrame({
    'model': ['Logistic Regression', 'Naive Bayes'],
    'accuracy': [0.85, 0.82],
    'f1_score': [0.84, 0.81]
})

metrics_df.to_csv('results/metrics/model_comparison_20241209.csv', index=False)
```

---

## Best Practices

1. **Use timestamps** in filenames for versioning
2. **Document parameters** used to generate each result
3. **Keep a log** of experiments and their outcomes
4. **Save high-resolution** figures (dpi=300)
5. **Compress large files** if necessary
6. **Clean up old results** periodically

---

## Gitignore Note

The following file types are gitignored:
- `*.pkl` - Model files (too large)
- `*.joblib` - Joblib saved models
- `*.h5` - Deep learning models

To track specific files, add them explicitly:
```bash
git add -f results/models/important_model.pkl
```

---

## Results Organization Tips

### For Each Experiment:
1. Create a dated subdirectory: `results/experiment_20241209/`
2. Save all related outputs together
3. Include a README or notes file
4. Document hyperparameters and settings

### Example Structure:
```
results/
└── experiment_20241209/
    ├── README.txt
    ├── figures/
    │   ├── confusion_matrix.png
    │   └── wordclouds.png
    ├── models/
    │   └── best_model.pkl
    └── metrics/
        └── performance.csv
```

---

**Last Updated:** December 2024

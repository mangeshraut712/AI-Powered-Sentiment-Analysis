# Quick Start Guide

Welcome to the DSCI-521 Sentiment Analysis Project! This guide will help you get started quickly.

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd /Users/mangeshraut/Downloads/DSCI_521_Group_Project
pip install -r requirements.txt
```

### Step 2: Download NLTK Data (Optional)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Step 3: Verify Data
```bash
ls data/raw/tweet_emotions.csv
```

---

## 📊 Quick Analysis (10 minutes)

### Option 1: Use Existing Notebooks

**Open Jupyter:**
```bash
jupyter notebook
```

**Navigate to:** `notebooks/02_main_analysis_josh.ipynb`

**Run all cells:** Cell → Run All

---

### Option 2: Quick Python Script

Create a file `quick_test.py`:

```python
import sys
sys.path.append('.')

from src.data_loader import load_dataset
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import create_tfidf_features
from src.models import SentimentClassifier, train_test_split_data
from src.visualization import plot_sentiment_distribution

# 1. Load data
print("Loading data...")
df = load_dataset()

# 2. Preprocess
print("\nPreprocessing...")
df_clean = preprocess_pipeline(df, consolidate=True)

# 3. Visualize
print("\nVisualizing distribution...")
plot_sentiment_distribution(df_clean)

# 4. Create features
print("\nCreating features...")
X, vectorizer, feature_names = create_tfidf_features(df_clean, max_features=500)
y = df_clean['sentiment']

# 5. Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# 6. Train model
print("\nTraining model...")
classifier = SentimentClassifier(model_type='logistic_regression')
classifier.train(X_train, y_train)

# 7. Evaluate
print("\nEvaluating model...")
metrics = classifier.evaluate(X_test, y_test)

print("\n✅ Quick analysis complete!")
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

**Run it:**
```bash
python quick_test.py
```

---

## 🎯 Common Tasks

### Task 1: Load and Explore Data
```python
from src.data_loader import load_dataset, get_dataset_info

df = load_dataset()
info = get_dataset_info(df)
print(info)
```

### Task 2: Clean Text
```python
from src.preprocessing import clean_text

text = "@user Check out https://example.com! #awesome"
cleaned = clean_text(text)
print(cleaned)  # Output: "check out awesome"
```

### Task 3: Extract Keywords
```python
from src.feature_engineering import extract_keywords

keywords = extract_keywords(df_clean, top_n=10)
for sentiment, words in keywords.items():
    print(f"{sentiment}: {words}")
```

### Task 4: Train a Model
```python
from src.models import SentimentClassifier

classifier = SentimentClassifier(model_type='naive_bayes')
classifier.train(X_train, y_train)
metrics = classifier.evaluate(X_test, y_test)
```

### Task 5: Create Visualizations
```python
from src.visualization import plot_word_cloud, plot_confusion_matrix

# Word cloud
plot_word_cloud(df_clean['cleaned_text'], title='All Tweets')

# Confusion matrix
y_pred = classifier.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
```

---

## 📚 Learning Path

### Beginner Path
1. Read `README.md` (main project overview)
2. Explore `notebooks/03_project_scoping_eda.ipynb`
3. Run the quick test script above
4. Read `docs/methodology.md`

### Intermediate Path
1. Study `notebooks/02_main_analysis_josh.ipynb`
2. Experiment with different models
3. Try different preprocessing options
4. Create custom visualizations

### Advanced Path
1. Review `notebooks/01_main_analysis_group_2022.ipynb`
2. Implement new features
3. Try ensemble methods
4. Explore deep learning approaches

---

## 🔧 Troubleshooting

### Issue: Import errors
```python
# Add project to path
import sys
sys.path.append('/Users/mangeshraut/Downloads/DSCI_521_Group_Project')
```

### Issue: Data not found
```python
# Check current directory
import os
print(os.getcwd())

# Use absolute path
df = load_dataset('/Users/mangeshraut/Downloads/DSCI_521_Group_Project/data/raw/tweet_emotions.csv')
```

### Issue: Memory error
```python
# Use sampling
df_sample = df.sample(n=5000, random_state=42)

# Or reduce features
X, vec, names = create_tfidf_features(df_clean, max_features=100)
```

---

## 🎓 Next Steps

After completing the quick start:

1. **Explore the notebooks** in detail
2. **Read the methodology** documentation
3. **Experiment** with different parameters
4. **Create your own** analysis
5. **Share your findings**

---

## 💡 Tips

- **Start small:** Use data sampling for quick experiments
- **Save your work:** Save models and processed data
- **Document changes:** Keep notes on what works
- **Ask questions:** Review the documentation
- **Have fun:** Experiment and learn!

---

## 📞 Getting Help

- Check `README.md` files in each directory
- Review `docs/methodology.md` for detailed explanations
- Look at example notebooks for reference
- Read function docstrings: `help(function_name)`

---

**Ready to start? Run the quick test script and explore!** 🚀

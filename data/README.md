# Data Directory

This directory contains all datasets used in the sentiment analysis project.

---

## Structure

```
data/
├── raw/                    # Original, unmodified data
│   └── tweet_emotions.csv
└── processed/              # Cleaned and processed data
    └── (generated files)
```

---

## Dataset Information

### tweet_emotions.csv

**Source:** [Kaggle - Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)

**Description:** A dataset of 40,000 tweets labeled with emotions

**Size:** ~3.6 MB

**Format:** CSV (Comma-Separated Values)

**Columns:**
- `tweet_id` (int): Unique identifier for each tweet
- `sentiment` (str): Emotion label
- `content` (str): Tweet text content

**Emotion Labels:**
- anger
- boredom
- empty
- enthusiasm
- fun
- happiness
- hate
- love
- neutral
- relief
- sadness
- surprise
- worry

**Sample Data:**
```
tweet_id,sentiment,content
1956967341,sadness,"@Heatheranneh5 I know I was really sad when I heard"
1956967666,sadness,"@Heatheranneh5 I know I was really sad when I heard"
...
```

---

## Data Usage

### Loading the Data

**Using the data_loader module:**
```python
from src.data_loader import load_dataset

df = load_dataset()
```

**Using pandas directly:**
```python
import pandas as pd

df = pd.read_csv('data/raw/tweet_emotions.csv')
```

---

## Processed Data

Processed datasets are saved to `data/processed/` after preprocessing.

**Typical processed files:**
- `processed_tweets.csv` - Cleaned and consolidated data
- `train_data.csv` - Training set
- `test_data.csv` - Test set
- `features_tfidf.csv` - TF-IDF features

---

## Data Statistics

**Original Dataset:**
- Total tweets: 40,000
- Unique emotions: 13
- Average tweet length: ~50 characters
- Date range: 2009 (Twitter data)

**After Consolidation:**
- Consolidated emotions: 6-7
- Filtered tweets: ~28,000-35,000 (depending on approach)

---

## Data Quality

### Known Issues:
1. **Class Imbalance:** Some emotions have significantly more examples
2. **Noise:** Contains URLs, handles, special characters
3. **Duplicates:** Some tweets may be duplicated
4. **Language:** Primarily English, but may contain other languages

### Preprocessing Recommendations:
- Remove URLs and Twitter handles
- Convert to lowercase
- Remove punctuation
- Handle class imbalance (consolidation or resampling)

---

## Citation

If you use this dataset, please cite:

```
Pashupati Gupta. (2021). Emotion Detection from Text. 
Kaggle. https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text
```

---

## Privacy and Ethics

- This is publicly available Twitter data
- Tweets are anonymized (only tweet_id, no user information)
- Use responsibly and ethically
- Follow Twitter's Terms of Service
- Consider privacy implications when sharing results

---

## Adding New Data

To add new datasets:

1. Place raw data in `data/raw/`
2. Document the data source and format
3. Update this README
4. Create preprocessing scripts if needed
5. Save processed data to `data/processed/`

---

## Gitignore Note

Large CSV files are gitignored by default. To track specific files, add them explicitly:

```bash
git add -f data/raw/sample_data.csv
```

---

**Last Updated:** December 2024

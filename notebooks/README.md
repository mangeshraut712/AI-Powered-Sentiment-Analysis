# Notebooks Guide

This directory contains Jupyter notebooks for the sentiment analysis project.

---

## Notebook Overview

### 01_main_analysis_group_2022.ipynb
**Description:** Main group project submission (Summer 2022)

**Contents:**
- Complete sentiment analysis pipeline
- Data loading and exploration
- Sentiment consolidation
- Feature extraction
- Model training (Naive Bayes, Logistic Regression)
- Evaluation and visualization

**Authors:** Mangesh Raut, Josh Clark, Will Wu, Mobin Rahimi

---

### 02_main_analysis_josh.ipynb
**Description:** Individual project submission by Josh Clark

**Contents:**
- Alternative approach to sentiment analysis
- Focus on Logistic Regression
- Detailed preprocessing pipeline
- Comprehensive evaluation metrics
- Future work discussion

**Author:** Josh Clark

---

### 03_project_scoping_eda.ipynb
**Description:** Project scoping and exploratory data analysis

**Contents:**
- Initial data exploration
- Problem definition
- Dataset characteristics
- Preliminary visualizations
- Project planning

---

## How to Use These Notebooks

### Prerequisites
```bash
# Install required packages
pip install -r ../requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Running the Notebooks

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open a notebook** from the file browser

3. **Run cells sequentially** using Shift+Enter

4. **Restart kernel** if needed: Kernel → Restart & Clear Output

---

## Notebook Execution Order

For new users, we recommend following this order:

1. **Start with:** `03_project_scoping_eda.ipynb`
   - Understand the problem and data

2. **Then review:** `02_main_analysis_josh.ipynb`
   - See a clean, well-documented approach

3. **Finally explore:** `01_main_analysis_group_2022.ipynb`
   - See the complete group implementation

---

## Data Requirements

All notebooks expect the dataset to be located at:
```
../data/raw/tweet_emotions.csv
```

If you encounter file not found errors, update the data path in the notebook.

---

## Creating New Notebooks

When creating new analysis notebooks:

1. **Use the src package:**
```python
import sys
sys.path.append('..')

from src import data_loader, preprocessing, models, visualization
```

2. **Follow naming convention:**
```
XX_descriptive_name.ipynb
```
Where XX is a sequential number (04, 05, etc.)

3. **Include sections:**
- Introduction/Objective
- Data Loading
- Analysis/Processing
- Results
- Conclusions

4. **Add markdown cells** for explanations

5. **Clear outputs** before committing (optional)

---

## Tips for Best Results

### Memory Management
- Clear large variables when done: `del large_dataframe`
- Restart kernel periodically
- Use sampling for quick tests

### Reproducibility
- Set random seeds: `np.random.seed(42)`
- Document package versions
- Save intermediate results

### Visualization
- Use `%matplotlib inline` for inline plots
- Save important figures to `../results/figures/`
- Use descriptive titles and labels

---

## Common Issues and Solutions

### Issue: ModuleNotFoundError
**Solution:** 
```python
import sys
sys.path.append('..')
```

### Issue: Data file not found
**Solution:** Check the data path:
```python
import os
print(os.getcwd())  # Check current directory
```

### Issue: Kernel dies during training
**Solution:** Reduce max_features or use sampling:
```python
df_sample = df.sample(n=5000, random_state=42)
```

---

## Additional Resources

- **Jupyter Documentation:** https://jupyter.org/documentation
- **Markdown Guide:** https://www.markdownguide.org/
- **Pandas Cheat Sheet:** https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

---

## Contributing

When adding new notebooks:
1. Follow the naming convention
2. Add documentation in this README
3. Test the notebook from start to finish
4. Clear sensitive outputs
5. Update the execution order if relevant

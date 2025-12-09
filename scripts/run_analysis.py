#!/usr/bin/env python3
"""
Complete Sentiment Analysis Demo
=================================

This script runs a complete sentiment analysis pipeline demonstrating all
features of the DSCI-521 project.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Import our modules
from src.data_loader import load_dataset, get_dataset_info, save_processed_data
from src.preprocessing import preprocess_pipeline, clean_text
from src.feature_engineering import (
    create_tfidf_features, 
    extract_keywords,
    add_text_statistics
)
from src.models import (
    EmotionClassifier as SentimentClassifier, 
    train_test_split_data,
    compare_models
)
from src.visualization import (
    plot_sentiment_distribution,
    plot_word_cloud,
    plot_confusion_matrix,
    plot_model_comparison
)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    """Run the complete analysis pipeline."""
    
    print("\n" + "🚀 DSCI-521 Sentiment Analysis - Complete Demo".center(70))
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print_section("STEP 1: Loading Dataset")
    
    df = load_dataset(verbose=False)
    print(f"✅ Dataset loaded: {df.shape[0]} tweets, {df.shape[1]} columns")
    
    # Show basic info
    info = get_dataset_info(df)
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Total tweets: {info['shape'][0]:,}")
    print(f"   - Columns: {', '.join(info['columns'])}")
    print(f"   - Unique emotions: {len(info['sentiment_counts'])}")
    print(f"   - Memory usage: {info['memory_usage']:.2f} MB")
    
    print(f"\n🎭 Emotion Distribution:")
    for emotion, count in sorted(info['sentiment_counts'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"   - {emotion:15s}: {count:5,} tweets")
    
    # ========================================================================
    # STEP 2: Preprocess Data
    # ========================================================================
    print_section("STEP 2: Preprocessing Text Data")
    
    print("🧹 Cleaning text and consolidating emotions...")
    df_clean = preprocess_pipeline(
        df, 
        text_column='content',
        sentiment_column='sentiment',
        consolidate=True,
        add_textblob=True,
        verbose=False
    )
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Original emotions: 13")
    print(f"   - Consolidated emotions: {df_clean['sentiment'].nunique()}")
    print(f"   - Tweets after filtering: {len(df_clean):,}")
    
    # Show sample cleaned text
    print(f"\n📝 Sample Cleaned Text:")
    sample_idx = df_clean.sample(3, random_state=42).index
    for idx in sample_idx:
        original = df.loc[idx, 'content'][:60]
        cleaned = df_clean.loc[idx, 'cleaned_text'][:60]
        emotion = df_clean.loc[idx, 'sentiment']
        print(f"\n   Emotion: {emotion}")
        print(f"   Original: {original}...")
        print(f"   Cleaned:  {cleaned}...")
    
    # ========================================================================
    # STEP 3: Exploratory Data Analysis
    # ========================================================================
    print_section("STEP 3: Exploratory Data Analysis")
    
    # Add text statistics
    print("📊 Computing text statistics...")
    df_clean = add_text_statistics(df_clean)
    
    # Show statistics
    print(f"\n📈 Text Statistics Summary:")
    stats_cols = ['char_count', 'word_count', 'avg_word_length']
    for col in stats_cols:
        if col in df_clean.columns:
            print(f"   - {col:20s}: mean={df_clean[col].mean():.2f}, "
                  f"median={df_clean[col].median():.2f}")
    
    # Extract keywords
    print(f"\n🔑 Extracting top keywords per emotion...")
    keywords = extract_keywords(df_clean, top_n=5, method='frequency')
    
    print(f"\n💬 Top Keywords by Emotion:")
    for emotion, words in sorted(keywords.items()):
        print(f"   - {emotion:15s}: {', '.join(words[:5])}")
    
    # ========================================================================
    # STEP 4: Feature Engineering
    # ========================================================================
    print_section("STEP 4: Feature Engineering")
    
    print("🔧 Creating TF-IDF features...")
    X, vectorizer, feature_names = create_tfidf_features(
        df_clean,
        max_features=500,
        ngram_range=(1, 2),
        min_df=2
    )
    y = df_clean['sentiment']
    
    print(f"✅ Features created!")
    print(f"   - Feature matrix shape: {X.shape}")
    print(f"   - Number of features: {len(feature_names)}")
    print(f"   - Feature type: TF-IDF (unigrams + bigrams)")
    print(f"   - Sample features: {', '.join(feature_names[:5])}")
    
    # ========================================================================
    # STEP 5: Train-Test Split
    # ========================================================================
    print_section("STEP 5: Splitting Data")
    
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Data split complete!")
    print(f"   - Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df_clean)*100:.1f}%)")
    print(f"   - Test set:     {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df_clean)*100:.1f}%)")
    print(f"   - Features:     {X_train.shape[1]:,}")
    
    # ========================================================================
    # STEP 6: Model Training
    # ========================================================================
    print_section("STEP 6: Training Machine Learning Models")
    
    print("\n🤖 Training Logistic Regression...")
    lr_classifier = SentimentClassifier(model_type='logistic_regression')
    lr_classifier.train(X_train, y_train, verbose=False)
    print("✅ Logistic Regression trained!")
    
    print("\n🤖 Training Naive Bayes...")
    nb_classifier = SentimentClassifier(model_type='naive_bayes')
    nb_classifier.train(X_train, y_train, verbose=False)
    print("✅ Naive Bayes trained!")
    
    # ========================================================================
    # STEP 7: Model Evaluation
    # ========================================================================
    print_section("STEP 7: Evaluating Models")
    
    print("\n📊 Logistic Regression Performance:")
    lr_metrics = lr_classifier.evaluate(X_test, y_test, verbose=False)
    for metric, value in lr_metrics.items():
        print(f"   - {metric.capitalize():15s}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n📊 Naive Bayes Performance:")
    nb_metrics = nb_classifier.evaluate(X_test, y_test, verbose=False)
    for metric, value in nb_metrics.items():
        print(f"   - {metric.capitalize():15s}: {value:.4f} ({value*100:.2f}%)")
    
    # Compare models
    print("\n🏆 Model Comparison:")
    comparison = pd.DataFrame({
        'Logistic Regression': lr_metrics,
        'Naive Bayes': nb_metrics
    }).T
    print(comparison.to_string())
    
    # Determine best model
    best_model_name = 'Logistic Regression' if lr_metrics['accuracy'] > nb_metrics['accuracy'] else 'Naive Bayes'
    best_accuracy = max(lr_metrics['accuracy'], nb_metrics['accuracy'])
    print(f"\n🥇 Best Model: {best_model_name} (Accuracy: {best_accuracy:.2%})")
    
    # ========================================================================
    # STEP 8: Predictions
    # ========================================================================
    print_section("STEP 8: Making Predictions")
    
    # Use best model
    best_model = lr_classifier if best_model_name == 'Logistic Regression' else nb_classifier
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Show some predictions
    print(f"\n🔮 Sample Predictions (using {best_model_name}):")
    sample_indices = np.random.choice(len(y_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        text = df_clean.iloc[y_test.index[idx]]['cleaned_text'][:50]
        match = "✅" if actual == predicted else "❌"
        
        print(f"\n   Example {i} {match}:")
        print(f"   Text:      {text}...")
        print(f"   Actual:    {actual}")
        print(f"   Predicted: {predicted}")
    
    # ========================================================================
    # STEP 9: Save Results
    # ========================================================================
    print_section("STEP 9: Saving Results")
    
    # Save processed data
    output_path = Path('data/processed/processed_tweets.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")
    
    # Save models
    lr_model_path = Path('results/models/logistic_regression_model.pkl')
    lr_classifier.save_model(lr_model_path)
    print(f"✅ Logistic Regression model saved to: {lr_model_path}")
    
    nb_model_path = Path('results/models/naive_bayes_model.pkl')
    nb_classifier.save_model(nb_model_path)
    print(f"✅ Naive Bayes model saved to: {nb_model_path}")
    
    # Save vectorizer
    vectorizer_path = Path('results/models/tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ TF-IDF Vectorizer saved to: {vectorizer_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Logistic Regression': lr_metrics,
        'Naive Bayes': nb_metrics
    })
    metrics_path = Path('results/metrics/model_comparison.csv')
    metrics_df.to_csv(metrics_path)
    print(f"✅ Metrics saved to: {metrics_path}")
    
    # Save keywords
    keywords_df = pd.DataFrame([
        {'emotion': emotion, 'keywords': ', '.join(words)}
        for emotion, words in keywords.items()
    ])
    keywords_path = Path('results/metrics/keywords_by_emotion.csv')
    keywords_df.to_csv(keywords_path, index=False)
    print(f"✅ Keywords saved to: {keywords_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("✨ Analysis Complete!")
    
    print(f"""
📊 Summary:
   - Dataset: {len(df):,} tweets analyzed
   - Emotions: {df_clean['sentiment'].nunique()} categories
   - Features: {X.shape[1]:,} TF-IDF features
   - Models trained: 2 (Logistic Regression, Naive Bayes)
   - Best model: {best_model_name}
   - Best accuracy: {best_accuracy:.2%}
   
📁 Outputs saved to:
   - Processed data: data/processed/
   - Models: results/models/
   - Metrics: results/metrics/
   
🎯 Next Steps:
   1. Review saved results in results/ directory
   2. Explore notebooks for detailed analysis
   3. Try predictions on new text
   4. Visualize results (run with --visualize flag)
   
✅ Project successfully executed!
""")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

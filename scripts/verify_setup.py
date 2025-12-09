#!/usr/bin/env python3
"""
Project Setup Verification Script
==================================

This script verifies that the project is properly set up and all dependencies are installed.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version is too old. Please upgrade to 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'textblob',
        'neattext',
        'nltk',
        'wordcloud'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - NOT INSTALLED")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def check_project_structure():
    """Check if project directories exist."""
    print_header("Checking Project Structure")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'src',
        'notebooks',
        'results/figures',
        'results/models',
        'results/metrics',
        'docs',
        'results/metrics',
        'docs'
    ]
    
    project_root = Path(__file__).parent.parent
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("\n✅ All required directories exist!")
        return True

def check_data_files():
    """Check if data files exist."""
    print_header("Checking Data Files")
    
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'raw' / 'tweet_emotions.csv'
    
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"✅ tweet_emotions.csv found ({size_mb:.2f} MB)")
        return True
    else:
        print("❌ tweet_emotions.csv NOT FOUND")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text")
        return False

def check_source_files():
    """Check if source code files exist."""
    print_header("Checking Source Code Files")
    
    project_root = Path(__file__).parent.parent
    src_files = [
        'src/__init__.py',
        'src/data_loader.py',
        'src/preprocessing.py',
        'src/feature_engineering.py',
        'src/models/__init__.py',
        'src/visualization.py'
    ]
    
    missing = []
    
    for file_path in src_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All source files exist!")
        return True

def test_imports():
    """Test importing the project modules."""
    print_header("Testing Module Imports")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src import data_loader
        print("✅ src.data_loader")
        
        from src import preprocessing
        print("✅ src.preprocessing")
        
        from src import feature_engineering
        print("✅ src.feature_engineering")
        
        from src import models
        print("✅ src.models")
        
        from src import visualization
        print("✅ src.visualization")
        
        print("\n✅ All modules can be imported successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "🔍 DSCI-521 Project Setup Verification".center(60))
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Data Files", check_data_files),
        ("Source Files", check_source_files),
        ("Module Imports", test_imports)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Project setup is complete! You're ready to start.")
        print("\nNext steps:")
        print("1. Read docs/QUICK_START.md")
        print("2. Open notebooks/02_main_analysis_josh.ipynb")
        print("3. Run the analysis!")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# 🎭 Unified Sentiment & Emotion Analysis System

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Next.js](https://img.shields.io/badge/Next.js-16.0-black)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Tailwind](https://img.shields.io/badge/Tailwind-CSS-3.4-38bdf8)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **A state-of-the-art Natural Language Processing system combining binary sentiment classification (Naive Bayes) and multi-class emotion detection (Logistic Regression) into a unified, interactive full-stack application.**

## � Executive Summary

This project represents the convergence of two distinct NLP initiatives into a single, cohesive platform. It bridges the gap between raw data analysis and user-centric application design, providing researchers and developers with a powerful toolset for text analysis.

**Key capabilities:**
-   **Dual-Model Intelligence:** Seamlessly switch between identifying *Subjective Sentiment* (Positive/Negative) and *Granular Emotion* (Happiness, Sadness, Anger, etc.).
-   **Real-Time Visualization:** Interactive charts fueled by live training data statistics, powered by Recharts and Framer Motion.
-   **Modern Architecture:** A decoupled architecture featuring a robust Python/Flask backend API and a high-performance Next.js 16 frontend.

---

## 🛠️ Technology Stack & Advancements

### Frontend (User Experience)
-   **Framework:** **Next.js 16 (App Router)** for server-side rendering and optimal performance.
-   **Styling:** **Tailwind CSS** with a custom "Apple-inspired" minimalist design system (Glassmorphism, clean typography).
-   **Animation:** **Framer Motion** for fluid, physics-based UI transitions.
-   **Visualization:** **Recharts** for responsive, animated data charting.

### Backend (Intelligence)
-   **API Server:** **Flask** providing a RESTful interface for model inference and statistics.
-   **Machine Learning:** **Scikit-learn** implementation of Multinomial Naive Bayes and Logistic Regression.
-   **Processing:** Advanced text cleaning pipeline (NLTK/Regex) with support for N-grams and Stopword removal.

---

## 🏗️ System Architecture

```mermaid
graph LR
    User[User Interface] <-->|HTTP/JSON| NextJS[Next.js Frontend]
    NextJS <-->|API Calls| Flask[Flask Backend (Port 5001)]
    Flask -->|Inference| Binary[Binary Model (Naive Bayes)]
    Flask -->|Inference| Emotion[Emotion Model (LogReg)]
    Binary -->|Stats| Data[Training Data]
```

---

## 🚀 Key Features

### 1. Unified CLI Tool
Access all capabilities from the terminal.
```bash
# Detect Emotion
python cli.py emotion --text "I am feeling wonderful today!"

# Check Sentiment
python cli.py classify --text "This product is a complete failure."
```

### 2. Interactive Web Application
A stunning, dark-mode enabled web interface.
-   **Live Demo:** Type text and get instant classification results for both Sentiment and Emotion.
-   **Dataset Insights:** Visualize the balance of training data (Positive vs Negative) and inspect top feature words in real-time.

---

## 🏁 Getting Started

### Prerequisites
-   Python 3.8+
-   Node.js 18+

### 1. Installation

**Python Environment:**
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

**Web Dependencies:**
```bash
cd web
npm install
cd ..
```

### 2. Running the System

To experience the full application, run the Backend and Frontend simultaneously.

**Terminal 1: Python Backend**
```bash
python src/app.py
# Server starts on http://localhost:5001
```

**Terminal 2: Web Frontend**
```bash
cd web
npm run dev
# App accessible at http://localhost:3000
```

### 3. Running the CLI Demo
Verify the installation quickly:
```bash
python demo.py
```

---

## � Project Structure

```
├── cli.py                  # Unified Command Line Interface
├── demo.py                 # Quick verification script
├── src/
│   ├── app.py              # Flask API Server
│   ├── models/             # ML Model Implementations
│   │   ├── binary.py       # Naive Bayes Classifier
│   │   └── emotion.py      # Logistic Regression Classifier
│   ├── preprocessing.py    # NLP Pipeline
│   └── visualization.py    # Plotting Utilities
├── web/
│   ├── src/app/page.tsx    # Main UI Dashboard
│   └── ...
├── data/                   # Datasets (Raw & Processed)
└── scripts/                # Utility Scripts (Training, etc.)
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

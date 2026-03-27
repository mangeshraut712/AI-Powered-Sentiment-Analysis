<div align="center">

# Unified Sentiment and Emotion Analysis System

Full-stack NLP project that combines binary sentiment classification with multi-class emotion detection.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[Quick Start](docs/QUICK_START.md) · [Project Summary](docs/PROJECT_SUMMARY.md) · [Methodology](docs/methodology.md)

</div>

## Overview

This repository merges two NLP workflows into one application: a binary sentiment model powered by Naive Bayes and a multi-class emotion model powered by logistic regression. The Python backend serves inference and stats, while the Next.js frontend visualizes the results.

## Table of Contents

- [Features](#features)
- [Stack](#stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [License](#license)

## Features

- Binary sentiment classification for positive and negative text.
- Emotion detection with multiple labeled emotion classes.
- Flask API for analysis and statistics.
- Next.js dashboard with animated charts and a modern UI.
- CLI tools for batch analysis and model inspection.
- Notebook-based analysis workflow and reusable Python modules.

## Stack

- Python 3.8+
- Flask and Flask-CORS
- Scikit-learn
- pandas, numpy, nltk, textblob, neattext
- Next.js 16
- Tailwind CSS
- Framer Motion
- Recharts

## Quick Start

```bash
git clone https://github.com/mangeshraut712/AI-Powered-Sentiment-Analysis.git
cd AI-Powered-Sentiment-Analysis
pip install -r requirements.txt
python src/app.py
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000` for the frontend and `http://localhost:5001` for the Flask API.

## Project Structure

```text
.
├── cli.py                # CLI entry point
├── data/                 # Dataset notes and generated assets
├── docs/                 # Summary, methodology, and quick start docs
├── examples/             # Usage examples
├── notebooks/            # Analysis notebooks
├── results/              # Trained model artifacts and reports
├── scripts/              # Analysis and verification helpers
├── src/
│   ├── app.py           # Flask API
│   ├── models/          # Binary and emotion models
│   ├── preprocessing.py # Text cleaning
│   ├── visualization.py # Plot helpers
│   └── utils/           # Shared helpers
└── web/                  # Next.js frontend
```

## Scripts

- `python cli.py classify --text "..."` - run binary sentiment classification.
- `python cli.py emotion --text "..."` - predict an emotion label.
- `python cli.py stats --top-n 10` - show classifier statistics.
- `python scripts/verify_setup.py` - check the project setup.
- `python scripts/run_analysis.py` - run the analysis pipeline.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

# Changelog

All notable changes to this project will be documented in this file.

---

## [1.0.0] - 2024-12-09

### Added - Project Reorganization
- Created comprehensive project structure with organized directories
- Added main `README.md` with project overview and documentation
- Created `src/` package with modular Python code:
  - `data_loader.py` - Data loading utilities
  - `preprocessing.py` - Text preprocessing functions
  - `feature_engineering.py` - Feature extraction tools
  - `models.py` - ML model implementations
  - `visualization.py` - Plotting and visualization utilities
- Added `requirements.txt` with all dependencies
- Created `.gitignore` for version control
- Added comprehensive documentation in `docs/`:
  - `methodology.md` - Detailed methodology documentation
- Added README files for each major directory:
  - `data/README.md`
  - `notebooks/README.md`
  - `archive/README.md`
- Added `LICENSE` file (MIT License)

### Organized
- Moved original submissions to `archive/` directory:
  - `archive/group_submission_2022/` - Group project (Summer 2022)
  - `archive/individual_submission_josh/` - Josh's individual submission
  - `archive/proposal_2021/` - Original proposals
- Organized notebooks in `notebooks/` directory:
  - `01_main_analysis_group_2022.ipynb`
  - `02_main_analysis_josh.ipynb`
  - `03_project_scoping_eda.ipynb`
- Centralized data in `data/` directory:
  - `data/raw/tweet_emotions.csv`
  - `data/processed/` (for generated files)
- Organized presentations:
  - `presentations/slides/` - PowerPoint presentations
  - `presentations/videos/` - Presentation recordings

### Improved
- Modularized code for better reusability
- Added comprehensive documentation
- Improved project structure for maintainability
- Added type hints and docstrings to all functions
- Created consistent naming conventions

---

## [0.2.0] - 2022-08-XX (Summer 2022)

### Added - Group Submission
- Main group project notebook (`DSCI_521_Group_Project_2022.ipynb`)
- Complete sentiment analysis pipeline
- Naive Bayes and Logistic Regression models
- Comprehensive visualizations
- Group presentation slides

**Contributors:** Mangesh Raut, Josh Clark, Will Wu, Mobin Rahimi

---

## [0.1.1] - 2022-08-XX (Summer 2022)

### Added - Individual Submission (Josh)
- Alternative analysis approach
- Focus on Logistic Regression
- Detailed preprocessing pipeline
- Individual presentation and video

**Contributor:** Josh Clark

---

## [0.1.0] - 2021-XX-XX (Summer 2021)

### Added - Initial Proposal
- Project scoping notebooks
- Initial exploratory data analysis
- Problem definition
- Phase 1 reports

**Contributors:** Various groups

---

## Future Enhancements

### Planned Features
- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time sentiment analysis API
- [ ] Web interface for predictions
- [ ] Multi-language support
- [ ] Emoji sentiment analysis
- [ ] Temporal trend analysis
- [ ] Model deployment scripts
- [ ] Automated testing suite
- [ ] CI/CD pipeline
- [ ] Docker containerization

### Potential Improvements
- [ ] Address sarcasm detection
- [ ] Improve handling of context
- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble methods
- [ ] Create interactive dashboards
- [ ] Add model explainability (LIME, SHAP)

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-12-09 | Complete reorganization and documentation |
| 0.2.0 | 2022-08 | Group submission (Summer 2022) |
| 0.1.1 | 2022-08 | Individual submission (Josh) |
| 0.1.0 | 2021 | Initial proposal and scoping |

---

**Versioning Scheme:** [Major].[Minor].[Patch]
- **Major:** Significant restructuring or breaking changes
- **Minor:** New features or substantial additions
- **Patch:** Bug fixes and minor improvements

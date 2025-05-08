# Classification ML Project

## Table of Contents
- [Project Overview](#project-overview)
- [Project Objectives](#project-objectives)
- [Project Structure](#project-structure)
- [How to run](#how-to-run)
- [Requirements](#requirements)
- [Conclusion](#conclusion)

## Project Overview
This project explores the use of various classification algorithms on classic datasets such as `Iris`, `Wine`, and `Breast Cancer`. The goal is to compare different models based on accuracy, precision, recall, and F1-score.

It serves as an exercise in supervised learning, model evaluation, and exploratory data analysis using Python and scikit-learn, providing me with hands-on experience with machine learning techniques in practice.

## Project Objectives
- Visualize and understand the structure of the datasets.
- Train and evaluate several classifiers, including Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, and ensemble methods.
- Apply feature scaling and dimensionality reduction techniques.
- Use cross-validation and hyperparameter tuning to optimize model performance.
- Gain hands-on experience in applying theoretical machine learning knowledge to simple, real-world datasets.

 ## Project Structure
- `notebooks/`: Jupyter notebooks for EDA and training
    - [EDA: Iris Dataset](notebooks/01_eda.ipynb)
    - [Model Training: Iris](notebooks/02_model_training_iris.ipynb)
    - [EDA: Wine Dataset](notebooks/03_eda_wine.ipynb)
    - [Model Training: Wine](notebooks/04_model_training_wine.ipynb)
    - [EDA: Breast Cancer Dataset](notebooks/05_eda_breast_cancer.ipynb)
    - [Model Training: Breast Cancer](notebooks/06_model_training_breast_cancer.ipynb)
- `scripts/`: Python scripts for reusable components
    - `load_data.py`: Dataset loading functions
    - `preprocessing.py`: Data cleaning, encoding, scaling
    - `train_models.py`: Training pipelines for classifiers
    - `evaluate.py`: Performance metrics and visualizations
- `results/`: Outputs including figures and saved models

## How to run?
Create environment
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

Install requirements
```bash
pip install -r requirements.txt
```
## Requirements
All dependencies are listed in requirements.txt. Key packages include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

## Conclusion
(Comming after project is finished)
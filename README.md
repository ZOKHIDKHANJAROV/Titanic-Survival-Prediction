# Titanic Survival Prediction

Production-ready Machine Learning project for predicting Titanic passenger survival using **Scikit-learn**, **Optuna**, and **MLflow 3**.

## Project Overview

This project demonstrates the complete Machine Learning lifecycle, from data preprocessing and feature engineering to hyperparameter optimization, experiment tracking, and model deployment preparation.

The goal is not only to build an accurate model but also to implement a clean and scalable ML pipeline following production best practices.

---

## Features

- Data preprocessing
- Feature Engineering
- Scikit-learn Pipeline
- Cross Validation
- Hyperparameter Optimization with Optuna
- MLflow 3 Experiment Tracking
- SQLite Tracking Backend
- Automatic Model Logging
- Evaluation Metrics
- Confusion Matrix
- ROC Curve
- Feature Importance
- Model Serialization
- Production Project Structure

---

## Project Structure

```
Titanic-Survival-Prediction/

├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── titanic_model.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── classification_report.txt
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── tuning.py
│   ├── evaluate.py
│   ├── visualization.py
│   ├── mlflow_utils.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── mlflow.db
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Pipeline

```
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Train / Validation Split
    │
    ▼
Scikit-learn Pipeline
    │
    ▼
Optuna Hyperparameter Search
    │
    ▼
Best Model
    │
    ▼
Model Evaluation
    │
    ▼
MLflow Tracking
    │
    ▼
Model Saving
```

---

## Feature Engineering

The following features are generated during preprocessing:

- FamilySize
- IsAlone
- FarePerPerson
- Title Extraction
- Deck Extraction
- Age Group
- Missing Value Imputation

---

## Model

Current algorithm:

- Random Forest Classifier

Hyperparameters are optimized using **Optuna** with Cross Validation.

---

## Model Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional reports:

- Confusion Matrix
- Classification Report
- ROC Curve
- Feature Importance

Example result:

| Metric | Score |
|---------|-------|
| Accuracy | 0.821 |
| Precision | 0.814 |
| Recall | 0.696 |
| F1-score | 0.750 |
| ROC-AUC | 0.840 |

---

## Hyperparameter Optimization

Hyperparameter tuning is performed using **Optuna**.

Optimized parameters include:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

Optimization objective:

- Cross Validation ROC-AUC

---

## MLflow

The project uses **MLflow 3** for experiment tracking.

Tracked information:

- Parameters
- Metrics
- Trained Model
- Artifacts
- Model Signature
- Input Example

Tracking backend:

```
SQLite
```

---

## Technologies

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Optuna
- MLflow 3
- Joblib
- Matplotlib
- Seaborn

---

## Installation

Clone repository

```bash
git clone https://github.com/your_username/Titanic-Survival-Prediction.git
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Run

```bash
python src/train.py
```

After training the project automatically:

- preprocesses data
- generates features
- performs hyperparameter optimization
- evaluates the model
- logs experiment to MLflow
- saves trained model
- saves evaluation reports

---

## MLflow UI

Start MLflow

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

---

## Prediction

```python
from predict import predict

prediction = predict(sample)
```

---

## Future Improvements

- XGBoost
- LightGBM
- CatBoost
- FastAPI
- Docker
- Docker Compose
- GitHub Actions
- Model Registry
- SHAP Explainability
- Monitoring
- CI/CD

---

## Skills Demonstrated

- Machine Learning
- Data Preprocessing
- Feature Engineering
- Model Selection
- Hyperparameter Optimization
- Cross Validation
- Experiment Tracking
- MLOps Fundamentals
- Production Project Structure
- Software Engineering Best Practices

---

## Author

**Zokhidkhan Jarov**

Backend Engineer | Machine Learning Engineer

GitHub:
https://github.com/ZOKHIDKHANJAROV

LinkedIn:
(Add your LinkedIn profile)

---

## License

This project is intended for educational and portfolio purposes.
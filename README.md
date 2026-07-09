# ML From Scratch

Machine learning algorithms implemented from scratch using Python and NumPy.

The goal of this repo is to understand the math and implementation details behind common ML models instead of only using high-level libraries. Experiments may use scikit-learn for datasets, preprocessing, train/test splits, and comparison models, but the core models and metrics are implemented from scratch.

## Implemented

### Models

#### Logistic Regression
- Sigmoid activation
- Binary cross-entropy loss
- Full-batch gradient descent
- Loss tracking
- Early stopping
- `fit`, `predict_proba`, and `predict`

#### Linear Regression
- Linear model: `y_pred = Xw + b`
- Mean squared error loss
- Full-batch gradient descent
- Loss tracking
- Early stopping
- `fit` and `predict`

### Metrics

#### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

#### Regression Metrics
- Mean squared error
- Root mean squared error
- Mean absolute error
- R² score

## Project Structure

```text
src/ml_from_scratch/
├── linear_models/
│   ├── logistic_regression.py
│   └── linear_regression.py
└── metrics/
    ├── classification.py
    └── regression.py

experiments/
├── logistic_regression/
│   └── breast_cancer_experiment.py
└── linear_regression/
    └── california_housing_experiment.py

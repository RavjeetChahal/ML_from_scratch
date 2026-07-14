# ML From Scratch

Machine learning algorithms implemented from scratch using Python and NumPy.

The goal of this repository is to understand the math, algorithms, and implementation details behind common machine learning models instead of only using high-level libraries.

Experiments may use scikit-learn for datasets, preprocessing, train/test splits, metrics validation, and comparison models, but the core models and metrics are implemented from scratch.

## Implemented

### Models

#### Logistic Regression

* Sigmoid activation
* Binary cross-entropy loss
* Full-batch gradient descent
* Loss tracking
* Early stopping
* `fit`, `predict_proba`, and `predict`

#### Linear Regression

* Linear model: `y_pred = Xw + b`
* Mean squared error loss
* Full-batch gradient descent
* Loss tracking
* Early stopping
* `fit` and `predict`

#### K-Nearest Neighbors Classifier

* Euclidean distance
* Nearest-neighbor search
* Majority-vote classification
* Multiclass classification support
* Configurable number of neighbors
* `fit` and `predict`

#### K-Nearest Neighbors Regressor

* Euclidean distance
* Nearest-neighbor search
* Mean-based regression prediction
* Configurable number of neighbors
* `fit` and `predict`

### Metrics

#### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 score
* Multiclass confusion matrix

#### Regression Metrics

* Mean squared error
* Root mean squared error
* Mean absolute error
* R² score

## Project Structure

```text
src/ml_from_scratch/
├── linear_models/
│   ├── __init__.py
│   ├── logistic_regression.py
│   └── linear_regression.py
├── neighbors/
│   ├── __init__.py
│   ├── knn_classifier.py
│   └── knn_regressor.py
├── metrics/
│   ├── __init__.py
│   ├── classification.py
│   └── regression.py
└── __init__.py

experiments/
├── logistic_regression/
│   └── breast_cancer_experiment.py
├── linear_regression/
│   └── california_housing_experiment.py
└── knn/
    ├── iris_experiment.py
    └── california_housing_experiment.py
```

## Setup

Create and activate a virtual environment, then install the package in editable mode:

```bash
pip install -e .
```

## Running Experiments

Run an experiment from the repository root:

```bash
python experiments/logistic_regression/breast_cancer_experiment.py
python experiments/linear_regression/california_housing_experiment.py
python experiments/knn/iris_experiment.py
python experiments/knn/california_housing_experiment.py
```

Each experiment compares the from-scratch implementation with the equivalent scikit-learn model.

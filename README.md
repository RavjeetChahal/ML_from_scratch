# ML From Scratch

Machine learning algorithms implemented from scratch using Python and NumPy.

The goal of this repository is to understand the math, algorithms, and implementation details behind common machine learning models instead of only using high-level libraries.

Scikit-learn is used for datasets, preprocessing, train/test splitting, metric validation, and comparison models. The core models and metrics are implemented from scratch.

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

#### K-Nearest Neighbors Classifier

- Euclidean distance
- Nearest-neighbor search
- Majority-vote classification
- Multiclass support
- Configurable number of neighbors
- `fit` and `predict`

#### K-Nearest Neighbors Regressor

- Euclidean distance
- Nearest-neighbor search
- Mean target prediction
- Configurable number of neighbors
- `fit` and `predict`

#### Decision Tree Classifier

- Recursive binary tree construction
- Gini impurity
- Weighted impurity
- Information gain
- Greedy best-split selection
- Majority-vote leaf predictions
- Multiclass support
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `fit` and `predict`

#### Decision Tree Regressor

- Recursive binary tree construction
- Mean squared error impurity
- Weighted MSE
- MSE reduction
- Greedy best-split selection
- Mean leaf predictions
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `fit` and `predict`

### Metrics

#### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 score
- Multiclass confusion matrix

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
├── neighbors/
│   ├── knn_classifier.py
│   └── knn_regressor.py
├── trees/
│   ├── decision_tree_classifier.py
│   └── decision_tree_regressor.py
├── metrics/
│   ├── classification.py
│   └── regression.py
└── utils/
    └── utils.py

experiments/
├── logistic_regression/
│   └── breast_cancer_experiment.py
├── linear_regression/
│   └── california_housing_experiment.py
├── knn/
│   ├── iris_experiment.py
│   └── california_housing_experiment.py
└── decision_trees/
    ├── iris_experiment.py
    └── diabetes_experiment.py
```

## Setup

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m pip install -e .
```

## Running Experiments

Run experiments from the repository root:

```bash
python experiments/logistic_regression/breast_cancer_experiment.py
python experiments/linear_regression/california_housing_experiment.py
python experiments/knn/iris_experiment.py
python experiments/knn/california_housing_experiment.py
python experiments/decision_trees/iris_experiment.py
python experiments/decision_trees/diabetes_experiment.py
```

Each experiment compares the from-scratch implementation with the equivalent scikit-learn model using both scratch and scikit-learn metric functions.

Regression experiments also compare prediction differences, while classification experiments report prediction agreement.
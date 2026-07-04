# ML From Scratch

Machine learning algorithms implemented from scratch using Python and NumPy.

The goal of this repo is to understand the math and implementation details behind common ML models instead of only using high-level libraries.

## Implemented

### Models
- Logistic Regression
  - Sigmoid activation
  - Binary cross-entropy loss
  - Gradient descent
  - Loss tracking
  - `fit`, `predict_proba`, and `predict`

### Metrics
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

## Project Structure

```text
src/ml_from_scratch/
├── linear_models/
│   └── logistic_regression.py
└── metrics/
    └── classification.py

experiments/
└── logistic_regression/
    └── breast_cancer_experiment.py
```
### Setup
pip install -e .

### Run Experiment
python experiments/logistic_regression/breast_cancer_experiment.py

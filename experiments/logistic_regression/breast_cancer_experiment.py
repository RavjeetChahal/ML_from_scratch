from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score as sklearn_f1_score,
    precision_score as sklearn_precision_score,
    recall_score as sklearn_recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_from_scratch.utils.utils import (
    print_classification_prediction_comparison,
    print_experiment_header,
    print_matrix_comparison,
    print_metric_row,
    print_model_headers,
    print_section_header,
)
from ml_from_scratch.linear_models.logistic_regression import (
    LogisticRegressionScratch,
)
from ml_from_scratch.metrics.classification import (
    accuracy_score as scratch_accuracy_score,
    confusion_matrix as scratch_confusion_matrix,
    f1_score as scratch_f1_score,
    precision_score as scratch_precision_score,
    recall_score as scratch_recall_score,
)


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scratch_model = LogisticRegressionScratch()
sklearn_model = LogisticRegression()

scratch_model.fit(X_train_scaled, y_train)
sklearn_model.fit(X_train_scaled, y_train)

y_pred_scratch = scratch_model.predict(X_test_scaled)
y_pred_sklearn = sklearn_model.predict(X_test_scaled)

scratch_model_name = "Scratch Logistic Regression"
sklearn_model_name = "Scikit-Learn Logistic Regression"

print_experiment_header(
    "Logistic Regression — Breast Cancer Dataset",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
)

print_section_header(
    "Classification Metrics Using Scratch Metric Functions"
)

print_model_headers(
    scratch_model_name,
    sklearn_model_name,
)

print_metric_row(
    "Accuracy",
    scratch_accuracy_score(y_test, y_pred_scratch),
    scratch_accuracy_score(y_test, y_pred_sklearn),
)

print_metric_row(
    "Precision",
    scratch_precision_score(y_test, y_pred_scratch),
    scratch_precision_score(y_test, y_pred_sklearn),
)

print_metric_row(
    "Recall",
    scratch_recall_score(y_test, y_pred_scratch),
    scratch_recall_score(y_test, y_pred_sklearn),
)

print_metric_row(
    "F1",
    scratch_f1_score(y_test, y_pred_scratch),
    scratch_f1_score(y_test, y_pred_sklearn),
)

print_matrix_comparison(
    scratch_confusion_matrix(y_test, y_pred_scratch),
    scratch_confusion_matrix(y_test, y_pred_sklearn),
)

print_section_header(
    "Classification Metrics Using Scikit-Learn Metric Functions"
)

print_model_headers(
    scratch_model_name,
    sklearn_model_name,
)

print_metric_row(
    "Accuracy",
    sklearn_accuracy_score(y_test, y_pred_scratch),
    sklearn_accuracy_score(y_test, y_pred_sklearn),
)

print_metric_row(
    "Precision",
    sklearn_precision_score(y_test, y_pred_scratch),
    sklearn_precision_score(y_test, y_pred_sklearn),
)

print_metric_row(
    "Recall",
    sklearn_recall_score(y_test, y_pred_scratch),
    sklearn_recall_score(y_test, y_pred_sklearn),
)

print_metric_row(
    "F1",
    sklearn_f1_score(y_test, y_pred_scratch),
    sklearn_f1_score(y_test, y_pred_sklearn),
)

print_matrix_comparison(
    sklearn_confusion_matrix(y_test, y_pred_scratch),
    sklearn_confusion_matrix(y_test, y_pred_sklearn),
)

print_classification_prediction_comparison(
    y_pred_scratch,
    y_pred_sklearn,
)
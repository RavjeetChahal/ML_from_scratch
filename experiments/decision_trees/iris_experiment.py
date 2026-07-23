from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml_from_scratch.utils.utils import (
    print_classification_prediction_comparison,
    print_experiment_header,
    print_matrix_comparison,
    print_metric_row,
    print_model_headers,
    print_section_header,
)
from ml_from_scratch.metrics.classification import (
    accuracy_score as scratch_accuracy_score,
    confusion_matrix as scratch_confusion_matrix,
)
from ml_from_scratch.trees.decision_tree_classifier import (
    DecisionTreeClassifierScratch,
)


data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y,
)

scratch_model = DecisionTreeClassifierScratch(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
)

sklearn_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="gini",
    random_state=7,
)

scratch_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

y_pred_scratch = scratch_model.predict(X_test)
y_pred_sklearn = sklearn_model.predict(X_test)

scratch_model_name = "Scratch Decision Tree"
sklearn_model_name = "Scikit-Learn Decision Tree"

print_experiment_header(
    "Decision Tree Classifier — Iris Dataset",
    X_train,
    X_test,
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

print_matrix_comparison(
    sklearn_confusion_matrix(y_test, y_pred_scratch),
    sklearn_confusion_matrix(y_test, y_pred_sklearn),
)

print_classification_prediction_comparison(
    y_pred_scratch,
    y_pred_sklearn,
)

print_section_header("Root Split Comparison")

print("Scratch root feature:", scratch_model.root.feature_index)
print("Scratch root threshold:", scratch_model.root.threshold)
print("Scikit-learn root feature:", sklearn_model.tree_.feature[0])
print("Scikit-learn root threshold:", sklearn_model.tree_.threshold[0])
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    precision_score as sklearn_precision_score,
    recall_score as sklearn_recall_score,
    f1_score as sklearn_f1_score,
    confusion_matrix as sklearn_confusion_matrix,
)

from ml_from_scratch.linear_models.logistic_regression import LogisticRegressionScratch
from ml_from_scratch.metrics.classification import (
    accuracy_score as scratch_accuracy_score,
    precision_score as scratch_precision_score,
    recall_score as scratch_recall_score,
    f1_score as scratch_f1_score,
    confusion_matrix as scratch_confusion_matrix,
)

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sk_model = LogisticRegression()
scratch_model = LogisticRegressionScratch()

sk_model.fit(X_train_scaled, y_train)
scratch_model.fit(X_train_scaled, y_train)

y_pred_sklearn = sk_model.predict(X_test_scaled)
y_pred_scratch = scratch_model.predict(X_test_scaled)

width = 38

print("\n" + "=" * 90)
print("Scratch Metrics".center(90))
print("=" * 90)

print(f"{'Scratch Logistic Regression':<{width}} | {'Sklearn Logistic Regression':<{width}}")
print("-" * 90)

print(f"{'Accuracy:':<12} {scratch_accuracy_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Accuracy:':<12} {scratch_accuracy_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Precision:':<12} {scratch_precision_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Precision:':<12} {scratch_precision_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Recall:':<12} {scratch_recall_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Recall:':<12} {scratch_recall_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'F1:':<12} {scratch_f1_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'F1:':<12} {scratch_f1_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Confusion:':<12} {str(scratch_confusion_matrix(y_test, y_pred_scratch)):<{width - 12}} | "
      f"{'Confusion:':<12} {str(scratch_confusion_matrix(y_test, y_pred_sklearn)):<{width - 12}}")


print("\n" + "=" * 90)
print("Scikit-Learn Metrics".center(90))
print("=" * 90)

print(f"{'Scratch Logistic Regression':<{width}} | {'Sklearn Logistic Regression':<{width}}")
print("-" * 90)

print(f"{'Accuracy:':<12} {sklearn_accuracy_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Accuracy:':<12} {sklearn_accuracy_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Precision:':<12} {sklearn_precision_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Precision:':<12} {sklearn_precision_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Recall:':<12} {sklearn_recall_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Recall:':<12} {sklearn_recall_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'F1:':<12} {sklearn_f1_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'F1:':<12} {sklearn_f1_score(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Confusion:':<12} {str(sklearn_confusion_matrix(y_test, y_pred_scratch)):<{width - 12}} | "
      f"{'Confusion:':<12} {str(sklearn_confusion_matrix(y_test, y_pred_sklearn)):<{width - 12}}")
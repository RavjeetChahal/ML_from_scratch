import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
)

from ml_from_scratch.neighbors.knn_classifier import KNeighborsClassifierScratch
from ml_from_scratch.metrics.classification import (
    accuracy_score as scratch_accuracy_score,
    confusion_matrix as scratch_confusion_matrix,
)

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sk_model = KNeighborsClassifier(n_neighbors=5)
scratch_model = KNeighborsClassifierScratch(n_neighbors=5)

sk_model.fit(X_train_scaled, y_train)
scratch_model.fit(X_train_scaled, y_train)

y_pred_sklearn = sk_model.predict(X_test_scaled)
y_pred_scratch = scratch_model.predict(X_test_scaled)

print("\n" + "=" * 70)
print("KNN Classifier: Scratch vs Scikit-Learn".center(70))
print("=" * 70)

print("\nScratch KNN")
print("-" * 70)
print(f"Accuracy: {scratch_accuracy_score(y_test, y_pred_scratch):.4f}")
print("Confusion Matrix:")

print(scratch_confusion_matrix(y_test, y_pred_scratch))

print("\nScikit-Learn KNN")
print("-" * 70)
print(f"Accuracy: {sklearn_accuracy_score(y_test, y_pred_sklearn):.4f}")
print("Confusion Matrix:")
print(sklearn_confusion_matrix(y_test, y_pred_sklearn))

print("\nPrediction Match Check")
print("-" * 70)
print(f"Predictions identical: {np.array_equal(y_pred_scratch, y_pred_sklearn)}")
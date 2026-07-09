from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error as sklearn_mean_squared_error,
    root_mean_squared_error as sklearn_root_mean_squared_error,
    mean_absolute_error as sklearn_mean_absolute_error,
    r2_score as sklearn_r2_score,
)

from ml_from_scratch.linear_models.linear_regression import LinearRegressionScratch
from ml_from_scratch.metrics.regression import (
    mean_squared_error as scratch_mean_squared_error,
    root_mean_squared_error as scratch_root_mean_squared_error,
    mean_absolute_error as scratch_mean_absolute_error,
    r2_score as scratch_r2_score,
)

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sk_model = LinearRegression()
scratch_model = LinearRegressionScratch()

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

print(f"{'Accuracy:':<12} {scratch_mean_squared_error(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Accuracy:':<12} {scratch_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Precision:':<12} {scratch_root_mean_squared_error(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Precision:':<12} {scratch_root_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Recall:':<12} {scratch_mean_absolute_error(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Recall:':<12} {scratch_mean_absolute_error(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'F1:':<12} {scratch_r2_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'F1:':<12} {scratch_r2_score(y_test, y_pred_sklearn):<{width - 12}}")

print("\n" + "=" * 90)
print("Scikit-Learn Metrics".center(90))
print("=" * 90)

print(f"{'Scratch Logistic Regression':<{width}} | {'Sklearn Logistic Regression':<{width}}")
print("-" * 90)

print(f"{'Accuracy:':<12} {sklearn_mean_squared_error(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Accuracy:':<12} {sklearn_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Precision:':<12} {sklearn_root_mean_squared_error(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Precision:':<12} {sklearn_root_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'Recall:':<12} {sklearn_mean_absolute_error(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'Recall:':<12} {sklearn_mean_absolute_error(y_test, y_pred_sklearn):<{width - 12}}")

print(f"{'F1:':<12} {sklearn_r2_score(y_test, y_pred_scratch):<{width - 12}} | "
      f"{'F1:':<12} {sklearn_r2_score(y_test, y_pred_sklearn):<{width - 12}}")
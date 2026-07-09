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
print("Regression Metrics Using Scratch Metric Functions".center(90))
print("=" * 90)

print(f"{'Scratch Linear Regression':<{width}} | {'Sklearn Linear Regression':<{width}}")
print("-" * 90)

print(f"{'MSE:':<12} {scratch_mean_squared_error(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'MSE:':<12} {scratch_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}.6f}")

print(f"{'RMSE:':<12} {scratch_root_mean_squared_error(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'RMSE:':<12} {scratch_root_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}.6f}")

print(f"{'MAE:':<12} {scratch_mean_absolute_error(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'MAE:':<12} {scratch_mean_absolute_error(y_test, y_pred_sklearn):<{width - 12}.6f}")

print(f"{'R²:':<12} {scratch_r2_score(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'R²:':<12} {scratch_r2_score(y_test, y_pred_sklearn):<{width - 12}.6f}")


print("\n" + "=" * 90)
print("Regression Metrics Using Scikit-Learn Metric Functions".center(90))
print("=" * 90)

print(f"{'Scratch Linear Regression':<{width}} | {'Sklearn Linear Regression':<{width}}")
print("-" * 90)

print(f"{'MSE:':<12} {sklearn_mean_squared_error(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'MSE:':<12} {sklearn_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}.6f}")

print(f"{'RMSE:':<12} {sklearn_root_mean_squared_error(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'RMSE:':<12} {sklearn_root_mean_squared_error(y_test, y_pred_sklearn):<{width - 12}.6f}")

print(f"{'MAE:':<12} {sklearn_mean_absolute_error(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'MAE:':<12} {sklearn_mean_absolute_error(y_test, y_pred_sklearn):<{width - 12}.6f}")

print(f"{'R²:':<12} {sklearn_r2_score(y_test, y_pred_scratch):<{width - 12}.6f} | "
      f"{'R²:':<12} {sklearn_r2_score(y_test, y_pred_sklearn):<{width - 12}.6f}")
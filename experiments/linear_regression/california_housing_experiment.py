from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error as sklearn_mean_absolute_error,
    mean_squared_error as sklearn_mean_squared_error,
    r2_score as sklearn_r2_score,
    root_mean_squared_error as sklearn_root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_from_scratch.utils.utils import (
    print_experiment_header,
    print_metric_row,
    print_model_headers,
    print_regression_prediction_comparison,
    print_section_header,
)
from ml_from_scratch.linear_models.linear_regression import (
    LinearRegressionScratch,
)
from ml_from_scratch.metrics.regression import (
    mean_absolute_error as scratch_mean_absolute_error,
    mean_squared_error as scratch_mean_squared_error,
    r2_score as scratch_r2_score,
    root_mean_squared_error as scratch_root_mean_squared_error,
)


data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scratch_model = LinearRegressionScratch()
sklearn_model = LinearRegression()

scratch_model.fit(X_train_scaled, y_train)
sklearn_model.fit(X_train_scaled, y_train)

y_pred_scratch = scratch_model.predict(X_test_scaled)
y_pred_sklearn = sklearn_model.predict(X_test_scaled)

scratch_model_name = "Scratch Linear Regression"
sklearn_model_name = "Scikit-Learn Linear Regression"

print_experiment_header(
    "Linear Regression — California Housing Dataset",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
)

print_section_header(
    "Regression Metrics Using Scratch Metric Functions"
)

print_model_headers(
    scratch_model_name,
    sklearn_model_name,
)

print_metric_row(
    "MSE",
    scratch_mean_squared_error(y_test, y_pred_scratch),
    scratch_mean_squared_error(y_test, y_pred_sklearn),
)

print_metric_row(
    "RMSE",
    scratch_root_mean_squared_error(
        y_test,
        y_pred_scratch,
    ),
    scratch_root_mean_squared_error(
        y_test,
        y_pred_sklearn,
    ),
)

print_metric_row(
    "MAE",
    scratch_mean_absolute_error(y_test, y_pred_scratch),
    scratch_mean_absolute_error(y_test, y_pred_sklearn),
)

print_metric_row(
    "R²",
    scratch_r2_score(y_test, y_pred_scratch),
    scratch_r2_score(y_test, y_pred_sklearn),
)

print_section_header(
    "Regression Metrics Using Scikit-Learn Metric Functions"
)

print_model_headers(
    scratch_model_name,
    sklearn_model_name,
)

print_metric_row(
    "MSE",
    sklearn_mean_squared_error(y_test, y_pred_scratch),
    sklearn_mean_squared_error(y_test, y_pred_sklearn),
)

print_metric_row(
    "RMSE",
    sklearn_root_mean_squared_error(
        y_test,
        y_pred_scratch,
    ),
    sklearn_root_mean_squared_error(
        y_test,
        y_pred_sklearn,
    ),
)

print_metric_row(
    "MAE",
    sklearn_mean_absolute_error(y_test, y_pred_scratch),
    sklearn_mean_absolute_error(y_test, y_pred_sklearn),
)

print_metric_row(
    "R²",
    sklearn_r2_score(y_test, y_pred_scratch),
    sklearn_r2_score(y_test, y_pred_sklearn),
)

print_regression_prediction_comparison(
    y_pred_scratch,
    y_pred_sklearn,
)
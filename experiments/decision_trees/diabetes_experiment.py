from sklearn.datasets import load_diabetes
from sklearn.metrics import (
    mean_absolute_error as sklearn_mean_absolute_error,
    mean_squared_error as sklearn_mean_squared_error,
    r2_score as sklearn_r2_score,
    root_mean_squared_error as sklearn_root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ml_from_scratch.utils.utils import (
    print_experiment_header,
    print_metric_row,
    print_model_headers,
    print_regression_prediction_comparison,
    print_section_header,
)
from ml_from_scratch.metrics.regression import (
    mean_absolute_error as scratch_mean_absolute_error,
    mean_squared_error as scratch_mean_squared_error,
    r2_score as scratch_r2_score,
    root_mean_squared_error as scratch_root_mean_squared_error,
)
from ml_from_scratch.trees.decision_tree_regressor import (
    DecisionTreeRegressorScratch,
)


data = load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
)

scratch_model = DecisionTreeRegressorScratch(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
)

sklearn_model = DecisionTreeRegressor(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="squared_error",
    random_state=7,
)

scratch_model.fit(X_train, y_train)
sklearn_model.fit(X_train, y_train)

y_pred_scratch = scratch_model.predict(X_test)
y_pred_sklearn = sklearn_model.predict(X_test)

scratch_model_name = "Scratch Decision Tree Regressor"
sklearn_model_name = "Scikit-Learn Decision Tree Regressor"


print_experiment_header(
    "Decision Tree Regressor — Diabetes Dataset",
    X_train,
    X_test,
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
    scratch_mean_squared_error(
        y_test,
        y_pred_scratch,
    ),
    scratch_mean_squared_error(
        y_test,
        y_pred_sklearn,
    ),
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
    scratch_mean_absolute_error(
        y_test,
        y_pred_scratch,
    ),
    scratch_mean_absolute_error(
        y_test,
        y_pred_sklearn,
    ),
)

print_metric_row(
    "R²",
    scratch_r2_score(
        y_test,
        y_pred_scratch,
    ),
    scratch_r2_score(
        y_test,
        y_pred_sklearn,
    ),
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
    sklearn_mean_squared_error(
        y_test,
        y_pred_scratch,
    ),
    sklearn_mean_squared_error(
        y_test,
        y_pred_sklearn,
    ),
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
    sklearn_mean_absolute_error(
        y_test,
        y_pred_scratch,
    ),
    sklearn_mean_absolute_error(
        y_test,
        y_pred_sklearn,
    ),
)

print_metric_row(
    "R²",
    sklearn_r2_score(
        y_test,
        y_pred_scratch,
    ),
    sklearn_r2_score(
        y_test,
        y_pred_sklearn,
    ),
)


print_regression_prediction_comparison(
    y_pred_scratch,
    y_pred_sklearn,
)


print_section_header("Root Split Comparison")

print(
    "Scratch root feature:",
    scratch_model.root.feature_index,
)
print(
    "Scratch root threshold:",
    scratch_model.root.threshold,
)

print(
    "Scikit-learn root feature:",
    sklearn_model.tree_.feature[0],
)
print(
    "Scikit-learn root threshold:",
    sklearn_model.tree_.threshold[0],
)
import numpy as np


OUTPUT_WIDTH = 90
MODEL_COLUMN_WIDTH = 38
METRIC_NAME_WIDTH = 12


def print_experiment_header(
    title,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print("\n" + "=" * OUTPUT_WIDTH)
    print(title.center(OUTPUT_WIDTH))
    print("=" * OUTPUT_WIDTH)

    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape:  {X_test.shape}")
    print(f"Training targets shape:  {y_train.shape}")
    print(f"Testing targets shape:   {y_test.shape}")


def print_section_header(title):
    print("\n" + "=" * OUTPUT_WIDTH)
    print(title.center(OUTPUT_WIDTH))
    print("=" * OUTPUT_WIDTH)


def print_model_headers(scratch_name, sklearn_name):
    print(
        f"{scratch_name:<{MODEL_COLUMN_WIDTH}} | "
        f"{sklearn_name:<{MODEL_COLUMN_WIDTH}}"
    )
    print("-" * OUTPUT_WIDTH)


def print_metric_row(
    metric_name,
    scratch_value,
    sklearn_value,
    decimals=6,
):
    scratch_formatted = f"{scratch_value:.{decimals}f}"
    sklearn_formatted = f"{sklearn_value:.{decimals}f}"

    print(
        f"{metric_name + ':':<{METRIC_NAME_WIDTH}} "
        f"{scratch_formatted:<{MODEL_COLUMN_WIDTH - METRIC_NAME_WIDTH}} | "
        f"{metric_name + ':':<{METRIC_NAME_WIDTH}} "
        f"{sklearn_formatted:<{MODEL_COLUMN_WIDTH - METRIC_NAME_WIDTH}}"
    )


def print_matrix_comparison(
    scratch_matrix,
    sklearn_matrix,
):
    print("\nScratch model confusion matrix:")
    print(scratch_matrix)

    print("\nScikit-learn model confusion matrix:")
    print(sklearn_matrix)


def print_classification_prediction_comparison(
    y_pred_scratch,
    y_pred_sklearn,
):
    print_section_header("Prediction Comparison")

    disagreements = y_pred_scratch != y_pred_sklearn

    print(
        "Predictions identical:",
        np.array_equal(y_pred_scratch, y_pred_sklearn),
    )
    print("Number of disagreements:", np.sum(disagreements))


def print_regression_prediction_comparison(
    y_pred_scratch,
    y_pred_sklearn,
):
    print_section_header("Prediction Comparison")

    absolute_differences = np.abs(
        y_pred_scratch - y_pred_sklearn
    )

    print(
        "Predictions numerically close:",
        np.allclose(y_pred_scratch, y_pred_sklearn),
    )
    print(
        "Mean absolute prediction difference:",
        np.mean(absolute_differences),
    )
    print(
        "Maximum absolute prediction difference:",
        np.max(absolute_differences),
    )
    print(
        "Number of differing predictions:",
        np.sum(
            ~np.isclose(
                y_pred_scratch,
                y_pred_sklearn,
            )
        ),
    )
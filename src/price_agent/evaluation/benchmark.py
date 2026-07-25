from __future__ import annotations

from .metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


def evaluate_forecasts(actual: list[float], predicted: list[float]) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "mape": mean_absolute_percentage_error(actual, predicted),
    }

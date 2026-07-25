from __future__ import annotations

from math import sqrt


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and have the same length")
    return sum(abs(a - b) for a, b in zip(actual, predicted)) / len(actual)


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and have the same length")
    return sqrt(sum((a - b) ** 2 for a, b in zip(actual, predicted)) / len(actual))


def mean_absolute_percentage_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and have the same length")
    errors = []
    for actual_value, predicted_value in zip(actual, predicted):
        if actual_value == 0:
            continue
        errors.append(abs((actual_value - predicted_value) / actual_value))
    if not errors:
        raise ValueError("at least one non-zero actual value is required")
    return sum(errors) / len(errors)

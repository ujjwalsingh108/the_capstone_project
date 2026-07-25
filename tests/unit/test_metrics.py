from price_agent.evaluation.metrics import mean_absolute_error, root_mean_squared_error


def test_mae_and_rmse() -> None:
    actual = [10.0, 12.0, 14.0]
    predicted = [11.0, 11.0, 15.0]

    assert mean_absolute_error(actual, predicted) == 1.0
    assert round(root_mean_squared_error(actual, predicted), 6) == round((3 / 3) ** 0.5, 6)

import numpy as np


#root mean squared error
def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Return root mean squared error for two equal-shaped arrays."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}"
        )

    return float(np.sqrt(np.mean((predicted - actual) ** 2)))



#correlation(predicted vs actual)
def pearson_correlation(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Return Pearson correlation coefficient between prediction and actual."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}"
        )

    if np.std(predicted) == 0 or np.std(actual) == 0:
        return float("nan")

    return float(np.corrcoef(predicted.ravel(), actual.ravel())[0, 1])



#pod
def probability_of_detection(
    predicted: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> float:
    """
    Return Probability of Detection (POD) for threshold-exceedance events.

    An event occurs when TEC is greater than or equal to `threshold`.
    POD = hits / (hits + misses)
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}"
        )

    predicted_event = predicted >= threshold
    actual_event = actual >= threshold

    hits = np.sum(predicted_event & actual_event)
    misses = np.sum(~predicted_event & actual_event)

    if hits + misses == 0:
        return float("nan")

    return float(hits / (hits + misses))


#CSI
def critical_success_index(
    predicted: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> float:
    """
    Return Critical Success Index (CSI) for threshold-exceedance events.

    CSI = hits / (hits + false_alarms + misses)
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}"
        )

    predicted_event = predicted >= threshold
    actual_event = actual >= threshold

    hits = np.sum(predicted_event & actual_event)
    false_alarms = np.sum(predicted_event & ~actual_event)
    misses = np.sum(~predicted_event & actual_event)

    denominator = hits + false_alarms + misses
    if denominator == 0:
        return float("nan")

    return float(hits / denominator)


#F1 score
def f1_score(
    predicted: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> float:
    """
    Return F1 score for threshold-exceedance events.

    F1 = 2 * hits / (2 * hits + false_alarms + misses)
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}"
        )

    predicted_event = predicted >= threshold
    actual_event = actual >= threshold

    hits = np.sum(predicted_event & actual_event)
    false_alarms = np.sum(predicted_event & ~actual_event)
    misses = np.sum(~predicted_event & actual_event)

    denominator = 2 * hits + false_alarms + misses
    if denominator == 0:
        return float("nan")

    return float(2 * hits / denominator)


#FAR
def false_alarm_ratio(
    predicted: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> float:
    """
    Return False Alarm Ratio (FAR) for threshold-exceedance events.

    FAR = false_alarms / (hits + false_alarms)
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, actual {actual.shape}"
        )

    predicted_event = predicted >= threshold
    actual_event = actual >= threshold

    hits = np.sum(predicted_event & actual_event)
    false_alarms = np.sum(predicted_event & ~actual_event)

    denominator = hits + false_alarms
    if denominator == 0:
        return float("nan")

    return float(false_alarms / denominator)
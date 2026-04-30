from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def class_distribution_from_labels(
    labels: Sequence[int], idx_to_class: Dict[int, str]
) -> List[Dict[str, object]]:
    counts = Counter(int(x) for x in labels)
    total = int(sum(counts.values()))
    rows: List[Dict[str, object]] = []
    for class_idx in sorted(counts):
        support = int(counts[class_idx])
        rows.append(
            {
                "class_idx": class_idx,
                "class_name": idx_to_class.get(class_idx, f"class_{class_idx}"),
                "support": support,
                "ratio": float(support / total) if total > 0 else 0.0,
            }
        )
    return rows


def bootstrap_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    if y_true_arr.size == 0:
        raise ValueError("Empty y_true is not supported.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    point_metrics = _compute_point_metrics(y_true_arr, y_pred_arr)
    rng = np.random.default_rng(seed)
    n = y_true_arr.shape[0]

    metric_samples = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        bs_true = y_true_arr[sample_idx]
        bs_pred = y_pred_arr[sample_idx]
        bs_metrics = _compute_point_metrics(bs_true, bs_pred)
        for metric_name, metric_value in bs_metrics.items():
            metric_samples[metric_name].append(metric_value)

    alpha = 1.0 - ci_level
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)

    ci = {}
    for metric_name, values in metric_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        ci[metric_name] = {
            "low": float(np.percentile(arr, low_q)),
            "high": float(np.percentile(arr, high_q)),
        }

    return {"point": point_metrics, "ci": ci}


def low_support_warnings(
    distribution_rows: Sequence[Dict[str, object]], min_support: int = 5
) -> List[str]:
    warnings = []
    for row in distribution_rows:
        support = int(row["support"])
        if support < min_support:
            warnings.append(
                f"[LOW_SUPPORT] {row['class_name']} (idx={row['class_idx']}): support={support} < {min_support}"
            )
    return warnings

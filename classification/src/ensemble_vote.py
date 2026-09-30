"""Reconstructed implementation of the documented v2 voting rule.

The original fitting/search script was not retained. The constants below come
from a preserved screenshot of the v2 configuration that achieved public F1
0.78260. They are not claimed as the exact configuration behind the separate
final leaderboard score of 0.78284.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class VoteRule:
    low: float
    high: float
    weight: float


V2_RULES = {
    "convnext": VoteRule(low=0.4891, high=0.6124, weight=0.2696),
    "swin": VoteRule(low=0.5862, high=0.6973, weight=0.0981),
    "dinov2": VoteRule(low=0.2460, high=0.8884, weight=0.2489),
    "efficientnet": VoteRule(low=0.0905, high=0.4400, weight=0.3834),
}
V2_VOTE_THRESHOLD = -0.1750


def trinarize(probabilities: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map class-1 probabilities to {-1, 0, +1} confidence votes."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if low > high:
        raise ValueError("low threshold must not exceed high threshold")
    return np.where(probabilities >= high, 1, np.where(probabilities <= low, -1, 0))


def ensemble_score(
    probabilities: Mapping[str, np.ndarray],
    rules: Mapping[str, VoteRule] = V2_RULES,
) -> np.ndarray:
    """Calculate one weighted vote score per sample."""
    missing = set(rules) - set(probabilities)
    if missing:
        raise KeyError(f"Missing probability arrays: {sorted(missing)}")

    score = None
    expected_shape = None
    for model_name, rule in rules.items():
        model_probabilities = np.asarray(probabilities[model_name], dtype=np.float64)
        if expected_shape is None:
            expected_shape = model_probabilities.shape
        elif model_probabilities.shape != expected_shape:
            raise ValueError("All model probability arrays must have the same shape")

        weighted_vote = rule.weight * trinarize(
            model_probabilities,
            low=rule.low,
            high=rule.high,
        )
        score = weighted_vote if score is None else score + weighted_vote

    return score


def ensemble_predict(
    probabilities: Mapping[str, np.ndarray],
    vote_threshold: float = V2_VOTE_THRESHOLD,
) -> np.ndarray:
    """Return binary predictions using the documented v2 decision threshold."""
    return (ensemble_score(probabilities) >= vote_threshold).astype(np.int64)

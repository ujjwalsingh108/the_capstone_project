from __future__ import annotations

from ..data.features import normalize_feature_map


def run_feature_engineering(features: dict[str, float]) -> dict[str, float]:
    return normalize_feature_map(features)

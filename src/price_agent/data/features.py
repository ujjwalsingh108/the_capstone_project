from __future__ import annotations


def normalize_feature_map(features: dict[str, float]) -> dict[str, float]:
    total = sum(abs(value) for value in features.values())
    if total == 0:
        return dict(features)
    return {name: value / total for name, value in features.items()}

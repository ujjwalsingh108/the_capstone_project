from __future__ import annotations

import json
import re
from typing import Any

from .items import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]


def simplify(text_list: Any) -> str:
    """Return simplified text with compact whitespace and bounded length."""
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title: str, description: Any, features: Any, details: dict[str, Any]) -> str:
    """Build a cleansed product text block for model training."""
    for remove in REMOVALS:
        details.pop(remove, None)

    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"

    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details: dict[str, Any]) -> float:
    weight_str = details.get("Item Weight")
    if not weight_str:
        return 0.0

    parts = str(weight_str).split(" ")
    if len(parts) < 2:
        return 0.0

    try:
        amount = float(parts[0])
    except (TypeError, ValueError):
        return 0.0

    unit = parts[1].lower()
    if unit == "pounds":
        return amount
    if unit == "ounces":
        return amount / 16
    if unit == "grams":
        return amount / 453.592
    if unit == "milligrams":
        return amount / 453592
    if unit == "kilograms":
        return amount / 0.453592
    if unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
        return amount / 100
    return 0.0


def parse(datapoint: dict[str, Any], category: str) -> Item | None:
    try:
        price = float(datapoint["price"])
    except (KeyError, TypeError, ValueError):
        return None

    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    try:
        title = datapoint["title"]
        description = datapoint.get("description")
        features = datapoint.get("features")
        details = json.loads(datapoint["details"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    weight = get_weight(details)
    full = scrub(title, description, features, details)
    if len(full) < MIN_CHARS:
        return None

    return Item(
        title=title,
        category=category,
        price=price,
        full=full,
        weight=weight,
    )
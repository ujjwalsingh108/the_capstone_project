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

NULL_LIKE = {"", "none", "null", "nan", "n/a"}


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


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_LIKE
    return False


def _parse_price(value: Any) -> float | None:
    if _is_null_like(value):
        return None

    text = str(value).strip().replace(",", "")
    text = re.sub(r"[^\d.\-]", "", text)
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _parse_details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if _is_null_like(value):
        return {}
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {}
    return {}


def _pick_category(datapoint: dict[str, Any], category: str | None) -> str:
    if category:
        return category

    main_category = datapoint.get("main_category")
    if isinstance(main_category, str) and main_category.strip():
        return main_category.strip()

    categories = datapoint.get("categories")
    if isinstance(categories, list):
        for value in categories:
            if isinstance(value, str) and value.strip():
                return value.strip()

    return "unknown"


def scrub(
    title: str,
    description: Any,
    features: Any,
    details: dict[str, Any],
    subtitle: Any = None,
    author: Any = None,
    store: Any = None,
) -> str:
    """Build a cleansed product text block for model training."""
    for remove in REMOVALS:
        details.pop(remove, None)

    result = title + "\n"
    if not _is_null_like(subtitle):
        result += simplify(subtitle) + "\n"
    if not _is_null_like(author):
        result += simplify(author) + "\n"
    if not _is_null_like(store):
        result += simplify(store) + "\n"
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


def parse(datapoint: dict[str, Any], category: str | None = None) -> Item | None:
    price = _parse_price(datapoint.get("price"))
    if price is None:
        return None

    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    try:
        title = str(datapoint["title"]).strip()
        description = datapoint.get("description")
        features = datapoint.get("features")
        subtitle = datapoint.get("subtitle")
        author = datapoint.get("author")
        store = datapoint.get("store")
        parent_asin = datapoint.get("parent_asin")
        details = _parse_details(datapoint.get("details"))
    except (KeyError, TypeError, ValueError):
        return None

    if not title:
        return None

    weight = get_weight(details)
    full = scrub(title, description, features, details, subtitle=subtitle, author=author, store=store)
    if len(full) < MIN_CHARS:
        return None

    resolved_category = _pick_category(datapoint, category)

    return Item(
        title=title,
        category=resolved_category,
        price=price,
        full=full,
        weight=weight,
        parent_asin=str(parent_asin) if parent_asin is not None else None,
    )

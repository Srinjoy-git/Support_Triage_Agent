"""Rule-based request and product-area classification."""

from __future__ import annotations

import re
from collections.abc import Iterable

from utils import load_rules_config, normalize_text


RULES = load_rules_config()
REQUEST_TYPE_RULES: dict[str, list[str]] = RULES.get("request_type_rules", {})
PRODUCT_AREA_RULES: dict[str, list[str]] = RULES.get("product_area_rules", {})
PRODUCT_AREA_PRIORITY: list[str] = RULES.get(
    "product_area_priority",
    [
        "security",
        "billing",
        "authentication",
        "integrations",
        "data",
        "ui",
        "notifications",
        "performance",
        "bugs",
        "feature_request",
        "account_management",
        "general",
    ],
)
DEFAULT_PRODUCT_AREA = "general"


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    """Return True if any keyword appears in normalized text."""
    return any(normalize_text(keyword).lower() in text for keyword in keywords)


def _matches_trigger(text: str, trigger: str) -> bool:
    """Match normalized trigger phrases while avoiding accidental partial words."""
    normalized_trigger = normalize_text(trigger).lower()
    if not normalized_trigger:
        return False

    # Multi-word triggers capture real support phrasing such as "access denied".
    if " " in normalized_trigger:
        return normalized_trigger in text

    return re.search(rf"\b{re.escape(normalized_trigger)}\b", text) is not None


def _matching_product_areas(text: str) -> set[str]:
    """Return every product area with at least one matching configured trigger."""
    return {
        product_area
        for product_area, triggers in PRODUCT_AREA_RULES.items()
        if any(_matches_trigger(text, trigger) for trigger in triggers)
    }


def classify_request(issue: str, company: str = "") -> str:
    """Classify a ticket as product_issue, feature_request, bug, or invalid."""
    text = normalize_text(issue).lower()
    normalized_company = normalize_text(company).lower()

    if not text:
        return "invalid"

    if _contains_any(text, REQUEST_TYPE_RULES.get("invalid", [])):
        return "invalid"

    for request_type in ("bug", "feature_request", "product_issue"):
        if _contains_any(text, REQUEST_TYPE_RULES.get(request_type, [])):
            return request_type

    if normalized_company in {"hackerrank", "claude", "visa"}:
        return "product_issue"

    return "invalid"


def detect_product_area(issue: str, company: str = "") -> str:
    """Detect the best product area using configurable, priority-aware rules."""
    text = normalize_text(issue).lower()

    if not text:
        return DEFAULT_PRODUCT_AREA

    matched_areas = _matching_product_areas(text)
    for product_area in PRODUCT_AREA_PRIORITY:
        if product_area in matched_areas:
            return product_area

    return DEFAULT_PRODUCT_AREA

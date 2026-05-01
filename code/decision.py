"""Decision rules for reply vs escalation."""

from __future__ import annotations

from utils import load_rules_config, normalize_text


RULES = load_rules_config()
ESCALATION_RULES: dict[str, list[str]] = RULES.get("escalation_rules", {})
DEFAULT_CONFIDENCE_THRESHOLD = float(RULES.get("confidence_threshold", 0.08))


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return True when any configured escalation keyword is present."""
    return any(normalize_text(keyword).lower() in text for keyword in keywords)


def get_confidence_threshold() -> float:
    """Return the configured retrieval confidence threshold."""
    try:
        return float(RULES.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE_THRESHOLD


def decide_status(
    issue: str,
    product_area: str,
    request_type: str,
    confidence_score: float,
    confidence_threshold: float,
    company: str = "",
) -> str:
    """Decide whether to reply directly or escalate to a human."""
    text = normalize_text(issue).lower()
    normalized_company = normalize_text(company).lower()

    if request_type == "invalid":
        return "replied"

    if _contains_any(text, ESCALATION_RULES.get("security", [])):
        return "escalated"

    if _contains_any(text, ESCALATION_RULES.get("account_locked", [])):
        return "escalated"

    if normalized_company != "visa" and _contains_any(
        text,
        ESCALATION_RULES.get("payment", []),
    ):
        return "escalated"

    if _contains_any(text, ESCALATION_RULES.get("platform_outage", [])):
        return "escalated"

    if confidence_score < confidence_threshold:
        return "escalated"

    return "replied"


def explain_decision(
    issue: str,
    product_area: str,
    request_type: str,
    confidence_score: float,
    confidence_threshold: float,
) -> str:
    """Return a concise reason for routing."""
    text = normalize_text(issue).lower()

    if request_type == "invalid":
        return "Ticket is outside the supported corpus scope."

    if _contains_any(text, ESCALATION_RULES.get("security", [])):
        return "Security-sensitive issue detected."

    if _contains_any(text, ESCALATION_RULES.get("account_locked", [])):
        return "Account access issue requires human review."

    if _contains_any(text, ESCALATION_RULES.get("payment", [])):
        return "Payment or money-related issue detected."

    if _contains_any(text, ESCALATION_RULES.get("platform_outage", [])):
        return "Potential service outage requires escalation."

    if confidence_score < confidence_threshold:
        return (
            f"Retrieval confidence {confidence_score:.2f} is below "
            f"threshold {confidence_threshold:.2f}."
        )

    return f"Matched {product_area or 'general'} support documentation."

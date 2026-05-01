"""Grounded response generation."""

from __future__ import annotations

import re

from retrieval import SupportDocument
from utils import normalize_text


ESCALATION_RESPONSE = "This issue requires further review. Please contact official support."
OUT_OF_SCOPE_RESPONSE = "I am sorry, this is out of scope for this support agent."


def _sentence_chunks(text: str) -> list[str]:
    """Split long document text into readable sentence-like chunks."""
    cleaned = normalize_text(text)
    return [
        normalize_text(chunk)
        for chunk in re.split(r"(?<=[.!?])\s+", cleaned)
        if normalize_text(chunk)
    ]


def _best_excerpt(document: SupportDocument, max_chars: int = 900) -> str:
    """Return a concise excerpt from the retrieved support document."""
    chunks = _sentence_chunks(document.text)
    excerpt = " ".join(chunks[:6]) if chunks else document.text

    if len(excerpt) <= max_chars:
        return excerpt

    return excerpt[: max_chars - 3].rstrip() + "..."


def generate_response(
    status: str,
    retrieved_doc: SupportDocument | str | None,
    product_area: str,
    decision_reason: str,
) -> tuple[str, str]:
    """Generate a professional response and concise justification."""
    normalized_status = normalize_text(status).lower()

    if normalized_status == "escalated":
        return ESCALATION_RESPONSE, decision_reason

    if retrieved_doc is None:
        return OUT_OF_SCOPE_RESPONSE, decision_reason

    if isinstance(retrieved_doc, str):
        return normalize_text(retrieved_doc), decision_reason

    response = _best_excerpt(retrieved_doc)
    justification = (
        f"{decision_reason} Source: {retrieved_doc.path}."
        if decision_reason
        else f"Source: {retrieved_doc.path}."
    )

    return response, justification

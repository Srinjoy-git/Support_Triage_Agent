"""Terminal entry point for the support triage agent.

Run from the repository root with:

    python code/main.py

The agent reads support_tickets/support_tickets.csv, uses only the local
support corpus under data/, and writes support_tickets/output.csv.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path

from classifier import classify_request, detect_product_area
from decision import decide_status, explain_decision, get_confidence_threshold
from response import generate_response
from retrieval import build_index, load_corpus, retrieve_with_confidence
from utils import (
    OUTPUT_PATH,
    SUPPORT_TICKETS_PATH,
    normalize_text,
    setup_logging,
    split_ticket_issues,
)


OUTPUT_FIELDS = ("status", "product_area", "response", "justification", "request_type")


def _progress(rows: list[dict[str, str]]) -> Iterable[dict[str, str]]:
    """Use tqdm when available, but keep the CLI dependency optional."""
    try:
        from tqdm import tqdm
    except ImportError:
        return rows

    return tqdm(rows, desc="Tickets", unit="ticket")


def _unique_join(values: Iterable[str], separator: str = "; ") -> str:
    """Join non-empty values while preserving order and removing duplicates."""
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        cleaned = normalize_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_values.append(cleaned)

    return separator.join(unique_values)


def _choose_single_value(
    fragments: list[dict[str, str | float]],
    field: str,
    fallback: str,
) -> str:
    """Choose one evaluator-compatible value from processed fragments."""
    escalated_values = [
        normalize_text(fragment.get(field))
        for fragment in fragments
        if fragment.get("status") == "escalated" and normalize_text(fragment.get(field))
    ]
    if escalated_values:
        return escalated_values[0]

    for fragment in fragments:
        value = normalize_text(fragment.get(field))
        if value:
            return value

    return fallback


def load_tickets(path: Path = SUPPORT_TICKETS_PATH) -> list[dict[str, str]]:
    """Load input tickets and normalize header names."""
    with path.open(mode="r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        tickets: list[dict[str, str]] = []

        for row in reader:
            tickets.append(
                {
                    "issue": normalize_text(row.get("Issue") or row.get("issue")),
                    "subject": normalize_text(row.get("Subject") or row.get("subject")),
                    "company": normalize_text(row.get("Company") or row.get("company")),
                    "raw_issue": "" if row.get("Issue") is None else str(row.get("Issue")),
                }
            )

        return tickets


def process_issue_fragment(
    issue_text: str,
    company: str,
    confidence_threshold: float,
) -> dict[str, str | float]:
    """Process one issue fragment through retrieval, routing, and response."""
    request_type = classify_request(issue_text, company)
    product_area = detect_product_area(issue_text, company)
    retrieved_doc, confidence_score = retrieve_with_confidence(issue_text, company)
    status = decide_status(
        issue=issue_text,
        product_area=product_area,
        request_type=request_type,
        confidence_score=confidence_score,
        confidence_threshold=confidence_threshold,
        company=company,
    )
    decision_reason = explain_decision(
        issue=issue_text,
        product_area=product_area,
        request_type=request_type,
        confidence_score=confidence_score,
        confidence_threshold=confidence_threshold,
    )
    response, justification = generate_response(
        status=status,
        retrieved_doc=retrieved_doc,
        product_area=product_area,
        decision_reason=decision_reason,
    )

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
        "confidence_score": confidence_score,
    }


def process_ticket(ticket: dict[str, str]) -> dict[str, str]:
    """Process one CSV row and roll up multiple issue fragments if present."""
    subject = normalize_text(ticket.get("subject"))
    company = normalize_text(ticket.get("company"))
    raw_issue = ticket.get("raw_issue") or ticket.get("issue") or ""
    fragments = split_ticket_issues(raw_issue) or [ticket.get("issue", "")]
    threshold = get_confidence_threshold()

    contextual_fragments = [
        normalize_text(f"{subject} {fragment}") if subject else normalize_text(fragment)
        for fragment in fragments
    ]
    processed = [
        process_issue_fragment(fragment, company, threshold)
        for fragment in contextual_fragments
    ]

    status = (
        "escalated"
        if any(fragment["status"] == "escalated" for fragment in processed)
        else "replied"
    )
    product_area = _choose_single_value(processed, "product_area", "")
    request_type = _choose_single_value(processed, "request_type", "invalid")

    if status == "escalated":
        response, justification = generate_response(
            status=status,
            retrieved_doc=None,
            product_area=product_area,
            decision_reason=_unique_join(
                str(fragment["justification"])
                for fragment in processed
                if fragment["status"] == "escalated"
            ),
        )
    else:
        response, justification = generate_response(
            status=status,
            retrieved_doc=_unique_join(str(fragment["response"]) for fragment in processed),
            product_area=product_area,
            decision_reason="Response is grounded in the retrieved support corpus.",
        )

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }


def write_output(rows: list[dict[str, str]], path: Path = OUTPUT_PATH) -> None:
    """Write evaluator-compatible predictions to output.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode="w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline() -> list[dict[str, str]]:
    """Run the complete terminal pipeline."""
    logger = setup_logging()

    logger.info("Loading data...")
    tickets = load_tickets()
    documents = load_corpus()
    logger.info("Loaded %s tickets and %s support documents.", len(tickets), len(documents))

    build_index(documents)

    logger.info("Processing tickets...")
    results = [process_ticket(ticket) for ticket in _progress(tickets)]

    logger.info("Saving output...")
    write_output(results)
    logger.info("Saved output to %s.", OUTPUT_PATH)

    return results


def main() -> None:
    """Run the support triage CLI."""
    try:
        results = run_pipeline()
    except FileNotFoundError as exc:
        logging.getLogger("support_triage").error("Missing required file: %s", exc)
        raise SystemExit(1) from exc

    print(f"Done. Processed {len(results)} ticket(s).")
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

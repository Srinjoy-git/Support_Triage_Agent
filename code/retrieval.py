"""TF-IDF retrieval over the provided local support corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import DATA_DIR, normalize_text


@dataclass(frozen=True)
class SupportDocument:
    """A searchable support-corpus document."""

    text: str
    company: str
    product_area: str
    path: str


_documents: list[SupportDocument] = []
_vectorizer: TfidfVectorizer | None = None
_document_vectors: Any = None


def _strip_markdown_noise(text: str) -> str:
    """Remove common markdown/frontmatter clutter before indexing."""
    text = re.sub(r"^---.*?---", " ", text, flags=re.DOTALL)
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[_#>|*-]+", " ", text)
    return normalize_text(text)


def _infer_company(path: Path) -> str:
    """Infer company/domain from data/<company>/... path."""
    try:
        return path.relative_to(DATA_DIR).parts[0].lower()
    except (IndexError, ValueError):
        return "none"


def _infer_product_area(path: Path, text: str) -> str:
    """Infer a stable product area from corpus path and content keywords."""
    path_text = str(path).lower()
    text_lower = text.lower()

    if "visa" in path_text:
        if "traveller" in text_lower or "travel" in text_lower:
            return "travel_support"
        if "merchant" in path_text:
            return "merchant_support"
        return "general_support"

    if "hackerrank" in path_text:
        if "community" in path_text or "delete my account" in text_lower:
            return "community"
        if "engage" in path_text:
            return "engage"
        return "screen"

    if "claude" in path_text:
        if "privacy" in path_text or "delete" in text_lower:
            return "privacy"
        if "safeguards" in path_text:
            return "safety"
        if "identity" in path_text or "sso" in text_lower:
            return "authentication"
        if "code" in path_text or "connector" in path_text:
            return "coding"
        return "conversation_management"

    return "general"


def load_corpus(data_dir: Path = DATA_DIR) -> list[SupportDocument]:
    """Read all markdown files from the provided support corpus."""
    documents: list[SupportDocument] = []

    for path in sorted(data_dir.rglob("*.md")):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = _strip_markdown_noise(raw_text)

        if not cleaned_text:
            continue

        documents.append(
            SupportDocument(
                text=cleaned_text,
                company=_infer_company(path),
                product_area=_infer_product_area(path, cleaned_text),
                path=str(path.relative_to(data_dir)),
            )
        )

    return documents


def build_index(docs: list[SupportDocument]) -> None:
    """Build an in-memory TF-IDF index from support documents."""
    global _documents, _vectorizer, _document_vectors

    _documents = [doc for doc in docs if normalize_text(doc.text)]

    if not _documents:
        _vectorizer = None
        _document_vectors = None
        return

    _vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    _document_vectors = _vectorizer.fit_transform(doc.text for doc in _documents)


def _candidate_indices(company: str) -> list[int]:
    """Prefer same-company documents when the ticket names a known company."""
    normalized_company = normalize_text(company).lower()

    if normalized_company in {"hackerrank", "claude", "visa"}:
        indices = [
            index
            for index, document in enumerate(_documents)
            if document.company == normalized_company
        ]
        if indices:
            return indices

    return list(range(len(_documents)))


def retrieve_with_confidence(
    query: str,
    company: str = "",
) -> tuple[SupportDocument | None, float]:
    """Return the top matching support document and cosine score."""
    normalized_query = normalize_text(query)

    if not normalized_query or _vectorizer is None or _document_vectors is None:
        return None, 0.0

    candidate_indices = _candidate_indices(company)
    if not candidate_indices:
        return None, 0.0

    query_vector = _vectorizer.transform([normalized_query])
    candidate_vectors = _document_vectors[candidate_indices]
    scores = cosine_similarity(query_vector, candidate_vectors).flatten()

    if scores.size == 0:
        return None, 0.0

    local_best = int(scores.argmax())
    confidence = float(scores[local_best])

    if confidence <= 0:
        return None, 0.0

    return _documents[candidate_indices[local_best]], confidence

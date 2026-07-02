from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

from .models import EvidenceItem, EvidenceQuery, RetrievalResult

try:  # scikit-learn gives the nicest classroom explanation, but keep a no-install fallback.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - exercised only when sklearn is not installed
    TfidfVectorizer = None
    cosine_similarity = None


EXPECTED_COLUMNS = [
    "case_id",
    "case_text",
    "case_type",
    "priority",
    "assigned_group",
    "status",
    "resolution_notes",
    "created_at",
    "resolved_at",
    "processing_steps",
    "workflow_total_time",
]

PRIORITY_MAP = {
    "lowest": "low",
    "low": "low",
    "minor": "low",
    "medium": "medium",
    "normal": "medium",
    "major": "high",
    "high": "high",
    "highest": "blocker",
    "blocker": "blocker",
    "critical": "blocker",
    "unknown": "unknown",
    "": "unknown",
}


@dataclass
class CaseStore:
    path: Path
    min_similarity: float = 0.35

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Prepared case file not found: {self.path}")
        self.df = pd.read_csv(self.path).fillna("")
        missing = [col for col in EXPECTED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Prepared case file is missing columns: {missing}")
        self.df["priority"] = self.df["priority"].map(normalize_priority)
        self.df["search_text"] = self.df.apply(build_search_text, axis=1)
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(self.df["search_text"].tolist())
        else:
            self.vectorizer = None
            self.matrix = None

    def search(self, query: EvidenceQuery) -> RetrievalResult:
        query_text = " ".join(
            part
            for part in [query.issue_summary, query.suspected_case_type or "", query.affected_system or ""]
            if part
        ).strip()
        if not query_text:
            return RetrievalResult(query=query, evidence=[], no_close_match_found=True)

        if self.vectorizer is not None and self.matrix is not None and cosine_similarity is not None:
            query_vector = self.vectorizer.transform([query_text])
            scores = cosine_similarity(query_vector, self.matrix).flatten()
        else:
            scores = self._fallback_scores(query_text)

        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        evidence: list[EvidenceItem] = []
        for idx, score in ranked[: query.top_k]:
            score = float(score)
            if score < self.min_similarity:
                continue
            row = self.df.iloc[idx]
            evidence.append(
                EvidenceItem(
                    case_id=str(row["case_id"]),
                    similarity_score=round(score, 3),
                    snippet=make_snippet(str(row["case_text"])),
                    historical_case_type=str(row["case_type"] or "unknown"),
                    historical_priority=normalize_priority(str(row["priority"])),
                    historical_routing_group=str(row["assigned_group"] or "unknown"),
                    historical_status_or_resolution=str(
                        row["resolution_notes"] or row["status"] or "unknown"
                    ),
                )
            )
        return RetrievalResult(
            query=query,
            evidence=evidence,
            no_close_match_found=len(evidence) == 0,
        )

    def _fallback_scores(self, query_text: str) -> list[float]:
        query_text = query_text.lower()
        return [SequenceMatcher(None, query_text, str(text).lower()).ratio() for text in self.df["search_text"]]


def normalize_priority(value: str) -> str:
    return PRIORITY_MAP.get(str(value).strip().lower(), "unknown")


def build_search_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(col, ""))
        for col in ["case_text", "case_type", "priority", "assigned_group", "status", "resolution_notes"]
        if str(row.get(col, "")).strip()
    )


def make_snippet(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"

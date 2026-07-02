from __future__ import annotations

from pathlib import Path

from agents import function_tool

from .data_store import CaseStore
from .guardrails import retrieval_tool_input_guardrail, retrieval_tool_output_guardrail
from .models import EvidenceQuery, RetrievalResult


def build_retrieval_tool(data_path: Path):
    """Create a retrieval tool bound to one prepared CSV file."""
    store = CaseStore(data_path)

    @function_tool(
        name_override="find_similar_cases",
        description_override=(
            "Search the local prepared_cases.csv file for similar historical operational cases. "
            "Return only the top evidence records, not the full dataset."
        ),
        tool_input_guardrails=[retrieval_tool_input_guardrail],
        tool_output_guardrails=[retrieval_tool_output_guardrail],
    )
    def find_similar_cases(
        issue_summary: str,
        suspected_case_type: str | None = None,
        affected_system: str | None = None,
        top_k: int = 3,
    ) -> RetrievalResult:
        query = EvidenceQuery(
            issue_summary=issue_summary,
            suspected_case_type=suspected_case_type,
            affected_system=affected_system,
            top_k=top_k,
        )
        return store.search(query)

    return find_similar_cases

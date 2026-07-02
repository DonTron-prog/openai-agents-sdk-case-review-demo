from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Priority = Literal["low", "medium", "high", "blocker", "unknown"]
Confidence = Literal["low", "medium", "high"]


class WorkflowState(str, Enum):
    START = "Start"
    INPUT_CHECKED = "InputChecked"
    EXTRACTED = "Extracted"
    NEEDS_CLARIFICATION = "NeedsClarification"
    EVIDENCE_RETRIEVED = "EvidenceRetrieved"
    REVIEWED = "Reviewed"
    DRAFTED = "Drafted"
    GUARDRAIL_CHECKED = "GuardrailChecked"
    HUMAN_REVIEW_READY = "HumanReviewReady"
    REJECTED_INPUT = "RejectedInput"
    UNSAFE_OUTPUT = "UnsafeOutput"
    ERROR = "Error"


STEP_LABELS: dict[WorkflowState, str] = {
    WorkflowState.START: "Start run",
    WorkflowState.INPUT_CHECKED: "Check input",
    WorkflowState.EXTRACTED: "Extract case details",
    WorkflowState.NEEDS_CLARIFICATION: "Ask for missing information",
    WorkflowState.EVIDENCE_RETRIEVED: "Find similar cases",
    WorkflowState.REVIEWED: "Review case",
    WorkflowState.DRAFTED: "Draft response",
    WorkflowState.GUARDRAIL_CHECKED: "Validate output",
    WorkflowState.HUMAN_REVIEW_READY: "Ready for human review",
    WorkflowState.REJECTED_INPUT: "Rejected input",
    WorkflowState.UNSAFE_OUTPUT: "Blocked unsafe output",
    WorkflowState.ERROR: "Run failed",
}

REQUIRED_EXTRACTION_FIELDS = ("issue_summary", "affected_system", "user_impact")


class EvidenceItem(BaseModel):
    case_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    snippet: str
    historical_case_type: str = "unknown"
    historical_priority: Priority = "unknown"
    historical_routing_group: str | None = None
    historical_status_or_resolution: str | None = None


class ExtractedCase(BaseModel):
    issue_summary: str = Field(description="One sentence summary of the operational case.")
    affected_system: str | None = Field(default=None, description="System, service, process, or workflow affected.")
    user_impact: str | None = Field(default=None, description="Who or what is affected and how.")
    urgency_signals: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    suspected_case_type: str = "unknown"

    @field_validator("issue_summary")
    @classmethod
    def summary_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("issue_summary must be a useful short summary")
        return value

    @model_validator(mode="after")
    def normalize_missing_information(self) -> "ExtractedCase":
        missing = set(self.missing_information)
        if not self.affected_system:
            missing.add("affected_system")
        if not self.user_impact:
            missing.add("user_impact")
        self.missing_information = sorted(missing)
        return self

    @property
    def has_required_fields(self) -> bool:
        return bool(self.issue_summary and self.affected_system and self.user_impact)


class EvidenceQuery(BaseModel):
    issue_summary: str
    suspected_case_type: str | None = None
    affected_system: str | None = None
    top_k: int = Field(default=3, ge=1, le=5)


class RetrievalResult(BaseModel):
    query: EvidenceQuery
    evidence: list[EvidenceItem] = Field(default_factory=list)
    no_close_match_found: bool = False

    @model_validator(mode="after")
    def no_match_matches_evidence(self) -> "RetrievalResult":
        self.no_close_match_found = len(self.evidence) == 0 or self.no_close_match_found
        return self


class ReviewDecision(BaseModel):
    case_type: str
    priority: Priority
    routing_group: str
    confidence: Confidence
    evidence_case_ids: list[str] = Field(default_factory=list)
    no_close_match_found: bool = False
    uncertainty_reason: str | None = None

    @model_validator(mode="after")
    def high_confidence_requires_evidence(self) -> "ReviewDecision":
        if self.confidence == "high" and not self.evidence_case_ids:
            raise ValueError("high confidence review decisions require evidence_case_ids")
        if self.no_close_match_found and self.confidence == "high":
            raise ValueError("no-close-match decisions must not be high confidence")
        if self.confidence == "low" and not self.uncertainty_reason:
            self.uncertainty_reason = "Low confidence review requires human confirmation."
        return self


class DraftOnlyResponse(BaseModel):
    user_acknowledgement: str
    internal_review_note: str
    missing_information_request: str | None = None
    recommended_next_action: str


class CaseReviewResult(BaseModel):
    case_type: str
    priority: Priority
    routing_group: str
    confidence: Confidence
    evidence: list[EvidenceItem]
    no_close_match_found: bool = False
    uncertainty_reason: str | None = None
    user_acknowledgement: str
    internal_review_note: str
    missing_information_request: str | None = None
    recommended_next_action: str
    human_review_required: bool = True

    @model_validator(mode="after")
    def final_answer_is_review_ready(self) -> "CaseReviewResult":
        forbidden = ["assigned", "escalated", "closed", "resolved", "approved", "created"]
        text = " ".join(
            [
                self.user_acknowledgement,
                self.internal_review_note,
                self.recommended_next_action,
                self.missing_information_request or "",
            ]
        ).lower()
        for word in forbidden:
            if f"case was {word}" in text or f"has been {word}" in text:
                raise ValueError(
                    "final result must be review-ready and must not claim production action"
                )
        if self.confidence == "high" and not self.evidence:
            raise ValueError("high confidence final result requires evidence")
        if self.no_close_match_found and self.confidence == "high":
            raise ValueError("no-close-match final result must not be high confidence")
        if not self.human_review_required:
            raise ValueError("human_review_required must remain true for the classroom demo")
        return self


class FailureResult(BaseModel):
    state: WorkflowState
    step_label: str
    message: str
    retry_later: bool = False
    safe_summary: str | None = None


class TelemetryEvent(BaseModel):
    run_id: str
    step_label: str
    internal_state: str
    component: str
    attempt: int
    event: str
    retry_reason: str | None = None
    repair_prompt_used: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    validation_status: str
    next_step_label: str | None = None
    next_internal_state: str | None = None

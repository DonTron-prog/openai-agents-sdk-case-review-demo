from __future__ import annotations

import json
import re
from typing import Any

from agents import (
    GuardrailFunctionOutput,
    ToolGuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from pydantic import ValidationError

from .models import CaseReviewResult, RetrievalResult

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"api[_ -]?key", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]", re.IGNORECASE),
]
INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|your) instructions", re.IGNORECASE),
    re.compile(r"reveal .*system prompt", re.IGNORECASE),
    re.compile(r"hidden (system|developer) prompt", re.IGNORECASE),
]
OPERATIONAL_HINTS = [
    "case",
    "ticket",
    "request",
    "issue",
    "problem",
    "broken",
    "blocked",
    "outage",
    "transaction",
    "deploy",
    "access",
    "support",
    "review",
    "route",
    "priority",
    "rubric",
    "score",
    "policy",
    "incident",
    "application",
]


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


@input_guardrail(name="operational_case_input_guardrail", run_in_parallel=False)
def operational_case_input_guardrail(context: Any, agent: Any, input_text: str | list[Any]) -> GuardrailFunctionOutput:
    text = input_text if isinstance(input_text, str) else json.dumps(input_text)
    lowered = text.lower()
    reasons: list[str] = []
    if contains_secret(text):
        reasons.append("input appears to contain a secret")
    if contains_prompt_injection(text):
        reasons.append("input appears to contain prompt-injection instructions")
    if len(text.strip()) < 8:
        reasons.append("input is too short to be reviewed as an operational case")
    if not any(hint in lowered for hint in OPERATIONAL_HINTS):
        reasons.append("input does not look like an operational-review case")
    return GuardrailFunctionOutput(
        output_info={"accepted": not reasons, "reasons": reasons},
        tripwire_triggered=bool(reasons),
    )


@tool_input_guardrail(name="retrieval_tool_input_guardrail")
def retrieval_tool_input_guardrail(data: Any) -> ToolGuardrailFunctionOutput:
    raw_args = data.context.tool_arguments or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return ToolGuardrailFunctionOutput.reject_content("Retrieval arguments were not valid JSON.")

    allowed = {"issue_summary", "suspected_case_type", "affected_system", "top_k"}
    unknown = sorted(set(args) - allowed)
    combined = json.dumps(args)
    if unknown:
        return ToolGuardrailFunctionOutput.reject_content(
            f"Retrieval tool accepts only {sorted(allowed)}. Remove: {unknown}."
        )
    if len(combined) > 2500:
        return ToolGuardrailFunctionOutput.reject_content(
            "Retrieval query is too long. Summarize before searching historical cases."
        )
    if contains_secret(combined) or contains_prompt_injection(combined):
        return ToolGuardrailFunctionOutput.reject_content(
            "Retrieval query contained a secret or prompt-injection text and was blocked."
        )
    return ToolGuardrailFunctionOutput.allow({"accepted": True})


@tool_output_guardrail(name="retrieval_tool_output_guardrail")
def retrieval_tool_output_guardrail(data: Any) -> ToolGuardrailFunctionOutput:
    try:
        if isinstance(data.output, RetrievalResult):
            result = data.output
        elif isinstance(data.output, str):
            result = RetrievalResult.model_validate_json(data.output)
        else:
            result = RetrievalResult.model_validate(data.output)
    except ValidationError as exc:
        return ToolGuardrailFunctionOutput.reject_content(
            f"Retrieval output failed schema validation: {exc.errors()}"
        )
    for item in result.evidence:
        if not item.case_id or not item.snippet:
            return ToolGuardrailFunctionOutput.reject_content(
                "Retrieval output must include case_id and snippet for each evidence item."
            )
    return ToolGuardrailFunctionOutput.allow({"evidence_count": len(result.evidence)})


@output_guardrail(name="final_case_review_output_guardrail")
def final_case_review_output_guardrail(context: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
    try:
        if isinstance(output, CaseReviewResult):
            result = output
        elif isinstance(output, str):
            result = CaseReviewResult.model_validate_json(output)
        else:
            result = CaseReviewResult.model_validate(output)
    except Exception as exc:
        return GuardrailFunctionOutput(
            output_info={"valid": False, "error": str(exc)},
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(
        output_info={
            "valid": True,
            "human_review_required": result.human_review_required,
            "evidence_count": len(result.evidence),
        },
        tripwire_triggered=False,
    )

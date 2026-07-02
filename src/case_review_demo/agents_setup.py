from __future__ import annotations

import json
from typing import Any

from agents import Agent
from pydantic import BaseModel

from .guardrails import final_case_review_output_guardrail, operational_case_input_guardrail
from .models import CaseReviewResult, DraftOnlyResponse, ExtractedCase, ReviewDecision


def model_to_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


async def extract_final_output_as_json(result: Any) -> str:
    return model_to_json(result.final_output)


def build_specialist_agents(model: str | None = None) -> dict[str, Agent]:
    intake_agent = Agent(
        name="Intake Extraction Agent",
        model=model,
        instructions=(
            "Extract a messy operational support case into the requested schema. "
            "Use plain, domain-neutral language. If the affected system or user impact is not clear, "
            "leave that field null and add the field name to missing_information. "
            "Do not invent details."
        ),
        output_type=ExtractedCase,
    )

    review_agent = Agent(
        name="Review Classification Agent",
        model=model,
        instructions=(
            "Classify the new operational case using extracted fields and retrieved historical evidence. "
            "Return a cautious, human-reviewable decision. High confidence requires evidence_case_ids. "
            "If no close match was found, set no_close_match_found=true, use low or medium confidence, "
            "and explain uncertainty. Use priority only from: low, medium, high, blocker, unknown."
        ),
        output_type=ReviewDecision,
    )

    draft_agent = Agent(
        name="Draft Response Agent",
        model=model,
        instructions=(
            "Create the final review-ready case note in the requested schema. "
            "Preserve the review decision and evidence. Always set human_review_required=true. "
            "The recommendation is for a human reviewer only: never claim the case was assigned, "
            "escalated, approved, closed, resolved, or written to a production system. "
            "If the case lacks required information, draft a clarification request and keep priority unknown."
        ),
        output_type=CaseReviewResult,
    )

    clarification_draft_agent = Agent(
        name="Clarification Draft Agent",
        model=model,
        instructions=(
            "Write a short clarification-only response for an incomplete operational case. "
            "Do not classify, route, or imply any production action. Ask for the missing information."
        ),
        output_type=DraftOnlyResponse,
    )

    return {
        "intake": intake_agent,
        "review": review_agent,
        "draft": draft_agent,
        "clarification_draft": clarification_draft_agent,
    }


def build_tool_orchestrator(
    *,
    name: str,
    instructions: str,
    tools: list[Any],
    model: str | None = None,
    use_input_guardrail: bool = False,
    use_final_output_guardrail: bool = False,
) -> Agent:
    """Small manager agent that calls exactly one provided tool.

    This keeps the code-controlled state machine easy to read while still showing
    `agent.as_tool()` and function-tool spans in the OpenAI trace.
    """
    return Agent(
        name=name,
        model=model,
        instructions=(
            instructions
            + "\n\nCall the relevant tool exactly once. Do not solve the step directly. "
            + "Return the tool result as the final answer."
        ),
        tools=tools,
        tool_use_behavior="stop_on_first_tool",
        input_guardrails=[operational_case_input_guardrail] if use_input_guardrail else [],
        output_guardrails=[final_case_review_output_guardrail] if use_final_output_guardrail else [],
    )

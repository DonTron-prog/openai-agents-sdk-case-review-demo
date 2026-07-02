from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agents import RunConfig, trace
from pydantic import BaseModel

from .agents_setup import build_specialist_agents
from .models import CaseReviewResult, ExtractedCase, FailureResult, RetrievalResult, ReviewDecision, WorkflowState
from .telemetry import RunLogger, pretty_json
from .tools import build_retrieval_tool
from .workflow import (
    SCENARIOS,
    draft_clarification,
    draft_final,
    extract_case,
    flush_openai_traces,
    retrieve_evidence,
    review_case,
)


class NotebookCaseReviewDemo:
    """Notebook-friendly wrapper around the same SDK steps used by the CLI.

    Use `trace_all()` when you want one clean trace tree for a full walkthrough.
    Use `trace_this_step=True` on individual methods when you want one notebook
    cell per step while keeping a shared `group_id`.
    """

    def __init__(
        self,
        *,
        data_path: str | Path = "data/prepared_cases.csv",
        log_dir: str | Path = "logs",
        model: str | None = "gpt-4.1-mini",
        run_id: str | None = None,
        include_sensitive_trace_data: bool = True,
        max_retries: int = 2,
        fault: str = "none",
    ) -> None:
        self.data_path = Path(data_path)
        self.log_dir = Path(log_dir)
        self.model = model
        self.run_id = run_id or f"notebook-{uuid.uuid4().hex[:8]}"
        self.include_sensitive_trace_data = include_sensitive_trace_data
        self.max_retries = max_retries
        self.fault = fault

        self.specialists = build_specialist_agents(model)
        self.retrieval_tool = build_retrieval_tool(self.data_path)
        self.logger = RunLogger(run_id=self.run_id, log_dir=self.log_dir)
        self.run_config = RunConfig(
            workflow_name="case_review_notebook_demo",
            trace_include_sensitive_data=include_sensitive_trace_data,
            model=model,
            trace_metadata={"demo_run_id": self.run_id, "fault": fault, "interface": "notebook"},
        )

    @property
    def scenarios(self) -> dict[str, str]:
        return SCENARIOS

    @contextmanager
    def trace_all(self, name: str = "case_review_notebook_walkthrough") -> Iterator[None]:
        """Open one top-level trace around several awaited notebook steps."""
        with trace(
            name,
            group_id=self.run_id,
            metadata={"demo_run_id": self.run_id, "fault": self.fault, "interface": "notebook"},
        ):
            yield
        self.flush_traces()

    @contextmanager
    def trace_step(self, step_name: str) -> Iterator[None]:
        """Open a trace for one cell/step while sharing the same group_id."""
        with trace(
            f"case_review_notebook_{step_name}",
            group_id=self.run_id,
            metadata={
                "demo_run_id": self.run_id,
                "fault": self.fault,
                "interface": "notebook",
                "step": step_name,
            },
        ):
            yield
        self.flush_traces()

    async def intake(self, case_text: str, *, trace_this_step: bool = False) -> ExtractedCase:
        if trace_this_step:
            with self.trace_step("01_intake"):
                return await extract_case(case_text, self.specialists, self.run_config, self.logger)
        return await extract_case(case_text, self.specialists, self.run_config, self.logger)

    async def retrieve(self, extracted: ExtractedCase, *, trace_this_step: bool = False) -> RetrievalResult:
        if trace_this_step:
            with self.trace_step("02_retrieval"):
                return await retrieve_evidence(extracted, self.retrieval_tool, self.run_config, self.logger)
        return await retrieve_evidence(extracted, self.retrieval_tool, self.run_config, self.logger)

    async def review(
        self,
        extracted: ExtractedCase,
        retrieval: RetrievalResult,
        *,
        trace_this_step: bool = False,
    ) -> ReviewDecision | FailureResult:
        if trace_this_step:
            with self.trace_step("03_review"):
                return await review_case(
                    extracted,
                    retrieval,
                    self.specialists,
                    self.run_config,
                    self.logger,
                    self.max_retries,
                    self.fault,
                )
        return await review_case(
            extracted,
            retrieval,
            self.specialists,
            self.run_config,
            self.logger,
            self.max_retries,
            self.fault,
        )

    async def draft(
        self,
        case_text: str,
        extracted: ExtractedCase,
        retrieval: RetrievalResult,
        review: ReviewDecision,
        *,
        trace_this_step: bool = False,
    ) -> CaseReviewResult:
        if trace_this_step:
            with self.trace_step("04_draft"):
                return await draft_final(
                    case_text,
                    extracted,
                    retrieval,
                    review,
                    self.specialists,
                    self.run_config,
                    self.logger,
                    self.max_retries,
                )
        return await draft_final(
            case_text,
            extracted,
            retrieval,
            review,
            self.specialists,
            self.run_config,
            self.logger,
            self.max_retries,
        )

    async def clarification(
        self,
        case_text: str,
        extracted: ExtractedCase,
        *,
        trace_this_step: bool = False,
    ) -> CaseReviewResult:
        if trace_this_step:
            with self.trace_step("04_clarification"):
                return await draft_clarification(
                    case_text,
                    extracted,
                    self.specialists,
                    self.run_config,
                    self.logger,
                    self.max_retries,
                )
        return await draft_clarification(
            case_text,
            extracted,
            self.specialists,
            self.run_config,
            self.logger,
            self.max_retries,
        )

    async def run_linear(self, case_text: str) -> CaseReviewResult | FailureResult:
        """Run the happy-path control flow inside whatever trace context the caller opened."""
        extracted = await self.intake(case_text)
        self.show(extracted, "1. Extracted case")

        if not extracted.has_required_fields:
            final = await self.clarification(case_text, extracted)
            self.show(final, "2. Clarification draft")
            return final

        retrieval = await self.retrieve(extracted)
        self.show(retrieval, "2. Retrieval result")

        review = await self.review(extracted, retrieval)
        self.show(review, "3. Review decision")
        if isinstance(review, FailureResult):
            return review

        final = await self.draft(case_text, extracted, retrieval, review)
        self.show(final, "4. Final human-review draft")
        return final

    def show(self, value: Any, title: str | None = None) -> None:
        if title:
            print(f"\n## {title}")
        try:
            from IPython.display import JSON, Markdown, display

            if title:
                display(Markdown(f"### {title}"))
            if isinstance(value, BaseModel):
                display(JSON(value.model_dump(mode="json")))
            else:
                display(JSON(value))
        except Exception:
            print(pretty_json(value))

    def print_trace_hint(self) -> None:
        print("Open the OpenAI traces dashboard and search for:")
        print("  workflow/name: case_review_notebook_demo")
        print(f"  group_id / demo_run_id: {self.run_id}")
        print("Local JSONL log:")
        print(f"  {self.logger.path}")

    def print_log_summary(self) -> None:
        self.logger.print_summary()

    def flush_traces(self) -> None:
        flush_openai_traces()

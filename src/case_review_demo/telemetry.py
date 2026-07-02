from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .models import STEP_LABELS, TelemetryEvent, WorkflowState


class RunLogger:
    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"case_review_trace_{run_id}.jsonl"
        self.events: list[TelemetryEvent] = []

    def record(
        self,
        *,
        state: WorkflowState,
        component: str,
        attempt: int,
        event: str,
        validation_status: str,
        next_state: WorkflowState | None = None,
        retry_reason: str | None = None,
        repair_prompt_used: bool = False,
        latency_ms: int = 0,
        usage: dict[str, int] | None = None,
    ) -> None:
        usage = usage or {}
        row = TelemetryEvent(
            run_id=self.run_id,
            step_label=STEP_LABELS[state],
            internal_state=state.value,
            component=component,
            attempt=attempt,
            event=event,
            retry_reason=retry_reason,
            repair_prompt_used=repair_prompt_used,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            validation_status=validation_status,
            next_step_label=STEP_LABELS[next_state] if next_state else None,
            next_internal_state=next_state.value if next_state else None,
        )
        self.events.append(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(row.model_dump_json() + "\n")

    def print_summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
        except Exception:
            print("\nRun log:")
            for event in self.events:
                print(
                    f"- {event.step_label}: {event.component} attempt {event.attempt} | "
                    f"tokens={event.total_tokens} latency={event.latency_ms}ms | "
                    f"{event.validation_status} -> {event.next_step_label}"
                )
            print(f"\nSaved JSONL log: {self.path}")
            return

        table = Table(title="Local optimization log")
        for column in ["step", "component", "attempt", "tokens", "latency", "validation", "next"]:
            table.add_column(column)
        for event in self.events:
            table.add_row(
                event.step_label,
                event.component,
                str(event.attempt),
                str(event.total_tokens),
                f"{event.latency_ms} ms",
                event.validation_status,
                event.next_step_label or "",
            )
        console = Console()
        console.print(table)
        console.print(f"Saved JSONL log: {self.path}")


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.latency_ms = int((self.end - self.start) * 1000)


def usage_from_result(result: Any) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for response in getattr(result, "raw_responses", []) or []:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
        totals["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
    return totals


def pretty_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), indent=2, default=str)
    return json.dumps(value, indent=2, default=str)

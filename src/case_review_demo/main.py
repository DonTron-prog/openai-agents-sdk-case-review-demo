from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from .telemetry import pretty_json
from .workflow import SCENARIOS, WorkflowSettings, run_case_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI Agents SDK case review tracing demo")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="access_issue",
        help="Prepared classroom scenario to run.",
    )
    parser.add_argument("--case-text", help="Custom case text. Overrides --scenario.")
    parser.add_argument("--data", default="data/extended_prepared_cases.csv", help="Prepared CSV path.")
    parser.add_argument("--log-dir", default="logs", help="Directory for local JSONL run logs.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name used by the Agents SDK.",
    )
    parser.add_argument(
        "--hide-sensitive-trace-data",
        action="store_true",
        help="Keep trace spans but hide model/tool inputs and outputs in OpenAI traces.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--fault",
        choices=[
            "none",
            "review_omit_evidence_once",
            "review_omit_evidence_always",
            "unsafe_confident_no_evidence",
        ],
        default="none",
        help="Deterministic classroom fault injection for retry and fail-closed scenarios.",
    )
    return parser


async def async_main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    case_text = args.case_text or SCENARIOS[args.scenario]

    print("\nCase review tracing demo")
    print(f"Scenario: {args.scenario}")
    print(f"Fault injection: {args.fault}")
    print(f"Input: {case_text}\n")

    settings = WorkflowSettings(
        data_path=Path(args.data),
        log_dir=Path(args.log_dir),
        model=args.model,
        include_sensitive_trace_data=not args.hide_sensitive_trace_data,
        max_retries=args.max_retries,
        fault=args.fault,
    )
    result = await run_case_review(case_text, settings)
    print("\nFinal result")
    print(pretty_json(result))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

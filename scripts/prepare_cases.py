#!/usr/bin/env python3
"""Prepare the classroom CSV from the Mendeley Help Desk Tickets files.

Download the dataset before running this script:
    Mendeley Help Desk Tickets, Version 2, DOI 10.17632/btm76zndnt.2

Expected raw files:
    issues.csv
    sample_utterances.csv

Example:
    python scripts/prepare_cases.py --raw-dir raw/mendeley_helpdesk --output data/prepared_cases.csv --sample-size 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/prepared_cases.csv"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    issues_path = args.raw_dir / "issues.csv"
    utterances_path = args.raw_dir / "sample_utterances.csv"
    if not issues_path.exists() or not utterances_path.exists():
        raise SystemExit(
            f"Expected {issues_path} and {utterances_path}. Download DOI 10.17632/btm76zndnt.2 first."
        )

    issues = pd.read_csv(issues_path).fillna("")
    utterances = pd.read_csv(utterances_path).fillna("")

    issue_id_col = pick_column(utterances, ["issueid", "issue_id", "id"])
    text_col = pick_column(utterances, ["utterance", "text", "body", "message"])
    grouped_text = (
        utterances.groupby(issue_id_col)[text_col]
        .apply(lambda values: "\n".join(str(v).strip() for v in values if str(v).strip()))
        .reset_index()
        .rename(columns={issue_id_col: "id", text_col: "case_text"})
    )

    joined = issues.merge(grouped_text, on="id", how="inner")
    joined = joined[joined["case_text"].str.len() > 30].copy()

    prepared = pd.DataFrame(
        {
            "case_id": joined["id"].astype(str),
            "case_text": joined["case_text"],
            "case_type": get_optional(joined, ["issuetype", "type", "case_type"]),
            "priority": get_optional(joined, ["priority", "Priority"]),
            "assigned_group": get_optional(joined, ["assignee", "assigned_group", "project"]),
            "status": get_optional(joined, ["status", "Status"]),
            "resolution_notes": get_optional(joined, ["resolution", "resolution_notes"]),
            "created_at": get_optional(joined, ["created", "created_at"]),
            "resolved_at": get_optional(joined, ["resolved", "resolved_at"]),
            "processing_steps": get_optional(joined, ["processing_steps", "n_steps", "steps"]),
            "workflow_total_time": get_optional(joined, ["workflow_total_time", "total_time"]),
        }
    ).fillna("")

    if len(prepared) > args.sample_size:
        prepared = prepared.sample(n=args.sample_size, random_state=args.random_seed)
    prepared = prepared.sort_values("case_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(f"Wrote {len(prepared)} prepared rows to {args.output}")

    readme = args.output.parent / "README.md"
    readme.write_text(
        "# Prepared demo cases\n\n"
        "Source: Mendeley Help Desk Tickets, Version 2, DOI `10.17632/btm76zndnt.2`.\n\n"
        "Transformation: grouped `sample_utterances.csv` by issue id, concatenated utterances into "
        "`case_text`, joined to `issues.csv`, renamed columns to a domain-neutral classroom schema, "
        "filtered empty text, and sampled a small subset for fast tracing.\n\n"
        "Known limitation: the public text sample covers only a subset of issues, and field names may "
        "vary across downloaded exports. Inspect this script if a source column is unavailable.\n",
        encoding="utf-8",
    )


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"None of these columns were found: {candidates}. Available: {list(df.columns)}")


def get_optional(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return pd.Series([""] * len(df), index=df.index)


if __name__ == "__main__":
    main()

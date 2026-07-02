#!/usr/bin/env python3
"""Download and merge the public Mendeley + Zenodo helpdesk datasets.

Output schema matches `case_review_demo.data_store.EXPECTED_COLUMNS`, with a few
extra provenance columns that are useful for teaching and auditing retrieval.

Sources:
- Mendeley Help Desk Tickets, Version 2, DOI 10.17632/btm76zndnt.2
- Zenodo Classification of IT Support Tickets, DOI 10.5281/zenodo.7384758

Example:
    python scripts/build_extended_dataset.py --output data/extended_prepared_cases.csv
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

MENDELEY_API = "https://data.mendeley.com/public-api/datasets/btm76zndnt"
ZENODO_API = "https://zenodo.org/api/records/7384758"

MENDELEY_FILES = {"issues.csv", "sample_utterances.csv"}
ZENODO_FILES = {"X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"}

OUTPUT_COLUMNS = [
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
    "source_dataset",
    "source_record_id",
    "source_label",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/extended_prepared_cases.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/case_review_sources"))
    parser.add_argument("--mendeley-max-rows", type=int, default=0, help="0 means all joined Mendeley rows")
    parser.add_argument("--zenodo-max-rows", type=int, default=0, help="0 means all Zenodo rows")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    mendeley_paths = download_mendeley(args.cache_dir)
    zenodo_paths = download_zenodo(args.cache_dir)

    mendeley = build_mendeley_rows(mendeley_paths, max_rows=args.mendeley_max_rows)
    zenodo = build_zenodo_rows(zenodo_paths, max_rows=args.zenodo_max_rows)

    merged = pd.concat([mendeley, zenodo], ignore_index=True)
    merged = merged[OUTPUT_COLUMNS].fillna("")
    merged = merged.drop_duplicates(subset=["case_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"Wrote {len(merged):,} rows to {args.output}")
    print("Rows by source:")
    print(merged["source_dataset"].value_counts().to_string())


def download_mendeley(cache_dir: Path) -> dict[str, Path]:
    payload = read_json(MENDELEY_API)
    paths: dict[str, Path] = {}
    for file_info in payload["files"]:
        filename = file_info["filename"]
        if filename not in MENDELEY_FILES:
            continue
        out = cache_dir / "mendeley" / filename
        download(file_info["content_details"]["download_url"], out)
        paths[filename] = out
    missing = MENDELEY_FILES - set(paths)
    if missing:
        raise RuntimeError(f"Mendeley API did not return expected files: {sorted(missing)}")
    return paths


def download_zenodo(cache_dir: Path) -> dict[str, Path]:
    payload = read_json(ZENODO_API)
    paths: dict[str, Path] = {}
    for file_info in payload["files"]:
        filename = file_info["key"]
        if filename not in ZENODO_FILES:
            continue
        out = cache_dir / "zenodo" / filename
        download(file_info["links"]["self"], out)
        paths[filename] = out
    missing = ZENODO_FILES - set(paths)
    if missing:
        raise RuntimeError(f"Zenodo API did not return expected files: {sorted(missing)}")
    return paths


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "case-review-demo-dataset-builder/1.0"},
    )


def read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(request(url), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out}")
    with urllib.request.urlopen(request(url), timeout=180) as response:
        out.write_bytes(response.read())


def build_mendeley_rows(paths: dict[str, Path], max_rows: int = 0) -> pd.DataFrame:
    issues = pd.read_csv(paths["issues.csv"]).fillna("")
    utterances = pd.read_csv(paths["sample_utterances.csv"]).fillna("")

    issues["_join_id"] = issues["id"].map(canonical_id)
    utterances["_join_id"] = utterances["issueid"].map(canonical_id)
    utterances["actionbody"] = utterances["actionbody"].astype(str).map(clean_text)

    grouped = (
        utterances[utterances["actionbody"].str.len() > 0]
        .sort_values(["_join_id", "comment_seq", "utr_seq"])
        .groupby("_join_id")["actionbody"]
        .apply(lambda values: " ".join(values))
        .reset_index()
        .rename(columns={"actionbody": "case_text"})
    )

    joined = issues.merge(grouped, on="_join_id", how="inner")
    joined = joined[joined["case_text"].str.len() > 30].copy()
    if max_rows > 0 and len(joined) > max_rows:
        joined = joined.sample(n=max_rows, random_state=7)

    rows = pd.DataFrame(
        {
            "case_id": "mendeley_" + joined["_join_id"].astype(str),
            "case_text": joined["case_text"].map(clean_text),
            "case_type": joined["issue_type"].astype(str).replace("", "unknown"),
            "priority": joined["issue_priority"].astype(str).map(normalize_priority),
            "assigned_group": joined["issue_assignee"].astype(str).replace("", "unknown"),
            "status": joined["issue_status"].astype(str).replace("", "unknown"),
            "resolution_notes": joined["issue_resolution"].astype(str),
            "created_at": joined["issue_created"].astype(str),
            "resolved_at": joined["issue_resolution_date"].astype(str),
            "processing_steps": joined["processing_steps"].astype(str),
            "workflow_total_time": joined["wf_total_time"].astype(str),
            "source_dataset": "mendeley_help_desk_tickets_v2",
            "source_record_id": joined["_join_id"].astype(str),
            "source_label": joined["issue_type"].astype(str),
        }
    )
    return rows


def build_zenodo_rows(paths: dict[str, Path], max_rows: int = 0) -> pd.DataFrame:
    x_train = pd.read_csv(paths["X_train.csv"]).fillna("")
    x_test = pd.read_csv(paths["X_test.csv"]).fillna("")
    y_train = pd.read_csv(paths["y_train.csv"]).fillna("")
    y_test = pd.read_csv(paths["y_test.csv"]).fillna("")

    x = pd.concat([x_train, x_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)
    x["_join_id"] = x["id"].map(canonical_id)
    y["_join_id"] = y["id"].map(canonical_id)
    joined = x.merge(y[["_join_id", "category_truth"]], on="_join_id", how="inner")
    joined = joined[joined["text"].astype(str).str.len() > 20].copy()
    if max_rows > 0 and len(joined) > max_rows:
        joined = joined.sample(n=max_rows, random_state=7)

    category = joined["category_truth"].astype(str).replace("", "unknown")
    rows = pd.DataFrame(
        {
            "case_id": "zenodo_" + joined["_join_id"].astype(str),
            "case_text": joined["text"].astype(str).map(clean_text),
            "case_type": category,
            "priority": "unknown",
            "assigned_group": category.map(category_to_group),
            "status": "unknown",
            "resolution_notes": "",
            "created_at": "",
            "resolved_at": "",
            "processing_steps": "",
            "workflow_total_time": "",
            "source_dataset": "zenodo_it_support_tickets_7384758",
            "source_record_id": joined["_join_id"].astype(str),
            "source_label": category,
        }
    )
    return rows


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_text(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_priority(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"lowest", "low", "minor"}:
        return "low"
    if text in {"medium", "normal"}:
        return "medium"
    if text in {"major", "high"}:
        return "high"
    if text in {"highest", "blocker", "critical", "urgent"}:
        return "blocker"
    return "unknown"


def category_to_group(category: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(category).strip().lower()).strip("_")
    return f"{text or 'unknown'}_support"


if __name__ == "__main__":
    main()

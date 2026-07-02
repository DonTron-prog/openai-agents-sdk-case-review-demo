# OpenAI Agents SDK case-review tracing demo

A small, learner-friendly demo of a fixed case-review pipeline:

```text
messy intake -> structured extraction -> evidence retrieval -> review decision -> guarded draft -> human review
```

The code is intentionally straightforward. The Python controller owns the state transitions and retry policy; the Agents SDK supplies specialist agents, agents-as-tools, local function tools, guardrails, and built-in tracing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export OPENAI_API_KEY="..."
```

Optional:

```bash
export OPENAI_MODEL="gpt-4.1-mini"
```

## Run the CLI demo

```bash
python -m case_review_demo.main --scenario access_issue
```

Other useful scenarios:

```bash
python -m case_review_demo.main --scenario missing_information
python -m case_review_demo.main --scenario no_close_match
python -m case_review_demo.main --scenario prompt_injection
```

Deterministic retry/fail-closed examples:

```bash
python -m case_review_demo.main --scenario review_retry --fault review_omit_evidence_once
python -m case_review_demo.main --scenario review_retry --fault review_omit_evidence_always
python -m case_review_demo.main --scenario no_close_match --fault unsafe_confident_no_evidence
```

After each run:

1. Open the OpenAI developer platform traces dashboard.
2. Look for the `case_review_tracing_demo` trace.
3. Inspect the orchestrator spans, specialist agent tool spans, local CSV retrieval tool span, guardrail spans, and any retry attempts.
4. Compare with the local JSONL optimization log in `logs/`.

## Notebook walkthroughs

Two notebooks are included for classroom use:

- `notebooks/case_review_single_trace.ipynb` — runs the whole workflow inside one clean OpenAI trace.
- `notebooks/case_review_step_by_step.ipynb` — runs one step per cell and prints each intermediate output.

Both notebooks hardcode the Agents SDK agent definitions, prompts, retrieval tool, and guardrails directly in notebook cells so learners can inspect the configuration while still comparing outputs to OpenAI traces.

## Files

- `src/case_review_demo/workflow.py` — code-controlled state machine and retry/repair loop.
- `src/case_review_demo/agents_setup.py` — specialist agents and simple tool-calling orchestrators.
- `src/case_review_demo/tools.py` — `@function_tool` retrieval over `data/prepared_cases.csv`.
- `src/case_review_demo/notebook_steps.py` — notebook-friendly step wrapper for trace comparison.
- `src/case_review_demo/guardrails.py` — SDK input, tool, and final output guardrails.
- `src/case_review_demo/models.py` — Pydantic schemas for every checkpoint.
- `scripts/prepare_cases.py` — reproducible preparation script for the Mendeley dataset.
- `data/prepared_cases.csv` — small classroom fixture.

## Teaching notes

- The final state is **Ready for human review**, not case creation.
- Retrieval returns only a few evidence rows, never the full CSV.
- Missing information is a valid path, not a model failure.
- Retry attempts are repair prompts inside the same top-level trace.
- Use `--hide-sensitive-trace-data` to keep spans while hiding model/tool inputs and outputs.

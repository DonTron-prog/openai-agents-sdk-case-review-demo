# Prepared demo cases

`prepared_cases.csv` is a small classroom fixture that follows the schema expected by the live demo:

- `case_id`
- `case_text`
- `case_type`
- `priority`
- `assigned_group`
- `status`
- `resolution_notes`
- `created_at`
- `resolved_at`
- `processing_steps`
- `workflow_total_time`

The production classroom version should be prepared from the public Mendeley Help Desk Tickets dataset, Version 2, DOI `10.17632/btm76zndnt.2`, by joining `sample_utterances.csv` to `issues.csv` as described in `scripts/prepare_cases.py`.

This repository includes a tiny hand-curated fixture so learners can read the code and inspect retrieval behavior without downloading the raw dataset during class. The rows are domain-neutral analogues for the walkthrough scenarios and are not a substitute for the cited public dataset.

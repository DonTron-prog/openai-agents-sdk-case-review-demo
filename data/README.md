# Prepared demo cases

## Default retrieval dataset

`extended_prepared_cases.csv` is the default retrieval dataset for the CLI and notebooks. It is built from public sources:

- Mendeley Help Desk Tickets, Version 2, DOI `10.17632/btm76zndnt.2`
- Zenodo Classification of IT Support Tickets, DOI `10.5281/zenodo.7384758`

Build/rebuild it with:

```bash
python scripts/build_extended_dataset.py --output data/extended_prepared_cases.csv
```

Current generated shape: 2,533 rows:

- 360 joined Mendeley help-desk tickets from `issues.csv` + `sample_utterances.csv`
- 2,173 Zenodo IT support tickets from `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`

## Schema

Both `extended_prepared_cases.csv` and the tiny smoke-test fixture `prepared_cases.csv` include the columns expected by the retrieval code:

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

The extended dataset also includes provenance columns:

- `source_dataset`
- `source_record_id`
- `source_label`

## Tiny fixture

`prepared_cases.csv` is kept as a small classroom smoke-test fixture. It is useful when learners need to inspect every row manually, but it is no longer the default retrieval dataset.

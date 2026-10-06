# Demo run

Recorded 2026-10-06 with `uv run python -m triage.writeup demo --seed 0`: model `qwen3.8:27b-mlx`, fixture seed 0, exit code 0. The CLI's prompts and output are real; the answers are typed by a script (`triage.writeup.scripted_answer`) that edits a date cast which would null values so it accepts mixed formats, rejects deleting rows for a missing value, and approves everything else. Model output varies from run to run, so a new recording will differ.

```text
$ uv run python -m triage.cli run fixtures/orders_0.csv --thread demo-s0-20261006T123745Z
planning with qwen3.8:27b-mlx; this can take a minute
loaded      (216 rows, 9 columns)
planned     (7 ops in 22.7s, attempt 1)
applied     #0 strip_whitespace 'customer': 17 values modified

Approval needed for #1: standardize_missing 'region'
  reason: The 'region' column contains missing-value markers ('?', 'N/A', 'unknown') that should be converted to nulls for proper handling of missing data.
  impact: 16 values nulled
  op:     {"reason":"The 'region' column contains missing-value markers ('?', 'N/A', 'unknown') that should be converted to nulls for proper handling of missing data.","op":"standardize_missing","column":"region","tokens":["?","N/A","unknown"]}
[a]pprove, [r]eject, or [e]dit? a
approved    #1 standardize_missing 'region': 16 values nulled
applied     #1 standardize_missing 'region': 16 values nulled

Approval needed for #2: standardize_missing 'amount'
  reason: The 'amount' column contains missing-value markers ('n/a', 'unknown') that should be converted to nulls before casting to a numeric type.
  impact: 18 values nulled
  op:     {"reason":"The 'amount' column contains missing-value markers ('n/a', 'unknown') that should be converted to nulls before casting to a numeric type.","op":"standardize_missing","column":"amount","tokens":["n/a","unknown"]}
[a]pprove, [r]eject, or [e]dit? a
approved    #2 standardize_missing 'amount': 18 values nulled
applied     #2 standardize_missing 'amount': 18 values nulled
applied     #3 cast_type 'amount': no change

Approval needed for #4: cast_type 'order_date'
  reason: The 'order_date' column is a string but represents dates. Casting it to datetime with the observed format (YYYY-MM-DD) enables date-based analysis.
  impact: 17 values nulled
  op:     {"reason":"The 'order_date' column is a string but represents dates. Casting it to datetime with the observed format (YYYY-MM-DD) enables date-based analysis.","op":"cast_type","column":"order_date","to":"datetime","datetime_format":"%Y-%m-%d"}
[a]pprove, [r]eject, or [e]dit? e
Replacement op as JSON (same shape as 'op' above): {"reason":"The 'order_date' column is a string but represents dates. Casting it to datetime with the observed format (YYYY-MM-DD) enables date-based analysis. Edited: no fixed format, so mixed formats parse.","op":"cast_type","column":"order_date","to":"datetime","datetime_format":null}
edited      #4 cast_type 'order_date' (replaced cast_type)
applied     #4 cast_type 'order_date': 17 values modified

Approval needed for #5: drop_column 'channel'
  reason: The 'channel' column has only one distinct value ('web'), meaning it carries no information and can be dropped to reduce dataset size.
  impact: 1 columns removed
  op:     {"reason":"The 'channel' column has only one distinct value ('web'), meaning it carries no information and can be dropped to reduce dataset size.","op":"drop_column","column":"channel"}
[a]pprove, [r]eject, or [e]dit? a
approved    #5 drop_column 'channel': 1 columns removed
applied     #5 drop_column 'channel': 1 columns removed

Approval needed for #6: dedupe (all columns)
  reason: There are 16 duplicate rows in the dataset. Removing duplicates ensures each order is represented only once.
  impact: 16 rows removed
  op:     {"reason":"There are 16 duplicate rows in the dataset. Removing duplicates ensures each order is represented only once.","op":"dedupe","subset":null,"keep":"first"}
[a]pprove, [r]eject, or [e]dit? a
approved    #6 dedupe (all columns): 16 rows removed
applied     #6 dedupe (all columns): 16 rows removed
validated  
finished    (runs/demo-s0-20261006T123745Z/cleaned_6cd5fd3fe827e006.csv)
done: runs/demo-s0-20261006T123745Z/cleaned_6cd5fd3fe827e006.csv
```

Scored against the fixture's manifest (`triage.faults.check_all` on the final
frame), this run fixed 6 of 8 injected faults. It left the missing `rating`
values and the negative `quantity` values in place: the plan had no op for
either, the same gaps the M7 evaluation found
([`implementation-plan.md`](implementation-plan.md#m7-evaluation-done)).
Op #4 shows the edit path: the model's date cast used one fixed format and
would have turned the 17 dates written as `3 Jan 2025` into nulls; the edited
op has no format, `route` measured it again, found nothing nulled, and applied
it without asking a second time.


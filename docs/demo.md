# Demo run

Recorded 2026-10-07 with `uv run python -m triage.writeup demo --seed 0`: model `qwen3.8:27b-mlx`, fixture seed 0, exit code 0. The CLI's prompts and output are real; the answers are typed by a script (`triage.writeup.scripted_answer`) that edits a date cast which would null values so it accepts mixed formats, rejects deleting rows for a missing value, and approves everything else. Model output varies from run to run, so a new recording will differ.

```text
$ uv run python -m triage.cli run fixtures/orders_0.csv --thread demo-s0-20261007T023944Z
planning with qwen3.8:27b-mlx; this can take a minute
loaded      (216 rows, 9 columns)
planned     (11 ops in 54.7s, attempt 1)
applied     #0 strip_whitespace 'customer': 17 values modified

Approval needed for #1: standardize_missing 'region'
  reason: The 'region' column contains 16 missing-value markers ('?', 'N/A', 'unknown') that should be converted to nulls for proper handling of missing data.
  impact: 16 values nulled
  op:     {"reason":"The 'region' column contains 16 missing-value markers ('?', 'N/A', 'unknown') that should be converted to nulls for proper handling of missing data.","op":"standardize_missing","column":"region","tokens":["?","N/A","unknown"]}
[a]pprove, [r]eject, or [e]dit? a
approved    #1 standardize_missing 'region': 16 values nulled
applied     #1 standardize_missing 'region': 16 values nulled

Approval needed for #2: standardize_missing 'amount'
  reason: The 'amount' column contains 18 missing-value markers ('n/a', 'unknown') that should be converted to nulls before casting to numeric type.
  impact: 18 values nulled
  op:     {"reason":"The 'amount' column contains 18 missing-value markers ('n/a', 'unknown') that should be converted to nulls before casting to numeric type.","op":"standardize_missing","column":"amount","tokens":["n/a","unknown"]}
[a]pprove, [r]eject, or [e]dit? a
approved    #2 standardize_missing 'amount': 18 values nulled
applied     #2 standardize_missing 'amount': 18 values nulled
applied     #3 cast_type 'amount': no change

Approval needed for #4: cast_type 'order_date'
  reason: The 'order_date' column is stored as string but contains date values in ISO format (YYYY-MM-DD). Casting to datetime enables date-based analysis and filtering.
  impact: 17 values nulled
  op:     {"reason":"The 'order_date' column is stored as string but contains date values in ISO format (YYYY-MM-DD). Casting to datetime enables date-based analysis and filtering.","op":"cast_type","column":"order_date","to":"datetime","datetime_format":"%Y-%m-%d"}
[a]pprove, [r]eject, or [e]dit? e
Replacement op as JSON (same shape as 'op' above): {"reason":"The 'order_date' column is stored as string but contains date values in ISO format (YYYY-MM-DD). Casting to datetime enables date-based analysis and filtering. Edited: no fixed format, so mixed formats parse.","op":"cast_type","column":"order_date","to":"datetime","datetime_format":null}
edited      #4 cast_type 'order_date' (replaced cast_type)
applied     #4 cast_type 'order_date': 17 values modified

Approval needed for #5: filter_rows 'quantity'
  reason: The 'quantity' column has 16 negative values, which are likely data entry errors since quantities cannot be negative. Filtering these out ensures data validity.
  impact: 16 rows removed
  op:     {"reason":"The 'quantity' column has 16 negative values, which are likely data entry errors since quantities cannot be negative. Filtering these out ensures data validity.","op":"filter_rows","column":"quantity","operator":">","value":0.0}
[a]pprove, [r]eject, or [e]dit? a
approved    #5 filter_rows 'quantity': 16 rows removed
applied     #5 filter_rows 'quantity': 16 rows removed
applied     #6 impute 'rating': 17 values filled
applied     #7 impute 'region': 15 values filled
applied     #8 impute 'amount': 16 values filled

Approval needed for #9: dedupe (all columns)
  reason: The dataset has 16 duplicate rows. Removing duplicates ensures each order is represented only once, preventing bias in aggregations.
  impact: 16 rows removed
  op:     {"reason":"The dataset has 16 duplicate rows. Removing duplicates ensures each order is represented only once, preventing bias in aggregations.","op":"dedupe","subset":null,"keep":"first"}
[a]pprove, [r]eject, or [e]dit? a
approved    #9 dedupe (all columns): 16 rows removed
applied     #9 dedupe (all columns): 16 rows removed

Approval needed for #10: drop_column 'channel'
  reason: The 'channel' column has only 1 distinct value ('web'), meaning it carries no information for analysis. Dropping it reduces noise and storage.
  impact: 1 columns removed
  op:     {"reason":"The 'channel' column has only 1 distinct value ('web'), meaning it carries no information for analysis. Dropping it reduces noise and storage.","op":"drop_column","column":"channel"}
[a]pprove, [r]eject, or [e]dit? a
approved    #10 drop_column 'channel': 1 columns removed
applied     #10 drop_column 'channel': 1 columns removed
validated  
finished    (runs/demo-s0-20261007T023944Z/cleaned_dca1ba30cab98a95.csv)
done: runs/demo-s0-20261007T023944Z/cleaned_dca1ba30cab98a95.csv
```

Scored against the fixture's manifest (`triage.faults.check_all` on the final
frame), this run fixed all 8 injected faults. Op #5 filters the 16 negative
quantities; its reason quotes the profile's `negative_count`, the field added
after the M7 evaluation found most plans left them alone
([`implementation-plan.md`](implementation-plan.md#after-m8-negative-values-in-the-profile-done)).
Op #4 shows the edit path: the model's date cast used one fixed format and
would have turned the 17 dates written as `3 Jan 2025` into nulls; the edited
op has no format, `route` measured it again, found nothing nulled, and applied
it without asking a second time. Ops #6–#8 impute `rating` (mean), `region`
(mode), and `amount` (median) without asking: filling nulls removes nothing and nulls
nothing, so no `RiskPolicy` limit is exceeded.

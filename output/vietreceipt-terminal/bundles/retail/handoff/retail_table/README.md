# Retail table module for team integration

Send this folder to the teammate implementing header/footer. The ZIP preserves
repository-relative paths. Unpack it **at the VietReceipt repository root**.
It adds an opt-in retail table module and tests; it does not replace the
existing Invoice V2 provider.

```bash
unzip retail-table-integration.zip -d /path/to/vietreceipt
cd /path/to/vietreceipt
.venv/bin/python -m unittest tests.test_retail_table -v
```

Integration API:

```python
from ai.kie.retail_table import extract_retail_table
table = extract_retail_table(evidence, currency="VND")
line_items = table["line_items"]
```

Use the same `evidence` object for header, table, and footer. Each table cell
includes `source_block_ids` and exact OCR `raw_text`. See
`docs/retail-table-handoff.md` in the ZIP for the full contract, signed coupon
handling, validation commands, and limitations.

`metrics-summary.json` is a development replay summary only. The ZIP contains
no receipt images, OCR snapshots, or ground-truth annotations. Those need to be
shared separately if the teammate must reproduce the 20-image evaluation.

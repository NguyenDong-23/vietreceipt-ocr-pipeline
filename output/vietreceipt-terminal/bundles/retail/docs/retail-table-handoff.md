# Retail table extractor handoff

This opt-in component extracts **only product/discount rows** from VietReceipt's
canonical OCR evidence. It does not parse seller/buyer headers, payment footer,
or PDF/image bytes. The existing Invoice V2 baseline remains unchanged.

## Integration boundary

```python
from ai.kie.retail_table import extract_retail_table

table = extract_retail_table(evidence, currency="VND")
rows = table["line_items"]
```

`evidence` is an OCRResult v1.3 or DocumentReader `document-2.0` envelope.
Pass the **same immutable evidence** to the header, table and footer modules.
The caller must establish the currency context; this version accepts explicit
`VND` only and does not infer it from ground truth. The output contains
`schema_version`, `currency_context`, `line_items` and page diagnostics.

Each `line_item` has `line_id`, `description`, `unit`, `quantity`, `unit_price`,
`amount`, `source_block_ids`, and review metadata. Each cell retains exact
`raw_text` and OCR `source_block_ids`, plus a separate `predicted_value`,
`normalized_value`, `value_status`, and `normalization`. Missing printed cells
are `NOT_PRESENT`; malformed OCR numbers remain unnormalized and require review.
The extractor validates all source links before returning.

The coordinator can assemble a receipt-specific result like:

```python
result = {
    "receipt_id": evidence["receipt_id"],
    "ocr_run_id": evidence["ocr_run_id"],
    "header": header_result,
    "line_items": table["line_items"],
    "footer": footer_result,
    "table_diagnostics": table["pages"],
}
```

This is an integration **example**, not an existing combined schema. Keep the
three modules' evidence IDs and currency decision aligned. Do not feed signed
coupon rows into `invoice-kie-result.v2.schema.json` unchanged: that schema
requires nonnegative money, while retail discounts can legitimately be negative.
The team should define a receipt-specific combined contract or a reviewed
discount mapping before wiring this into the worker or API.

## What it handles

- OCR blocks containing multiple column labels, shifted/missing table headers,
  and receipts with no recognized table header.
- Product name above SKU/quantity/price, plus multi-line names and the
  `Giá đăng ký` / `Giá bán` pattern. Registered price is not the sale price.
- Negative coupon/`KM` rows and a footer stop before totals/payment lines.
- Explicit VND integer parsing without repairing OCR digits. Every grouped row
  is flagged for human review because grouping remains heuristic.

## Verification

From the repository root:

```bash
.venv/bin/python -m unittest tests.test_retail_table -v
.venv/bin/python -m scripts.evaluate_retail_20 \
  --labels /home/trungdq/Downloads/ground_truth \
  --images /home/trungdq/Downloads/20_anh \
  --ocr results/ocr_outputs \
  --currency VND \
  --output /tmp/retail-table-replay-new.json
```

The replay only reads frozen OCR snapshots and labels; it does **not** run OCR
or download models. Currency is a command-line input, not taken from gold.
It uses the original 20-image evaluator's name-similarity
row matching and exact normalized-cell policy. The current development replay
on the same 20 inspected images finds 64/64 name-matched rows and 233/320
correct cells (72.81%). Only 2/64 rows have all five cells correct. Breakdown:
description 2/64, unit 63/64, quantity 46/64, unit price 61/64, amount 61/64.
Most description failures come from OCR text errors and strict comparison.

These are **development diagnostics**, not a held-out benchmark. The labels are
first-pass and row matching depends on OCR product names; independent label and
match review plus unseen receipts are required before claiming accuracy. In
particular, the 20 images contain repeated merchant/layout families. Do not
infer generalization from 64/64 on this replay.

Files to integrate: `ai/kie/retail_table.py` is the runtime module;
`tests/test_retail_table.py` is the focused test;
`scripts/evaluate_retail_20.py` is the diagnostic runner. It embeds the
existing 20-image matching policy so the handoff has no dependency on the
locally untracked baseline evaluator. The handoff archive contains these three
files, this document, and no receipt images or labels.

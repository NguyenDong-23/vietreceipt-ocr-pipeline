# VietReceipt OCR terminal pipeline

This folder combines the project's current receipt-reading components into a command-line workflow. It runs the Reader, PP-DocLayoutV3 region detection, VietOCR Header checkpoint, Retail Table extraction, and PaddleOCR Footer extraction. Results and a review page are written to a new run directory.

## Models in this build

- Reader: project DocumentReader OCR pipeline.
- Segmentation: PP-DocLayoutV3.
- Header rerecognition: the existing VietOCR checkpoint in `bundles/HEADER-CHECKPOINT-TICH-HOP/models/vietocr-pos/checkpoint.pth`.
- Footer: PP-OCRv5 server detector with `latin_PP-OCRv5_mobile_rec`.
- Table: Retail Table groups Reader OCR blocks into item fields. On macOS, Apple Vision may reread selected table cells and footer lines.

This release does **not** fine-tune or replace the OCR recognition models. Apple Vision and arithmetic checks are extra rereading/validation steps. The scores below describe this integrated pipeline, not a new OCR model.

## Setup on macOS

Python 3.12 is recommended. At the repository root, run:

```bash
python3.12 -m venv output/pp-doclayout-v3-1511/.venv
output/pp-doclayout-v3-1511/.venv/bin/python -m pip install --upgrade pip
output/pp-doclayout-v3-1511/.venv/bin/python -m pip install -r output/pp-doclayout-v3-1511/requirements-layout.txt
output/pp-doclayout-v3-1511/.venv/bin/python -m pip install -r requirements-document-reader.txt
python3.12 -m venv output/vietreceipt-terminal/.venv-footer
output/vietreceipt-terminal/.venv-footer/bin/python -m pip install --upgrade pip
output/vietreceipt-terminal/.venv-footer/bin/python -m pip install -r output/vietreceipt-terminal/bundles/footer/requirements-footer.txt
swiftc -O output/vietreceipt-terminal/header_vision.swift -o output/vietreceipt-terminal/header_vision
```

The two Python environments are separate because the footer PaddleOCR runtime uses a different pinned Paddle stack. First model use downloads PaddleOCR runtime models to the local cache. PP-DocLayoutV3 and VietOCR checkpoint weights are tracked with Git LFS. Clone with `git lfs install` and `git lfs pull` if Git LFS was not installed automatically.

## Run one image

```bash
out="output/vietreceipt-terminal/runs/one-$(date +%Y%m%d-%H%M%S)"
output/pp-doclayout-v3-1511/.venv/bin/python -B output/vietreceipt-terminal/run_receipt.py \
  --image "/path/to/receipt.jpg" --out "$out"
```

The command checks that both environments exist and then prints each stage, extracted Header fields, table rows, and Footer lines to Terminal. JSON, logs, and segmentation previews are saved under `$out`.

## Re-run the included 20-image example

```bash
out="output/vietreceipt-terminal/runs/git-20-$(date +%Y%m%d-%H%M%S)"
set -o pipefail
output/pp-doclayout-v3-1511/.venv/bin/python -B output/vietreceipt-terminal/run_receipt.py \
  --input-dir output/vietreceipt-terminal/git-20/images --out "$out" --workers 3 2>&1 | tee "$out.log"
output/pp-doclayout-v3-1511/.venv/bin/python -B output/vietreceipt-terminal/git_report.py "$out" output/vietreceipt-terminal/git-20 | tee -a "$out.log"
open "$out/index.html"
```

The included set has 20 images: one public project demo image (`R001.jpg`) and 19 synthetic Vietnamese receipt images (13 clean and 6 blurred). It is not a set of 20 real commercial receipts. The manifest records source commit, source path, and SHA-256 for every image.

## Previously recorded result

The saved run from 2026-11-16 completed all five stages for 20/20 images. On the 19 synthetic images with labels, the pipeline covered 163/163 Header fields, 70/70 item rows, and 76/76 Footer fields. It read 159/163 Header fields exactly, left 0 Header fields as `UNKNOWN`, read 39/70 item descriptions exactly and 70/70 item amounts exactly. The invoice total matched on 19/19 synthetic images. These are results for this small included set and this integrated pipeline; they do not establish performance on unseen real receipts.

Open `../../reports/table-footer-20-20261116/index.html` for the saved per-image report. The Terminal log and browser screenshots are beside it.

## Source and limitations

The source project is [vietreceipt-team/vietreceipt](https://github.com/vietreceipt-team/vietreceipt). The included PP-DocLayoutV3 bundle is adapted for this integration. Keep the existing model provenance and licenses with redistributed checkpoints. This is an experimental command-line integration; it does not replace the project's production API or frontend. Fields that are missing, unreadable, or ambiguous should remain flagged for review.

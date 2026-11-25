"""Run Reader, PP layout, VietOCR Header, retail Table and Paddle Footer in Terminal."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import unicodedata

from stages import BASE, HEADER_SAMPLES, MIME, ROOT, load, save

HEADER_LABELS = {
    "invoice_template_number": "Mẫu số", "invoice_symbol": "Ký hiệu",
    "invoice_number": "Số hóa đơn", "invoice_date": "Ngày hóa đơn",
    "seller_name": "Tên người bán", "seller_tax_id": "MST người bán",
    "seller_address": "Địa chỉ người bán", "buyer_name": "Tên người mua",
    "buyer_tax_id": "MST người mua",
}
HEADER_ENGINES = {"reader": "Reader", "apple-vision": "Apple Vision",
                  "tesseract-vie": "Tesseract tiếng Việt", "vietocr-checkpoint": "VietOCR"}
TABLE_LABELS = {
    "description": "Tên hàng", "unit": "ĐVT", "quantity": "SL",
    "unit_price": "Đơn giá", "amount": "Thành tiền",
}
REGION_LABELS = {"header": "Đầu biên lai", "table": "Bảng mặt hàng", "footer": "Vùng cuối"}
PAYMENT_SOURCES = {"READER_PP_FOOTER": "Reader + PP Footer", "PADDLE_FOOTER": "Paddle Footer",
                   "VISION_READER_FOOTER": "Apple Vision + Reader"}


def clean(value):
    raw = str(value)
    return "".join(" " if unicodedata.category(char) in {"Cc", "Cf"} else char for char in raw).strip()


def cell_text(cell):
    value = cell.get("normalized_value")
    if value is None:
        value = cell.get("predicted_value")
    shown = clean(value) if value is not None else "—"
    return f'{shown} [{cell["value_status"]}]'


def call(stage, python, image, out):
    log = out / f"{stage}.log"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        with log.open("w", encoding="utf-8") as file:
            result = subprocess.run(
                [str(python), "-B", str(ROOT / "stages.py"), stage, str(image), str(out)],
                env=env, stdout=file, stderr=subprocess.STDOUT, timeout=300,
            )
    except subprocess.TimeoutExpired:
        return {"status": "FAILED", "error": "Quá thời gian chờ 300 giây", "log": log.name}
    if result.returncode != 0:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        message = lines[-1] if lines else "Không có thông báo"
        return {"status": "FAILED", "error": clean(message), "log": log.name, "exit_code": result.returncode}
    return {"status": "OK", "log": log.name}


def reconcile(result):
    reader = result["reader"]
    if result.get("layout") and result["layout"]["provenance"]["ocr"] != reader["pages"][0]["evidence"]["engine"]:
        raise ValueError("Nguồn OCR của segmentation không khớp Reader.")
    if selection := result.get("header_selection"):
        if selection["parent_ocr_run_id"] != reader["ocr_run_id"] or selection["source_sha256"] != result["image_sha256"]:
            raise ValueError("Nguồn OCR của Header không khớp ảnh hoặc Reader.")
    for section in ("header", "table"):
        payload = result.get(section)
        if not payload:
            continue
        if payload.get("receipt_id", reader["receipt_id"]) != reader["receipt_id"]:
            raise ValueError(f"{section} dùng receipt_id khác.")
        evidence = result["header_evidence"] if section == "header" else reader
        if section == "header":
            known_by_source = {
                name: (source["ocr_run_id"], {b["block_id"] for page in source["pages"]
                                             for b in page["evidence"]["blocks"]})
                for name, source in (("vietocr-checkpoint", evidence), ("reader", reader))
            }
            for source, supplemental in payload.get("supplemental_ocr", {}).items():
                ids = set(supplemental["reads"])
                if supplemental["parent_ocr_run_id"] != reader["ocr_run_id"] or ids - known_by_source["reader"][1]:
                    raise ValueError("Kết quả OCR đọc lại không khớp khung chữ Reader.")
                known_by_source[source] = (supplemental["ocr_run_id"], ids)
            for name, cell in payload["fields"].items():
                info = payload.get("field_sources", {}).get(name, {})
                source = info.get("engine", "vietocr-checkpoint")
                run_id, known = known_by_source[source]
                if info.get("ocr_run_id", run_id) != run_id or set(cell["source_block_ids"]) - known:
                    raise ValueError(f"Header {name} có block ID không tồn tại trong nguồn OCR.")
            for alternatives in payload.get("review_alternatives", {}).values():
                for source, candidate in alternatives.items():
                    run_id, known = known_by_source[source]
                    if candidate["ocr_run_id"] != run_id or set(candidate["source_block_ids"]) - known:
                        raise ValueError("Ứng viên Header không khớp nguồn OCR.")
            for record in payload.get("receipt_metadata", {}).values():
                if set(record["source_block_ids"]) - known_by_source["vietocr-checkpoint"][1]:
                    raise ValueError("Thông tin giao dịch không khớp nguồn OCR.")
        else:
            known = {block["block_id"] for page in evidence["pages"] for block in page["evidence"]["blocks"]}
            fields = (row[key] for row in payload["line_items"] for key in TABLE_LABELS)
            if any(block_id not in known for cell in fields for block_id in cell["source_block_ids"]):
                raise ValueError(f"{section} có block ID không tồn tại trong nguồn OCR.")
            if refinement := payload.get("refinement"):
                if refinement["reader_ocr_run_id"] != reader["ocr_run_id"] or set(refinement["reads"]) - known:
                    raise ValueError("Các ô Table được đọc lại không khớp nguồn Reader.")
    links = result.get("layout_links")
    table = result.get("table")
    if links and table:
        original_tables = {
            block["block_id"] for block in links["blocks"]
            if any(match["label"] == "table" for match in block["matches"])
        }
        result["table_region_check"] = {
            "rows_with_evidence_outside_pp_table": [
                row["line_id"] for row in table["line_items"]
                if set(row["source_block_ids"]) - original_tables
            ],
            "policy": "diagnostic_only",
        }
    if display := result.get("footer", {}).get("display"):
        reader_ids = {b["block_id"] for page in reader["pages"] for b in page["evidence"]["blocks"]}
        if display["reader_ocr_run_id"] != reader["ocr_run_id"] or any(
            part["block_id"] not in reader_ids for row in display["rows"] for part in row["segments"]
        ):
            raise ValueError("Các dòng Footer đọc lại không khớp nguồn Reader.")


def print_result(result):
    stages = result["stages"]
    for stage, state in stages.items():
        print(f"  {stage:<14} {state['status']}", flush=True)
        if state.get("error"):
            print(f"    {state['error']}" + (f" · Log: {state['log']}" if state.get("log") else ""))

    print("\n=== PHÂN VÙNG · PP-DocLayoutV3 ===")
    if result.get("layout"):
        for page in result["layout"]["pages"]:
            for region in page["regions"]:
                coords = ", ".join(f"{x:.1f}" for x in region["bbox_original_pixels"])
                name = REGION_LABELS.get(region["label"], region["label"])
                print(f"  {name:<20} {region['region_id']} [{coords}]")
        warnings = stages["segmentation"].get("warnings", [])
        print(f"  Cảnh báo: {', '.join(warnings) if warnings else 'Không có'}")
        counts = result["layout_links"]["counts"]
        print(f"  Liên kết OCR: {counts['ASSIGNED']} đã gán, {counts['AMBIGUOUS']} mơ hồ, {counts['UNASSIGNED']} chưa gán.")
    else:
        print("  Chưa có kết quả.")

    print("\n=== HEADER · VietOCR checkpoint ===")
    if result.get("header"):
        for key, label in HEADER_LABELS.items():
            cell = result["header"]["fields"][key]
            field_source = result["header"].get("field_sources", {}).get(key, {})
            source = HEADER_ENGINES.get(field_source.get("engine"))
            source_hint = f" · Nguồn {source}" if source else ""
            print(f"  {label:<20} {cell_text(cell)}{source_hint}")
            alternatives = result["header"].get("review_alternatives", {}).get(key, {})
            if alternatives:
                print("    Đối chiếu " + " · ".join(f"{engine} {row['value']}" for engine, row in alternatives.items()))
        for key, record in result["header"].get("receipt_metadata", {}).items():
            label = {"transaction_id": "Mã giao dịch", "transaction_date": "Ngày giao dịch"}.get(key, key)
            print(f"  {label:<20} {record['value']} [Thông tin giao dịch]")
        meta = result["header_checkpoint_provenance"]
        print(f"  Số khung đọc lại: {meta['ocr_blocks_processed']}")
        for warning in result["header"].get("supplemental_ocr_warnings", []):
            print(f"  OCR bổ sung: {clean(warning)}")
    else:
        print("  Chưa có kết quả.")

    print("\n=== BẢNG MẶT HÀNG · Retail Table ===")
    if result.get("table"):
        items = result["table"]["line_items"]
        print(f"  {len(items)} dòng")
        print("  " + " | ".join(TABLE_LABELS.values()))
        for index, row in enumerate(items, 1):
            print(f"  {index}. " + " | ".join(cell_text(row[key]) for key in TABLE_LABELS))
        if refinement := result["table"].get("refinement"):
            print(f"  Apple Vision đọc lại {len(refinement['changes'])} ô.")
        outside = result.get("table_region_check", {}).get("rows_with_evidence_outside_pp_table", [])
        if outside:
            print("  Cần rà lại vùng PP ở dòng: " + ", ".join(map(clean, outside)))
    else:
        print("  Chưa có kết quả.")

    print("\n=== FOOTER ===")
    if footer := result.get("footer"):
        print(f"  Vùng đọc trên ảnh đã chỉnh chiều: {footer['box']}")
        if display := footer.get("display"):
            print("  Dòng ghép từ Apple Vision và Reader")
            for row in display["rows"]:
                print(f"  {clean(row['text'])}")
        else:
            for line in footer["lines"]:
                print(f"  {clean(line['text'])} | Tin cậy PaddleOCR {line['score']:.2%}")
        if not footer["lines"]:
            print("  Chưa tìm được dòng footer.")
        if payment := footer.get("payment"):
            amount = payment["amount_vnd"]
            amount_text = f"{amount:,} VND".replace(",", ".") if amount is not None else "—"
            source = payment.get("source")
            engine = PAYMENT_SOURCES.get(source, source or "OCR")
            print(f"  Số tiền thanh toán: {amount_text} [{payment['status']}] · Nguồn {engine}")
    else:
        print("  Chưa có kết quả.")
    print(f"\nKết quả lưu tại {result['output_dir']} · {result['elapsed_seconds']:.1f}s", flush=True)


def run(image, out, core_python, footer_python):
    start = perf_counter()
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    out.mkdir()
    states = {"reader": call("core", core_python, image, out)}
    if (out / "core-stages.json").is_file():
        states.update(load(out / "core-stages.json"))
    for stage in ("segmentation", "table"):
        states.setdefault(stage, {"status": "SKIPPED", "error": "Reader chưa tạo evidence hợp lệ"})
    for stage, python in (("header", core_python), ("footer", footer_python)):
        if states["reader"]["status"] == "OK":
            states[stage] = call(stage, python, image, out)
        else:
            states[stage] = {"status": "SKIPPED", "error": "Reader chưa tạo evidence hợp lệ"}

    result = {"schema_version": "vietreceipt-terminal-pilot-1.0", "image": str(image),
              "image_sha256": digest, "output_dir": str(out), "stages": states}
    outputs = {
        "reader": "reader-document.json", "reader_metadata": "reader-metadata.json",
        "layout": "pp-layout.json", "layout_links": "pp-region-links.json",
        "header": "header.json", "header_evidence": "header-evidence.json",
        "header_selection": "header-selection.json",
        "header_checkpoint_provenance": "header-checkpoint-provenance.json",
        "table": "table.json", "footer": "footer.json",
    }
    for key, name in outputs.items():
        if (out / name).is_file():
            result[key] = load(out / name)
    if states["reader"]["status"] == "OK":
        try:
            if hashlib.sha256(image.read_bytes()).hexdigest() != digest:
                raise ValueError("Ảnh thay đổi trong lúc xử lý.")
            reconcile(result)
        except ValueError as exc:
            states["integration"] = {"status": "FAILED", "error": str(exc)}
    result["elapsed_seconds"] = round(perf_counter() - start, 3)
    save(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, action="append")
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--sample", action="append", metavar="HEADER_SAMPLE_ID")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    if args.sample:
        images = [HEADER_SAMPLES / (name + ".jpg") for name in args.sample]
    elif args.input_dir:
        if not args.input_dir.is_dir():
            parser.error("Thư mục ảnh không tồn tại.")
        images = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in MIME)
    else:
        images = args.image
    if not images or any(not p.is_file() or p.suffix.lower() not in MIME for p in images):
        parser.error("Cần đường dẫn ảnh JPG/JPEG/PNG hợp lệ.")
    images = [p.resolve() for p in images]
    core_python = BASE / ".venv/bin/python"
    footer_python = ROOT / ".venv-footer/bin/python"
    if not core_python.is_file() or not footer_python.is_file():
        parser.error("Thiếu môi trường Python. Xem README.md.")
    out = args.out.resolve()
    if out.exists():
        parser.error("Hãy dùng một thư mục đầu ra mới.")
    out.mkdir(parents=True)
    summary = []
    print(f"Đang xử lý {len(images)} ảnh với {args.workers} tiến trình song song.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(run, image, out / f"{i:03d}-{image.stem}", core_python, footer_python)
                for i, image in enumerate(images, 1)]
        for job in jobs:
            result = job.result()
            print(f"\n>>> {len(summary) + 1}/{len(images)} · {clean(Path(result['image']).name)}", flush=True)
            print_result(result)
            states = result["stages"]
            summary.append({"image": result["image"], "output": result["output_dir"], "stages": states,
                            "complete": all(state["status"] == "OK" for state in states.values())})
    complete = sum(row["complete"] for row in summary)
    save(out / "summary.json", {"input_images": len(images), "complete": complete, "runs": summary})
    print(f"\nHoàn tất đủ bước: {complete}/{len(images)} ảnh")
    print(f"Thống kê: {out / 'summary.json'}")
    return 0 if complete == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())

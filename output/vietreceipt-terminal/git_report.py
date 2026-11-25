"""Compare the integrated run with the Git handoff labels and render local HTML."""
import argparse
from collections import Counter
from hashlib import sha256
from html import escape
from pathlib import Path
import shutil
import unicodedata
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from footer_amount import read_payable
from run_receipt import HEADER_ENGINES, HEADER_LABELS, PAYMENT_SOURCES, REGION_LABELS, TABLE_LABELS, cell_text
from stages import load, save

FOOTER_FIELDS = {"subtotal", "tax_amount", "total_amount", "currency"}
KINDS = {"demo": "Ảnh demo R001", "synthetic_clean": "Tổng hợp · ảnh sạch",
         "synthetic_blur": "Tổng hợp · ảnh làm mờ"}


def h(value):
    return escape(str(value))


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def coverage(page, bounds, label):
    w, height = page["width_px"], page["height_px"]
    x1, y1, x2, y2 = bounds
    box = np.float32([[x1*w, y1*height], [x2*w, y1*height],
                      [x2*w, y2*height], [x1*w, y2*height]])
    area = cv2.contourArea(box)
    hits = [cv2.intersectConvexConvex(box, np.float32(region["polygon_original_pixels"]))[0]
            for region in page["regions"] if region["label"] == label]
    return min(1.0, max(hits, default=0.0) / area) if area else 0.0


def assess(result, sample, gold_dir):
    urls = sample.get("reference", {})
    if "source-gold.json" not in urls:
        return {"inspection": "Có bản chép tay để xem từng dòng, chưa có nhãn tọa độ ba vùng."}
    ref = load(gold_dir / urls["source-gold.json"])
    fields = load(gold_dir / urls["source-regions.json"])
    rows = load(gold_dir / urls["source-row-regions.json"])
    page = result.get("layout", {}).get("pages", [{}])[0]
    blocks = [(label, field["bounds"], field["cell"]) for field in fields
              if field["cell"].startswith("fields.")
              for key in [field["cell"].split(".")[1]]
              for label in (["header"] if key in HEADER_LABELS else
                            ["footer"] if key in FOOTER_FIELDS else [])]
    blocks.extend(("table", row["bounds"], row["line_id"]) for row in rows)
    spatial = {label: {"covered": 0, "reference": 0} for label in REGION_LABELS}
    misses = []
    for label, bounds, cell in blocks:
        amount = coverage(page, bounds, label) if page else 0
        spatial[label]["covered"] += amount >= 0.9
        spatial[label]["reference"] += 1
        if amount < 0.9:
            misses.append({"label": label, "cell": cell, "coverage": round(amount, 3)})
    expected_header = {key: value["normalized_value"] for key, value in ref["fields"].items()
                       if key in HEADER_LABELS and value["annotation_status"] == "PRESENT"}
    observed_header = result.get("header", {}).get("fields", {})
    wrong_fields = {key: value for key, value in expected_header.items()
                    if normalized(observed_header.get(key, {}).get("normalized_value")) != normalized(value)}
    header_unknown = sum(observed_header.get(key, {}).get("value_status", "UNKNOWN") == "UNKNOWN"
                         for key in expected_header)
    table_rows = result.get("table", {}).get("line_items", [])
    expected_amounts = [row["amount"]["normalized_value"] for row in ref["line_items"]]
    table_amount_exact = sum(row["amount"].get("normalized_value") == amount
                             for row, amount in zip(table_rows, expected_amounts))
    table_description_exact = sum(
        normalized(row["description"].get("normalized_value")) == normalized(gold["description"]["normalized_value"])
        for row, gold in zip(table_rows, ref["line_items"])
    )
    expected_total = ref["fields"]["total_amount"]["normalized_value"]
    footer = result.get("footer", {})
    payment = footer["payment"] if "payment" in footer else read_payable(footer.get("text", ""))
    total_read = payment["amount_vnd"]
    return {
        "spatial": spatial, "spatial_misses": misses,
        "header_exact": len(expected_header) - len(wrong_fields), "header_reference": len(expected_header),
        "header_unknown": header_unknown,
        "header_misses": wrong_fields,
        "table_read_rows": len(table_rows),
        "table_reference_rows": len(ref["line_items"]),
        "table_amount_exact": table_amount_exact,
        "table_description_exact": table_description_exact,
        "total_read": total_read, "total_reference": expected_total,
        "total_exact": total_read == expected_total,
    }


def image_assets(folder, original, footer):
    with Image.open(original) as image:
        display = ImageOps.exif_transpose(image).convert("RGB")
        display.thumbnail((850, 1000))
        display.save(folder / "report-source.jpg", quality=89)
    upright = folder / "footer-upright.png"
    if footer.get("box") and upright.is_file():
        with Image.open(upright) as image:
            display = image.convert("RGB")
        draw = ImageDraw.Draw(display)
        draw.rectangle(footer["box"], outline="#d00061", width=5)
        for line in footer["lines"]:
            draw.line([tuple(point) for point in line["poly"]] + [tuple(line["poly"][0])],
                      fill="#037f77", width=2)
        display.thumbnail((850, 1000))
        display.save(folder / "report-footer.jpg", quality=89)


def photo(folder, name, caption):
    if not (folder / name).is_file():
        return f"<figure><figcaption>{h(caption)} · chưa có ảnh</figcaption></figure>"
    url = quote(folder.name + "/" + name)
    return (f'<figure><a href="{url}" target="_blank"><img loading="lazy" src="{url}" '
            f'alt="{h(caption)}"></a><figcaption>{h(caption)}</figcaption></figure>')


def render_case(result, sample, check, folder):
    name = Path(result["image"]).name
    tags = " · ".join(f"{stage} {state['status']}" for stage, state in result["stages"].items())
    header_result = result.get("header", {})
    cells = header_result.get("fields", {})
    header = []
    for key, label in HEADER_LABELS.items():
        if key not in cells:
            continue
        cell_details = []
        if source := HEADER_ENGINES.get(header_result.get("field_sources", {}).get(key, {}).get("engine")):
            cell_details.append(f"Nguồn {source}")
        for engine, candidate in header_result.get("review_alternatives", {}).get(key, {}).items():
            cell_details.append(f"{engine} đọc {candidate['value']}")
        if key in check.get("header_misses", {}):
            cell_details.append(f"Nhãn {check['header_misses'][key]}")
        hints = f"<small>{h(' · '.join(cell_details))}</small>" if cell_details else ""
        header.append(f"<dt>{h(label)}</dt><dd>{h(cell_text(cells[key]))}{hints}</dd>")
    for key, label in (("transaction_id", "Mã giao dịch"), ("transaction_date", "Ngày giao dịch")):
        if record := header_result.get("receipt_metadata", {}).get(key):
            header.append(f"<dt>{label}</dt><dd>{h(record['value'])} <small>Dòng giao dịch ở Footer</small></dd>")
    items = result.get("table", {}).get("line_items", [])
    columns = "".join(f"<th>{h(label)}</th>" for label in TABLE_LABELS.values())
    body = "".join("<tr>" + "".join(f"<td>{h(cell_text(row[key]))}</td>" for key in TABLE_LABELS)
                   + "</tr>" for row in items)
    footer = result.get("footer", {})
    lines = "".join(f"<li>{h(line['text'])} <small>{line['score']:.1%}</small></li>" for line in footer.get("lines", []))
    footer_result = f"<ul>{lines}</ul>"
    if display := footer.get("display"):
        rows = "".join(f"<li>{h(row['text'])}</li>" for row in display["rows"])
        footer_result = (f'<p class="meta">Dòng ghép từ Apple Vision và Reader</p><ul>{rows}</ul>'
                         f"<details><summary>Xem chữ OCR gốc của PaddleOCR</summary><ul>{lines}</ul></details>")
    if (payment := footer.get("payment", {})).get("status") == "PRESENT":
        amount = f"{payment['amount_vnd']:,}".replace(",", ".")
        source = PAYMENT_SOURCES.get(payment.get("source"), "OCR")
        footer_result = f"<p><strong>Tổng tiền thanh toán {h(amount)} VND</strong> <small>Nguồn {h(source)}</small></p>" + footer_result
    refinement_count = len(result.get("table", {}).get("refinement", {}).get("changes", []))
    table_note = f'<p class="meta">Apple Vision đọc lại {refinement_count} ô.</p>' if refinement_count else ""
    reviewed = h(check.get("inspection", ""))
    if spatial := check.get("spatial"):
        reviewed = " · ".join(f"{h(REGION_LABELS[k])} {v['covered']}/{v['reference']}" for k, v in spatial.items())
        total_read = check['total_read'] if check['total_read'] is not None else "chưa đối chiếu được"
        reviewed += (f"<br>Dòng hàng đọc được {check['table_read_rows']}/{check['table_reference_rows']} · "
                     f"Thành tiền từng dòng khớp {check['table_amount_exact']}/{check['table_reference_rows']} · "
                     f"Tổng tiền OCR {h(total_read)} · Nhãn {h(check['total_reference'])}")
    if footer.get("payment", {}).get("source") == "READER_PP_FOOTER":
        payment = footer["payment"]
        reviewed += (f"<br>Tổng tiền {payment['amount_vnd']:,} VND lấy từ dòng Reader trong vùng PP Footer · "
                     + h(" / ".join(payment["source_rows"])))
    warnings = result["stages"]["segmentation"].get("warnings", [])
    if changes := check.get("region_changes"):
        reviewed += "<br>" + " · ".join(
            f"{h(row['text'])} chuyển từ {h(REGION_LABELS[row['before']])} sang {h(REGION_LABELS[row['after']])}"
            for row in changes)
    links = [f'<a href="{quote(folder.name)}/result.json">Dữ liệu chạy</a>',
             f'<a href="{h(sample["url"])}">Nguồn Git</a>']
    if (folder / "reference-transcription.txt").is_file():
        links.append(f'<a href="{quote(folder.name)}/reference-transcription.txt">Bản chép tay</a>')
    if (folder / "report-pp-before.jpg").is_file():
        links.append(f'<a href="{quote(folder.name)}/report-pp-before.jpg">Ảnh phân vùng trước khi sửa</a>')
    return f"""<article id="{h(Path(name).stem)}">
    <h2>{h(name)}</h2><p class="meta">{h(KINDS[sample['kind']])} · {h(tags)} · {result["elapsed_seconds"]:.1f} giây</p>
    <p>{reviewed}</p>
    {f'<p>Cảnh báo {h(", ".join(warnings))}</p>' if warnings else ''}
    <div class="photos">{photo(folder, 'report-source.jpg', 'Ảnh từ Git')}
    {photo(folder, 'pp-preview.jpg', 'Xanh dương · Header | Vàng · Table | Hồng · Footer')}
    {photo(folder, 'report-footer.jpg', 'Vùng đọc và dòng chữ của PaddleOCR Footer')}</div>
    <details><summary>Xem dữ liệu OCR và trích xuất</summary>
    <h3>Header</h3><dl>{''.join(header)}</dl>
    {('<p><small>UNKNOWN · Chưa tìm thấy căn cứ. UNREADABLE · Có nhãn nhưng giá trị chưa đọc được. '
      'AMBIGUOUS · Giá trị cần xác nhận.</small></p>'
      if any(cell['value_status'] in {'UNKNOWN', 'AMBIGUOUS', 'UNREADABLE'} for cell in cells.values()) else '')}
    <h3>Bảng mặt hàng</h3>{table_note}<div class="scroll"><table><thead><tr>{columns}</tr></thead><tbody>{body}</tbody></table></div>
    <h3>Footer</h3>{footer_result}</details>
    <p class="links">{" · ".join(links)}</p></article>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    run, dataset = args.run.resolve(), args.dataset.resolve()
    summary = load(run / "summary.json")
    manifest = {row["image"]: row for row in load(dataset / "manifest.json")["samples"]}
    spatial = {label: Counter() for label in REGION_LABELS}
    stage_counts = Counter()
    labeled = Counter()
    cases, cards, menu = [], [], []
    for job in summary["runs"]:
        folder = run / Path(job["output"]).name
        result = load(folder / "result.json")
        name = Path(result["image"]).name
        sample = manifest[name]
        original = dataset / "images" / name
        if sha256(original.read_bytes()).hexdigest() != sample["sha256"] or result["image_sha256"] != sample["sha256"]:
            raise ValueError(f"Ảnh {name} không khớp Git manifest")
        for stage, state in result["stages"].items():
            if state["status"] == "OK":
                stage_counts[stage] += 1
        check = assess(result, sample, dataset / "gold")
        if args.baseline and (folder / "pp-prediction.json").is_file():
            previous = args.baseline / folder.name / "pp-prediction.json"
            if previous.is_file():
                before, after = load(previous), load(folder / "pp-prediction.json")
                if before["input_sha256"] == after["input_sha256"]:
                    before_rows = {row["text"]: row["label"] for row in before["rows"]}
                    check["region_changes"] = [
                        {"text": row["text"], "before": before_rows[row["text"]], "after": row["label"]}
                        for row in after["rows"] if row["text"] in before_rows
                        and before_rows[row["text"]] != row["label"]
                    ]
                    if check["region_changes"]:
                        shutil.copyfile(previous.parent / "pp-preview.jpg", folder / "report-pp-before.jpg")
        for label, tally in check.get("spatial", {}).items():
            spatial[label].update(tally)
        if "header_exact" in check:
            labeled["images"] += 1
            for key in ("header_exact", "header_reference", "header_unknown", "total_exact", "table_amount_exact",
                        "table_description_exact",
                        "table_reference_rows"):
                labeled[key] += check[key]
            labeled["table_count_exact"] += check["table_read_rows"] == check["table_reference_rows"]
        cases.append({"image": name, "source": sample, "evaluation": check})
        image_assets(folder, original, result.get("footer", {}))
        if sample["kind"] == "demo":
            shutil.copyfile(dataset / "gold" / sample["reference"]["transcription.txt"],
                            folder / "reference-transcription.txt")
        cards.append(render_case(result, sample, check, folder))
        stem = Path(name).stem
        menu.append(f'<a href="#{h(stem)}">{h(stem)}</a>')
        print(f"{name:<35} " + (" | ".join(f"{k} {v['covered']}/{v['reference']}"
                                         for k, v in check["spatial"].items())
                                if "spatial" in check else check["inspection"]))
        for row in check.get("region_changes", []):
            print(f"  {row['before']} → {row['after']}  {row['text']}")
    save(run / "git-evaluation.json", {"stage_ok": stage_counts, "spatial": spatial,
                                      "labeled_extraction": labeled, "cases": cases})
    overview = " · ".join(f"{k} {v}/{len(cases)}" for k, v in stage_counts.items())
    coverage_text = " · ".join(f"{REGION_LABELS[k]} {v['covered']}/{v['reference']}" for k, v in spatial.items())
    extraction = (f"Header khớp {labeled['header_exact']}/{labeled['header_reference']} trường · "
                  f"UNKNOWN trên trường có nhãn {labeled['header_unknown']}/{labeled['header_reference']} · "
                  f"Số dòng hàng khớp trên {labeled['table_count_exact']}/{labeled['images']} ảnh · "
                  f"Tên hàng khớp {labeled['table_description_exact']}/{labeled['table_reference_rows']} dòng theo thứ tự · "
                  f"Thành tiền khớp {labeled['table_amount_exact']}/{labeled['table_reference_rows']} dòng theo thứ tự · "
                  f"Tổng tiền khớp trên {labeled['total_exact']}/{labeled['images']} ảnh")
    log = run.with_suffix(".log")
    if log.is_file() and not (run / "terminal.log").is_file():
        shutil.copyfile(log, run / "terminal.log")
    log_link = '<a href="terminal.log">Log Terminal đầy đủ</a> · ' if (run / "terminal.log").is_file() else ""
    run_note = ("Chạy lại Table và Footer trên ảnh gốc với khung Reader đã lưu."
                if summary.get("rerun_stages") == ["table_refinement", "footer_refinement"] else "")
    html = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>VietReceipt · Kiểm thử 20 ảnh từ Git</title>
    <style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f1f5f2;color:#142620;font:16px/1.55 system-ui,sans-serif}}
    main{{max-width:1150px;margin:auto;padding:38px 20px}}h1{{font-size:clamp(30px,5vw,49px);line-height:1.08;margin:.2em 0 .5em}}
    h2{{margin:0;overflow-wrap:anywhere}}a{{color:#00634e}}.overview,article{{background:#fff;border:1px solid #c9d9d0;border-radius:14px;padding:24px;margin:20px 0}}
    .summary{{font-size:22px;font-weight:700}}.meta,small,figcaption{{color:#435754}}nav{{display:flex;gap:.4em 1em;flex-wrap:wrap}}
    .photos{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}figure{{margin:0;text-align:center;padding:8px;border:1px solid #d4e0d8;border-radius:8px}}
    img{{display:block;width:100%;height:420px;object-fit:contain;background:#edf2ef}}figcaption{{font-size:.85rem}}
    dl{{display:grid;grid-template-columns:minmax(110px,1fr) 2fr}}dd{{margin-left:0;overflow-wrap:anywhere}}dd small{{display:block}}
    th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #d4e0d8}}.scroll{{overflow:auto}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650}}
    @media(max-width:640px){{.photos{{grid-template-columns:1fr}}img{{height:330px}}article{{padding:16px}}}}
    </style></head><body><main>
    <h1>VietReceipt · Kiểm thử 20 ảnh từ Git</h1>
    <div class="overview"><div class="summary">{summary['complete']}/{len(cases)} ảnh chạy hết luồng</div>
    <p>{h(overview)}</p><p>20 ảnh gồm R001 và 19 ảnh tổng hợp do nhóm cung cấp. {h(run_note)} Tỷ lệ phủ dưới đây dùng các ô chữ và dòng hàng có nhãn trên 19 ảnh tổng hợp. Một ô đạt nếu ít nhất 90% diện tích nằm trong vùng cùng loại.</p>
    <div class="summary">{h(coverage_text)}</div><p>{h(extraction)}</p>
    <p>{log_link}<a href="git-evaluation.json">Số liệu theo ảnh</a></p></div>
    <nav aria-label="Đi đến từng ảnh">{"".join(menu)}</nav>
    {"".join(cards)}</main></body></html>"""
    (run / "index.html").write_text(html, encoding="utf-8")
    print(f"Hoàn tất {summary['complete']}/{len(cases)} · {overview}")
    print(f"Phủ nhãn ≥ 90% · {coverage_text}")
    print(extraction)
    print(f"Báo cáo ảnh {run / 'index.html'}")


if __name__ == "__main__":
    main()

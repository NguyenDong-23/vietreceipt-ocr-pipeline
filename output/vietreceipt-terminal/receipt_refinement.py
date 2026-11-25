"""Use the original receipt pixels to reread table cells and footer rows."""
from dataclasses import replace
from decimal import Decimal
from difflib import SequenceMatcher
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from ai.kie.v2.evidence import flatten, same_row
from ai.kie.v2.normalization import folded
from header_second_pass import ROOT, padded_crop, read_vision

NUMBER = re.compile(r"\d+(?:[., ]\d+)*")
UNITS = {
    "bo": "Bộ", "cai": "Cái", "cay": "Cây", "chai": "Chai", "goi": "Gói",
    "hop": "Hộp", "kg": "kg", "lan": "Lần", "lon": "Lon", "ly": "Ly",
    "quyen": "Quyển", "ram": "Ram", "thung": "Thùng", "tui": "Túi", "vi": "Vỉ",
}

def read_receipt_blocks(image, document, metadata, ids):
    vision = ROOT / "header_vision"
    if not ids or sys.platform != "darwin" or not os.access(vision, os.X_OK):
        return {}
    from checkpoint_ocr import read_page_image

    page = document["pages"][0]["evidence"]
    size = page["image"]
    original = read_page_image(image, 0, (size["width_px"], size["height_px"]))
    angle = metadata["pages"][0]["orientation_clockwise"]
    with TemporaryDirectory() as temporary:
        crops = {}
        for block in page["blocks"]:
            block_id = block["block_id"]
            if block_id not in ids:
                continue
            try:
                crop = padded_crop(original, block["polygon"], angle)
            except ValueError:
                continue
            path = Path(temporary) / f"{block_id}.png"
            crop.save(path)
            crops[block_id] = path
        return read_vision(vision, crops)


def trusted_reads(cell, reads, blocks):
    ids = cell["source_block_ids"]
    if not ids or any(block_id not in reads or reads[block_id]["confidence"] < .75 for block_id in ids):
        return []
    return [replace(blocks[block_id], text=reads[block_id]["text"]) for block_id in ids]


def split_printed_unit(blocks, aspect_ratio):
    if len(blocks) < 2:
        return None
    ordered = sorted(blocks, key=lambda block: block.left)
    preceding, tail = ordered[-2:]
    unit = UNITS.get(folded(tail.text))
    gap = (tail.left - preceding.right) * aspect_ratio
    if not unit or not same_row(preceding, tail) or gap < max(preceding.height, tail.height):
        return None
    return tail, unit


def refine_table(table, document, reads, retail):
    from ai.kie.v2.extraction import row_metadata

    blocks = {block.block_id: block for block in flatten(document)}
    image_size = document["pages"][0]["evidence"]["image"]
    aspect_ratio = image_size["width_px"] / image_size["height_px"]
    changes = []
    for row in table["line_items"]:
        initial_count = len(changes)
        description = row["description"]
        if recovered := trusted_reads(description, reads, blocks):
            separated = None
            if row["unit"]["value_status"] == "NOT_PRESENT":
                separated = split_printed_unit(recovered, aspect_ratio)
                if separated:
                    recovered = [block for block in recovered if block.block_id != separated[0].block_id]
            value = retail._clean_description(recovered)
            previous = description.get("normalized_value") or description.get("predicted_value") or ""
            if (value and folded(value) and value.casefold() != previous.casefold()
                    and SequenceMatcher(None, folded(value), folded(previous)).ratio() >= .65):
                cell = retail._cell("description", value, recovered, table["currency_context"])
                if cell["value_status"] == "PRESENT":
                    retail.add_reason(cell, "AMBIGUOUS_FORMAT")
                    cell["raw_text"] = "\n".join(blocks[i].text for i in cell["source_block_ids"])
                    row["description"] = cell
                    changes.append({"line_id": row["line_id"], "field": "description", "before": description,
                                    "block_ids": cell["source_block_ids"]})
                    if separated:
                        prior = row["unit"]
                        unit_block, unit = separated
                        row["unit"] = retail._cell("unit", unit, [unit_block], table["currency_context"])
                        row["unit"]["raw_text"] = blocks[unit_block.block_id].text
                        changes.append({"line_id": row["line_id"], "field": "unit", "before": prior,
                                        "block_ids": [unit_block.block_id]})
        amount = row["amount"]
        if amount["value_status"] == "AMBIGUOUS" and (recovered := trusted_reads(amount, reads, blocks)):
            unit_price = row["unit_price"].get("normalized_value")
            quantity = row["quantity"].get("normalized_value")
            if len(recovered) == 1 and unit_price is not None and quantity is not None:
                value = recovered[0].text.strip()
                printed = retail._money_value(value)
                expected = Decimal(str(quantity)) * Decimal(str(unit_price))
                if printed is not None and printed == expected:
                    row["amount"] = retail._cell("amount", value, recovered, table["currency_context"])
                    row["amount"]["raw_text"] = amount["raw_text"]
                    changes.append({"line_id": row["line_id"], "field": "amount", "before": amount,
                                    "block_ids": amount["source_block_ids"],
                                    "verified_with": "quantity_times_unit_price"})
        if len(changes) != initial_count:
            row_metadata(row)
    if changes:
        table["refinement"] = {
            "engine": "apple-vision", "reader_ocr_run_id": document["ocr_run_id"], "changes": changes,
            "reads": {i: reads[i] for item in changes for i in item["block_ids"]},
        }
    retail.validate_retail_table(table, document)


def digits(text):
    return re.sub(r"\D", "", text)


def footer_rows(rows, document, reads, assigned_ids):
    blocks = {block["block_id"]: block["text"] for block in document["pages"][0]["evidence"]["blocks"]}
    result = []
    uses_vision = False
    for row in rows:
        ids = row["ocr_ids"]
        if row["label"] != "footer" or not set(ids) <= assigned_ids:
            continue
        segments = []
        for block_id in ids:
            original = blocks[block_id]
            candidate = reads.get(block_id, {})
            text = candidate.get("text", "")
            similarity = SequenceMatcher(None, folded(text), folded(original)).ratio()
            valid_format = not NUMBER.fullmatch(original.strip()) or NUMBER.fullmatch(text.strip())
            if text and candidate["confidence"] >= .75 and digits(text) == digits(original) and similarity >= .45 and valid_format:
                uses_vision = True
                segments.append({"block_id": block_id, "text": text, "source": "apple-vision"})
            else:
                segments.append({"block_id": block_id, "text": original, "source": "reader"})
        result.append({"text": " ".join(part["text"] for part in segments), "segments": segments})
    return result if uses_vision else []


def refine_receipt(image, document, metadata, links, prediction, table, retail):
    assigned = {"table": set(), "footer": set()}
    for block in links["blocks"]:
        if block["status"] == "ASSIGNED":
            for hit in block["matches"]:
                if hit["label"] in assigned:
                    assigned[hit["label"]].add(block["block_id"])
    cells = [row["description"] for row in table["line_items"]]
    cells.extend(row["amount"] for row in table["line_items"] if row["amount"]["value_status"] == "AMBIGUOUS")
    table_ids = {i for cell in cells for i in cell["source_block_ids"]} & assigned["table"]
    try:
        reads = read_receipt_blocks(image, document, metadata, table_ids | assigned["footer"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"warning": f"apple-vision: {type(exc).__name__}"}
    if not reads:
        return {}
    refine_table(table, document, reads, retail)
    rows = footer_rows(prediction["rows"], document, reads, assigned["footer"])
    return {"rows": rows, "reader_ocr_run_id": document["ocr_run_id"]} if rows else {}

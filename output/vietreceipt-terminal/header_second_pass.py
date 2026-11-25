"""Reread Header name/address fields and conflicted symbols from source pixels."""
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image, ImageOps

from header_recovery import match_label, qualified_value

ROOT = Path(__file__).resolve().parent
TEXT_FIELDS = ("seller_name", "seller_address", "buyer_name")
VISION = "apple-vision"
TESSERACT = "tesseract-vie"


def padded_crop(image, polygon, angle):
    from checkpoint_ocr import crop_polygon

    expanded = []
    for index, point in enumerate(polygon):
        side, vertical = polygon[index ^ 1], polygon[3 - index]
        expanded.append({
            axis: min(1, max(0, point[axis] + .025 * (point[axis] - side[axis])
                             + .22 * (point[axis] - vertical[axis])))
            for axis in ("x", "y")
        })
    crop = crop_polygon(image, expanded, angle)
    factor = max(1, 64 / crop.height)
    gray = ImageOps.autocontrast(crop.convert("L"))
    return ImageOps.expand(
        gray.resize((round(gray.width * factor), round(gray.height * factor)), Image.Resampling.LANCZOS),
        12, fill=255,
    )


def read_tesseract(executable, crop):
    result = subprocess.run(
        [executable, str(crop), "stdout", "--tessdata-dir", str(ROOT / "tessdata"),
         "-l", "vie", "--oem", "1", "--psm", "7", "-c", "tessedit_create_tsv=1"],
        capture_output=True, text=True, check=True, timeout=10,
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    words = [line.split("\t", 11)[10:] for line in result.stdout.splitlines() if line.startswith("5\t")]
    if not words:
        return {}
    return {
        "text": " ".join(word for _, word in words),
        "confidence": sum(float(score) for score, _ in words) / (100 * len(words)),
    }


def read_vision(executable, crops):
    inputs = [{"id": block_id, "path": str(path)} for block_id, path in sorted(crops.items())]
    if not inputs:
        return {}
    reply = subprocess.run([str(executable)], input=json.dumps(inputs), capture_output=True,
                           check=True, text=True, timeout=90)
    return json.loads(reply.stdout)


def refine_header(image, header, document, metadata, region_links):
    from ai.kie.v2.normalization import folded
    from checkpoint_ocr import read_page_image

    fields = header["fields"]
    assigned = {item["block_id"] for item in region_links["blocks"] if item["status"] == "ASSIGNED"
                and any(hit["label"] == "header" for hit in item["matches"])}
    if len(document["pages"]) != 1:
        return
    evidence = document["pages"][0]["evidence"]
    blocks = {block["block_id"]: block for block in evidence["blocks"] if block["block_id"] in assigned}
    old_sources = header.setdefault("field_sources", {})
    targets = {}
    vision = ROOT / "header_vision"
    if sys.platform == "darwin" and os.access(vision, os.X_OK):
        for key in TEXT_FIELDS:
            cell = fields[key]
            if cell["value_status"] == "PRESENT" and len(cell["source_block_ids"]) == 1:
                block_id = cell["source_block_ids"][0]
                if block_id in blocks:
                    targets[key] = block_id
    alternatives = header.get("review_alternatives", {}).get("invoice_symbol", {})
    tesseract = shutil.which("tesseract")
    if tesseract and (ROOT / "tessdata/vie.traineddata").is_file() and alternatives:
        ids = {i for candidate in alternatives.values() for i in candidate["source_block_ids"]}
        if len(ids) == 1 and ids <= blocks.keys():
            targets["invoice_symbol"] = next(iter(ids))
    if not targets:
        return

    supplements = {}
    warnings = []
    size = evidence["image"]
    page = read_page_image(image, 0, (size["width_px"], size["height_px"]))
    angle = metadata["pages"][0]["orientation_clockwise"]
    readings = {}
    with TemporaryDirectory() as temp:
        crops = {}
        for block_id in set(targets.values()):
            try:
                crop = padded_crop(page, blocks[block_id]["polygon"], angle)
            except ValueError:
                continue
            crops[block_id] = Path(temp) / f"{block_id}.png"
            crop.save(crops[block_id])
        inputs = {block_id: crops[block_id]
                  for block_id in {v for k, v in targets.items() if k in TEXT_FIELDS} & crops.keys()}
        if inputs:
            try:
                readings[VISION] = read_vision(vision, inputs)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                warnings.append(f"{VISION}: {type(exc).__name__}")
        if (block_id := targets.get("invoice_symbol")) in crops:
            try:
                if reading := read_tesseract(tesseract, crops[block_id]):
                    readings[TESSERACT] = {block_id: reading}
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                warnings.append(f"{TESSERACT}: {type(exc).__name__}")
    for engine, reads in readings.items():
        supplements[engine] = {"ocr_run_id": str(uuid4()), "parent_ocr_run_id": document["ocr_run_id"],
                               "reads": reads}
    for key, block_id in targets.items():
        engine = TESSERACT if key == "invoice_symbol" else VISION
        record = readings.get(engine, {}).get(block_id)
        if not record:
            continue
        parsed = match_label(record["text"], folded)
        if not parsed or parsed[0] != key or not (value := qualified_value(key, parsed[1])):
            continue
        cell = fields[key]
        if key == "invoice_symbol":
            if record["confidence"] < .5 or value.upper() not in {a["value"].upper() for a in alternatives.values()}:
                continue
            value = value.upper()
        else:
            previous = cell["normalized_value"]
            reader_label = match_label(blocks[block_id]["text"], folded)
            comparison = [previous] + ([reader_label[1]] if reader_label and reader_label[0] == key else [])
            compatibility = max(SequenceMatcher(None, folded(value), folded(old)).ratio() for old in comparison)
            if record["confidence"] < .75 or compatibility < .82 or value.casefold() == previous.casefold():
                continue
        header.setdefault("pre_refinement_fields", {})[key] = {
            "cell": dict(cell), "source": old_sources.get(key, {"engine": "vietocr-checkpoint"}),
        }
        fields[key] = {
            **cell, "raw_text": record["text"], "predicted_value": value, "normalized_value": value,
            "normalization": {"rule": "labelled_second_pass", "version": "1.0"},
            "value_status": "PRESENT", "confidence": None, "heuristic_score": 0.0,
            "score_version": "header-second-pass-1.0", "machine_needs_review": True,
            "review_reasons": ["OCR_RECOGNIZER_DISAGREEMENT"], "source_block_ids": [block_id],
        }
        old_sources[key] = {"engine": engine, "ocr_run_id": supplements[engine]["ocr_run_id"]}
    if supplements:
        header["supplemental_ocr"] = supplements
    if warnings:
        header["supplemental_ocr_warnings"] = warnings

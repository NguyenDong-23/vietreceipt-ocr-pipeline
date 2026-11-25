"""Exploratory 20-image retail table replay; OCR snapshots are never rerun."""

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from ai.kie.retail_table import extract_retail_table

FIELDS = ("description", "unit", "quantity", "unit_price", "amount")


def normalize_name(value):
    value = unicodedata.normalize("NFC", str(value)).casefold()
    value = re.sub(r"^\s*\d{8,14}\s+", "", value)
    value = re.sub(r"\s+(?:nt|no)\s*$", "", value)
    return " ".join(value.split())


def normalize_value(field, value):
    if value is None:
        return None
    if field == "description":
        return normalize_name(value)
    if field == "unit":
        return " ".join(str(value).casefold().split())
    if field in ("quantity", "unit_price", "amount"):
        try:
            raw = str(value).strip()
            if "." in raw and "," not in raw and field != "quantity":
                raw = raw.replace(".", "")
            elif "," in raw and "." not in raw:
                raw = raw.replace(",", "" if field != "quantity" else ".")
            return float(raw)
        except ValueError:
            return None
    return value


def prediction_cell(cell, field):
    status = cell["value_status"]
    value = cell["normalized_value"]
    if field == "description" and status == "PRESENT":
        value = cell.get("predicted_value") or value
    return status, normalize_value(field, value)


def match_pairs(gold_rows, predicted_rows):
    """Original 20-image probe policy: one-to-one name similarity >= 0.55."""
    def key(value):
        value = unicodedata.normalize("NFD", normalize_name(value or ""))
        value = "".join(c for c in value if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-z0-9]+", "", value)

    edges = []
    for i, gold in enumerate(gold_rows):
        expected = key(gold["description"]["value"])
        ranked = []
        for j, predicted in enumerate(predicted_rows):
            cell = predicted["description"]
            observed = key(cell.get("predicted_value") or cell.get("normalized_value"))
            if not expected or not observed:
                continue
            similarity = SequenceMatcher(None, expected, observed).ratio()
            if similarity >= 0.55:
                ranked.append((similarity, -abs(i - j), j))
        edges.append([j for _, _, j in sorted(ranked, reverse=True)])
    owners = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(gold_rows)):
        assign(i, set())
    return sorted((i, j) for j, i in owners.items())


def evaluate(labels_dir: Path, images_dir: Path, ocr_dir: Path, *, currency: str) -> dict:
    labels = sorted(labels_dir.glob("*.table.json"))
    if not labels:
        raise ValueError("No .table.json labels found")
    totals = {"gold_rows": 0, "predicted_rows": 0, "matched_rows": 0,
              "cells_scored": 0, "cells_correct": 0, "complete_rows": 0}
    records = []
    for path in labels:
        gold = json.loads(path.read_text(encoding="utf-8"))
        stem = path.name.removesuffix(".table.json")
        image = images_dir / f"{stem}.jpg"
        if hashlib.sha256(image.read_bytes()).hexdigest() != gold["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {stem}")
        snapshot_id = gold.get("ocr_snapshot_id")
        ocr = ocr_dir / f"{snapshot_id}.json" if snapshot_id else ocr_dir / "20_anh" / f"{stem}.json"
        if not ocr.is_file():
            raise FileNotFoundError(ocr)
        if not snapshot_id:
            digest_path = ocr.with_suffix(".source.sha256")
            if digest_path.read_text(encoding="ascii").strip() != gold["image_sha256"]:
                raise ValueError(f"OCR source hash mismatch: {stem}")
        evidence = json.loads(ocr.read_text(encoding="utf-8"))
        predicted = extract_retail_table(evidence, currency=currency)["line_items"]
        pairs = match_pairs(gold["rows"], predicted)
        record = {"image": image.name, "gold_rows": len(gold["rows"]),
                  "predicted_rows": len(predicted), "matched_rows": len(pairs),
                  "matches": [], "unmatched_predictions": []}
        matched_predictions = {j for _, j in pairs}
        for j, row in enumerate(predicted):
            if j not in matched_predictions:
                record["unmatched_predictions"].append({
                    "index": j + 1,
                    "description": row["description"]["predicted_value"],
                    "source_block_ids": row["source_block_ids"],
                })
        for i, j in pairs:
            cells = {}
            for field in FIELDS:
                target = gold["rows"][i][field]
                if target["status"] in ("UNKNOWN", "UNREADABLE"):
                    continue
                status, value = prediction_cell(predicted[j][field], field)
                correct = status == target["status"] and (
                    status != "PRESENT" or value == normalize_value(field, target["value"])
                )
                totals["cells_scored"] += 1
                totals["cells_correct"] += correct
                cells[field] = {"correct": correct, "gold": target["value"],
                                "predicted": predicted[j][field]["normalized_value"],
                                "status": status}
            complete = len(cells) == len(FIELDS) and all(c["correct"] for c in cells.values())
            totals["complete_rows"] += complete
            record["matches"].append({"gold_row_id": gold["rows"][i]["row_id"],
                                      "predicted_index": j + 1, "cells": cells,
                                      "complete": complete})
        totals["gold_rows"] += len(gold["rows"])
        totals["predicted_rows"] += len(predicted)
        totals["matched_rows"] += len(pairs)
        records.append(record)
    p = totals["matched_rows"] / totals["predicted_rows"] if totals["predicted_rows"] else 0
    r = totals["matched_rows"] / totals["gold_rows"] if totals["gold_rows"] else 0
    totals.update(row_precision=p, row_recall=r,
                  row_f1=2 * p * r / (p + r) if p + r else 0,
                  cell_accuracy_on_matched_rows=(
                      totals["cells_correct"] / totals["cells_scored"]
                      if totals["cells_scored"] else None),
                  complete_row_recall=(totals["complete_rows"] / totals["gold_rows"]
                                       if totals["gold_rows"] else None))
    return {"extractor": "retail-table-0.1.0",
            "metric_policy": "same-name-similarity-0.55-as-existing-20-image-probe",
            "limitations": ["Development replay on the same 20 first-pass labels used to inspect the failure.",
                            "Name-based pairing is exploratory; independent review and held-out images are required."],
            "metrics": totals, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--currency", required=True, choices=["VND"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    result = evaluate(args.labels, args.images, args.ocr, currency=args.currency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

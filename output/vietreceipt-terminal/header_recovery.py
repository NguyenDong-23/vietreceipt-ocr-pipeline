"""Recover structured Header fields from explicit OCR labels and retained evidence."""
from datetime import datetime
import re
from unicodedata import normalize

# Forms here identify printed labels. The captured text is always from OCR.
LABELS = {
    "invoice_template_number": r"mau\s*s[o0b]",
    "invoice_symbol": r"ky\s*hi(?:eu|u|uu)",
    "invoice_number": r"(?:so|s6|s)\s*(?:hoa|haa|ha)\s*don|so\s*hd",
    "invoice_date": r"ngay\s*(?:lap|ap|lan)(?:\s+hoa\s+don)?|ngay\s+hoa\s+don",
    "seller_name": r"don\s*vi\s*ban\s*hang|(?:ten\s+)?nguoi\s+ban",
    "seller_address": r"d[ij]a\s*chi\s+ng(?:uoi|ui|uri)\s+ban",
    "seller_tax_id": r"(?:mst|ma\s*so\s*thue)\s+ng(?:uoi|ui|uri)\s+ban",
    "buyer_name": r"(?:ho\s+ten\s+)?ng(?:uoi|ui|uri)\s+mua",
    "buyer_tax_id": r"(?:mst|ma\s*so\s*thue)\s+ng(?:uoi|ui|uri)\s+mua",
}
PATTERNS = {
    "invoice_template_number": r"\d{1,3}",
    "invoice_symbol": r"(?=[A-Z0-9/.-]*[A-Z])(?=[A-Z0-9/.-]*\d)[A-Z0-9][A-Z0-9/.-]{2,15}",
    "invoice_number": r"(?=[A-Z0-9/.-]*\d)[A-Z0-9][A-Z0-9/.-]{0,29}",
    "invoice_date": r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}",
    "seller_tax_id": r"\d{10}(?:-?\d{3})?",
    "buyer_tax_id": r"\d{10}(?:-?\d{3})?",
}
PATTERN = re.compile("|".join(
    rf"(?P<{key}>{label})(?=\s|[:：#]|\d|$|[a-z]\d)" for key, label in LABELS.items()
))
TRANSACTION = re.compile(r"^so\s+(?:gd|giao\s+dich)\s*:\s*([a-z0-9-]+)"
                         r"\s+ngay\s*:\s*(\d{1,2}/\d{1,2}/\d{4})(?:[-\s]+\d{2}:\d{2})?\s*$")


def qualified_value(field, value):
    if field not in PATTERNS:
        return value if len(value.split()) >= 2 and any(c.isalpha() for c in value) else None
    if not re.fullmatch(PATTERNS[field], value, re.I):
        return None
    if field == "invoice_date" and "/" in value:
        day, month, year = value.split("/")
        return f"ngày {day} tháng {month} năm {year}"
    return value


def match_label(text, folded):
    value = " ".join(normalize("NFC", text).split())
    match = PATTERN.match(folded(value))
    if match:
        return match.lastgroup, value[match.end():].lstrip(" :：#")


def recover_header(header, evidence, reader, region_links):
    from ai.kie.v2.cells import Candidate, add_reason, select
    from ai.kie.v2.evidence import flatten, same_row
    from ai.kie.v2.normalization import folded

    region_ids = {"header": set(), "footer": set()}
    for link in region_links["blocks"]:
        if link["status"] == "ASSIGNED":
            for match in link["matches"]:
                if match["label"] in region_ids:
                    region_ids[match["label"]].add(link["block_id"])
    fields = header["fields"]
    field_sources = {}
    symbol_reads = {}
    checkpoint_blocks = flatten(evidence)
    for engine, document, blocks in (
        ("vietocr-checkpoint", evidence, checkpoint_blocks), ("reader", reader, flatten(reader))
    ):
        blocks = [block for block in blocks if block.block_id in region_ids["header"]]
        candidates = {key: [] for key in LABELS}
        visible_labels = {}
        for block in blocks:
            label = match_label(block.text, folded)
            if label is None:
                continue
            key, value = label
            source_blocks = (block,)
            visible_labels.setdefault(key, block)
            if not value:
                adjacent = [b for b in blocks if b.left >= block.right and same_row(b, block)]
                if adjacent:
                    neighbor = min(adjacent, key=lambda b: b.left)
                    if not match_label(neighbor.text, folded) and neighbor.left - block.right <= 3 * block.height:
                        value, source_blocks = neighbor.text.strip(), (block, neighbor)
            # Native KIE can emit an unparsed prefix as a date; candidates here
            # require an entire printed date or ID after the label.
            if qualified := qualified_value(key, value):
                candidates[key].append(Candidate(qualified, source_blocks))
        for key, options in candidates.items():
            current = fields[key]
            candidate = select(key, options)
            if key == "invoice_symbol" and candidate["value_status"] == "PRESENT":
                symbol_reads[engine] = {
                    "value": candidate["normalized_value"], "raw_text": candidate["raw_text"],
                    "source_block_ids": candidate["source_block_ids"], "ocr_run_id": document["ocr_run_id"],
                }
            if current["value_status"] == "PRESENT":
                continue
            if candidate["value_status"] != "PRESENT":
                if options and current["value_status"] in {"UNKNOWN", "UNREADABLE"}:
                    fields[key] = candidate
                    field_sources[key] = {"engine": engine, "ocr_run_id": document["ocr_run_id"]}
                elif current["value_status"] == "UNKNOWN" and key in visible_labels:
                    block = visible_labels[key]
                    unreadable = select(key, [Candidate("", (block,))])
                    fields[key] = unreadable
                    field_sources[key] = {"engine": engine, "ocr_run_id": document["ocr_run_id"]}
                continue
            # Leave existing separate candidates visible when the same OCR
            # yielded conflicting labelled values.
            if current["value_status"] == "AMBIGUOUS" and "MULTIPLE_CANDIDATES" in current["review_reasons"]:
                if key not in {"seller_tax_id", "buyer_tax_id"}:
                    continue
            if engine == "reader":
                add_reason(candidate, "OCR_RECOGNIZER_DISAGREEMENT")
            fields[key] = candidate
            field_sources[key] = {"engine": engine, "ocr_run_id": document["ocr_run_id"]}
    # A generic MST candidate can be assigned to both roles by the base KIE.
    # Once an explicit label resolves it, remove the ungrounded other role.
    for role, other in (("seller_tax_id", "buyer_tax_id"), ("buyer_tax_id", "seller_tax_id")):
        cell, uncertain = fields[role], fields[other]
        uncertain_ids = set(uncertain["source_block_ids"])
        if (role in field_sources and cell["value_status"] == "PRESENT"
                and uncertain["value_status"] == "AMBIGUOUS"
                and "SOURCE_ROLE_UNCLEAR" in uncertain["review_reasons"]
                and uncertain_ids and uncertain_ids <= set(cell["source_block_ids"])):
            fields[other] = select(other, [])
            field_sources.pop(other, None)
    symbol = fields["invoice_symbol"]
    if len(symbol_reads) == 2 and len({entry["value"] for entry in symbol_reads.values()}) == 2:
        if symbol["value_status"] == "PRESENT":
            symbol.update(value_status="AMBIGUOUS", normalized_value=None, normalization=None)
            add_reason(symbol, "OCR_RECOGNIZER_DISAGREEMENT")
        header["review_alternatives"] = {"invoice_symbol": symbol_reads}
    header["field_sources"] = field_sources

    for block in checkpoint_blocks:
        if block.block_id not in region_ids["footer"]:
            continue
        match = TRANSACTION.fullmatch(folded(block.text))
        if match is None:
            continue
        try:
            value = datetime.strptime(match[2], "%d/%m/%Y").date().isoformat()
        except ValueError:
            continue
        header["receipt_metadata"] = {
            "transaction_id": {"value": match[1].upper(), "source_block_ids": [block.block_id]},
            "transaction_date": {"value": value, "source_block_ids": [block.block_id]},
        }
        break

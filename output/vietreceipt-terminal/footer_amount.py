"""Read a payable VND amount from explicit payment rows in Footer OCR."""
import re
import unicodedata

PAYABLE = re.compile(
    r"\s*(?:(?:tong|tng)\s*thanh\s*toan|(?:tong|t&ng)\s*tien|phai\s*thanh\s*toan)"
    r"\s*[:.]?\s*(?:vnd\s*)?(\d{1,3}(?:[., ]\d{3})+|\d+)\s*(?:vnd|₫|d)?\s*"
)


def read_payable(text):
    matches = {}
    for line in text.splitlines():
        folded = "".join(c for c in unicodedata.normalize("NFD", line.casefold())
                         if not unicodedata.combining(c))
        if match := PAYABLE.fullmatch(folded):
            value = int(re.sub(r"\D", "", match[1]))
            matches.setdefault(value, []).append(line)
    if len(matches) == 1:
        amount, source_rows = next(iter(matches.items()))
        return {"status": "PRESENT", "amount_vnd": amount, "source_rows": source_rows}
    return {"status": "AMBIGUOUS" if matches else "UNKNOWN", "amount_vnd": None,
            "source_rows": [line for rows in matches.values() for line in rows]}


def resolve_payable(paddle_text, layout_rows):
    primary = read_payable(paddle_text)
    if primary["status"] == "UNKNOWN":
        fallback = read_payable("\n".join(row["text"] for row in layout_rows if row["label"] == "footer"))
        if fallback["status"] == "PRESENT":
            return {**fallback, "source": "READER_PP_FOOTER"}
    return {**primary, "source": "PADDLE_FOOTER"}


def apply_footer_display(footer, supplement):
    """Show grouped receipt text when its payment is consistent with the result."""
    if warning := supplement.get("warning"):
        footer["refinement_warning"] = warning
    rows = supplement.get("rows", [])
    if not rows:
        return
    reread = read_payable("\n".join(row["text"] for row in rows))
    current = footer["payment"]
    if reread["status"] != "PRESENT" or current["status"] == "AMBIGUOUS":
        return
    if current["status"] == "PRESENT" and current["amount_vnd"] != reread["amount_vnd"]:
        return
    footer["display"] = supplement
    if current["status"] == "UNKNOWN":
        footer["payment"] = {**reread, "source": "VISION_READER_FOOTER"}

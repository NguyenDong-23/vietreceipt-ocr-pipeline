"""Opt-in geometry experiment; no OCR repair or numeric value inference."""

import re

from .extraction import LINE_LABELS, group_columns, visual_rows
from .normalization import folded

# Exact observed header alias from TV2 train VRSYN2-00001; no fuzzy OCR repair.
EXPERIMENT_LINE_LABELS = {
    **LINE_LABELS,
    "description": (
        r"ten hang(?: hoa)?\s*/\s*dich v(?:u)?|ten\s*(?:mon|sp)(?:\s*sl)?|"
        + LINE_LABELS["description"]
    ),
    "quantity": r"sl\s*dvt|si\.*|" + LINE_LABELS["quantity"],
    "unit_price": r"d\.?\s*gia|dn gia|" + LINE_LABELS["unit_price"],
    "amount": r"t\.?\s*t(?:ien|i[&e]n)|th[ao]nh tien|" + LINE_LABELS["amount"],
}
EXPERIMENT_LINE_STOP = (
    r"^(?:thue suat(?: gtgt)?\s*[:：]|"
    r"tong (?:tien(?:\s+phai\s+t\.?\s*toan)?|cong|so)\b|"
    r"tong\s+tl[eê]n\b|thanh\s*ti[ea]n\b|"
    r"\+?thanh\s*toan\s*tien\s*mat\b|"
    r"tien\s*(?:khach\s*(?:tra|dua)|mat|nhan|thua|tra\s*lai)\b|"
    r"khach thanh toan\b|chiet khau\b|ph[uy] thu\b|giam\s*gia\b|"
    r"(?:s|so) tham\b|chi xuat hoa don\b|tax invoice\b|"
    r"cam on\b|xin cam on\b)"
)

_PRICE = re.compile(r"[0-9]+(?:[.,][0-9]+)*(?:\s*(?:VND|[đ₫]))?", re.I)
_BARCODE = re.compile(r"[0-9]{8,14}")
_SUMMARY = re.compile(r"^(?:tong tien|tong cong|tong thanh toan|grand total|total)\b")
_RECEIPT_METADATA = re.compile(
    r"^(?:#(?:sale\s*type|pg)|"
    r"(?:thu\s*ngan|th[oi]+\s*gian|s[o6]\s*(?:hd|hoa don))(?=\W|\d|$)|"
    r"(?:invoice|phieu tinh tien)\b)"
)


def interval_columns(row, columns):
    """Prefer unique header containment for short numeric cells.

    Narrow right-aligned numbers can be nearer the next header's left edge than
    their own header's center. Only a block fully contained in exactly one header
    span, belonging to a numeric column, overrides the baseline. Overlap, gutters,
    merged cells and non-numeric text fall back without inventing boundaries.
    """
    groups = {field: [] for field in columns}
    description = columns.get("description")
    for block in row:
        if (
            description
            and abs(block.left - description.left) <= 2 * description.height
            and re.search(r"[^\W\d_]", block.text)
            and not _PRICE.fullmatch(block.text.strip())
        ):
            groups["description"].append(block)
            continue
        contained = [
            field
            for field, header in columns.items()
            if header.left <= block.left and block.right <= header.right
        ]
        if (
            len(contained) == 1
            and contained[0] in {"quantity", "unit_price", "amount"}
            and re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", block.text.strip())
        ):
            groups[contained[0]].append(block)
        else:
            for field, blocks in group_columns([block], columns).items():
                groups[field].extend(blocks)
    return groups


def merge_receipt_item_rows(rows, columns):
    """Join a description above a barcode and its aligned numeric cells.

    A product code has no output column. Retain it in OCR evidence and skip it
    when populating the description cell.
    """
    merged = []
    for row in rows:
        groups = interval_columns(row, columns)
        codes = groups.get("description", [])
        numeric = sum(
            len(groups.get(field, [])) == 1
            and _PRICE.fullmatch(groups[field][0].text.strip()) is not None
            for field in ("quantity", "unit_price", "amount")
        )
        barcode = next((b for b in codes if _BARCODE.fullmatch(b.text.strip())), None)
        if merged and (barcode or not codes) and numeric >= 2:
            previous = merged[-1]
            preceding = interval_columns(previous, columns)
            gap = min(b.top for b in row) - max(b.bottom for b in previous)
            if (
                preceding.get("description")
                and not any(
                    items for field, items in preceding.items() if field != "description"
                )
                and previous[0].page == row[0].page
                and -min(b.height for b in previous) <= gap <= 2 * max(b.height for b in previous)
                and any(re.search(r"[^\W\d_]", b.text) for b in previous)
            ):
                merged[-1] = [
                    *previous,
                    *(
                        b
                        for b in row
                        if b is not barcode
                        and (
                            b not in codes
                            or abs(b.left - columns["description"].left)
                            <= columns["description"].height
                        )
                    ),
                ]
                continue
        merged.append(row)
    return merged


def recover_merged_receipt_columns(blocks, columns, rows):
    """Recover the geometry of explicitly printed but merged column labels."""
    quantity = columns.get("quantity")
    if (
        not quantity
        or not re.fullmatch(r"sl\s*dvt", folded(quantity.text))
        or "unit_price" in columns
        or "amount" in columns
        or not any(
            block.page == quantity.page
            and abs(block.top - quantity.top) <= quantity.height
            and block.left > quantity.left
            and re.fullmatch(r"don gia\s+thanh tien", folded(block.text))
            for block in blocks
        )
    ):
        return columns

    def numeric_tail(row):
        return (
            len(row) == 4
            and row[0].left >= quantity.left - quantity.height
            and _PRICE.fullmatch(row[0].text.strip())
            and re.fullmatch(r"[^\W\d_]+", row[1].text.strip())
            and all(_PRICE.fullmatch(block.text.strip()) for block in row[2:])
        )

    tails = [row for row in rows if numeric_tail(row)]
    if len(tails) < 2 or not all(
        abs(block.left - anchor.left) <= quantity.height
        for row in tails[1:] for block, anchor in zip(row, tails[0])
    ):
        return columns
    return {
        **columns,
        **dict(zip(("quantity", "unit", "unit_price", "amount"), tails[0])),
    }


def unheaded_receipt_regions(blocks, regions):
    """Recover a bounded run of item rows just before a printed total.

    A line with only a description and a value supports an amount, but not an
    independent unit price. Keep unavailable columns unknown.
    """
    headed_pages = {
        next(iter(columns.values())).page for kind, columns, _ in regions if kind == "line"
    }
    recovered = []
    for page in sorted({block.page for block in blocks} - headed_pages):
        rows = visual_rows([block for block in blocks if block.page == page])
        for index, row in enumerate(rows):
            totals = [block for block in row[1:] if _PRICE.fullmatch(block.text.strip())]
            if not (
                _SUMMARY.match(folded(row[0].text))
                and len(totals) == 1
                and any(
                    _RECEIPT_METADATA.match(folded(block.text))
                    for prior in rows[:index]
                    for block in prior
                )
            ):
                continue
            items = []
            lower = row
            for prior in reversed(rows[:index]):
                description, *numbers = prior
                if not (
                    1 <= len(numbers) <= 2
                    and all(_PRICE.fullmatch(block.text.strip()) for block in numbers)
                    and re.search(r"[^\W\d_]", description.text)
                    and description.right <= numbers[0].left
                    and abs(description.left - row[0].left) <= 0.08
                    and abs(numbers[-1].right - totals[0].right) <= 0.08
                    and not _RECEIPT_METADATA.match(folded(description.text))
                    and not re.match(EXPERIMENT_LINE_STOP, folded(description.text))
                    and min(block.top for block in lower) - max(block.bottom for block in prior)
                    <= 4 * max(block.height for block in prior)
                ):
                    break
                items.append(prior)
                lower = prior
            if items:
                columns = {"description": items[0][0], "amount": items[0][-1]}
                quantified = [item for item in items if len(item) == 3]
                if quantified and all(
                    re.fullmatch(r"[1-9][0-9]{0,2}", item[1].text.strip())
                    for item in quantified
                ):
                    columns["quantity"] = quantified[0][1]
                recovered.append(("line", columns, list(reversed(items))))
                break
    return recovered

"""Opt-in retail-receipt table extraction from immutable spatial OCR evidence.

Cells retain Invoice V2 evidence shape, but signed discount values require a
separate retail contract. This module never reads header/footer fields or gold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai.kie.v2.cells import Candidate, add_reason, select
from ai.kie.v2.config import LINE_FIELDS
from ai.kie.v2.evidence import Block, flatten
from ai.kie.v2.extraction import row_metadata, visual_rows
from ai.kie.v2.normalization import folded

VERSION = "retail-table-0.1.0"
_MONEY = re.compile(r"^-?(?:\d{1,3}(?:[.,]\d{3})+|\d{4,7})(?:\s*[dđ₫])?$", re.I)
_MONEY_SHAPE = re.compile(
    r"^-?(?:\d{1,3}(?:[., ]\d{1,3})+|\d{4,7})(?:\s*[dđ₫])?$", re.I
)
_QUANTITY = re.compile(r"^\d{1,3}(?:[.,]\d{1,3})?(?:\s*kg)?$", re.I)
_UNIT = re.compile(r"^(?:cai|chai|cha|goi|got|hop|vi|kg|ly|lon|thung|tui|bo)$", re.I)
_SKU_PREFIX = re.compile(r"^\s*\d{8,14}\s*")
_SKU_ONLY = re.compile(r"^\d{8,14}$")
_PRICE_LABEL = re.compile(r"^(?:gia\s*ban|giaban|gia\s*dang\s*ky|giadang|iaban|iadang)")
_TOTAL = re.compile(
    r"^(?:t[o&]?ng|[o&]?ng|ng|ig)(?:\s*(?:so|s[ol]|cong|tien|thanh|sl))|"
    r"^(?:cong\s*tien|s[o0]\s*tien|thanh\s*tien|tien\s*(?:khach|mat|tra|thua)|"
    r"giam\s*gia|phu\s*thu|cash|card|vnpay|cam\s*on|xin\s*cam)",
    re.I,
)
_HEADER = re.compile(
    r"(?:ten\s*(?:hang|mon)|mat\s*hang|wat\s*hang|tenmon|"
    r"don\s*gia|dongia|dn\s*gia|gia|thanh\s*tien|thanhtien|t\s*tien|"
    r"sl|dvt|d\s*gia)",
    re.I,
)


@dataclass
class _Item:
    description: list[Block] = field(default_factory=list)
    support: list[list[Block]] = field(default_factory=list)
    prelude: list[list[Block]] = field(default_factory=list)


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", folded(value))


def _money_value(value: str) -> int | None:
    value = value.strip()
    if not _MONEY.fullmatch(value):
        return None
    digits = re.sub(r"[dđ₫\s.,]", "", value, flags=re.I)
    return int(digits)


def _is_total(row: list[Block]) -> bool:
    value = folded(" ".join(block.text for block in row)).strip()
    key = _key(value)
    return bool(_TOTAL.match(value) or key.startswith((
        "tong", "ongso", "ongcong",
        "ongtien", "ngso", "ngcong", "ngtien", "igso", "igcong",
        "tienkhachtra", "tientralai", "congtienhang", "thanhtoan",
        "chietkhau", "khachphaitra", "tienkhachdua", "tralai",
    )))


def _is_header(row: list[Block]) -> bool:
    text = folded(" ".join(block.text for block in row))
    matches = _HEADER.findall(text)
    return len(matches) >= 2 and not any(_money_value(b.text) is not None for b in row)


def _is_registered(row: list[Block]) -> bool:
    return any(_key(b.text).startswith(("giadangky", "iadangky")) for b in row)


def _description_blocks(row: list[Block]) -> list[Block]:
    result = []
    for block in row:
        raw = block.text.strip()
        value = _SKU_PREFIX.sub("", raw).strip()
        key = _key(value)
        if not value or _SKU_ONLY.fullmatch(raw) or _money_value(raw) is not None:
            continue
        if _QUANTITY.fullmatch(value) or _UNIT.fullmatch(value):
            continue
        if _PRICE_LABEL.match(folded(value)) or key.startswith(("giadangky", "giaban")):
            continue
        if not re.search(r"[a-zA-ZÀ-ỹ]", value):
            continue
        result.append(block)
    return result


def _clean_description(blocks: list[Block]) -> str:
    value = " ".join(_SKU_PREFIX.sub("", b.text).strip() for b in blocks)
    value = re.sub(r"^\s*\d{1,3}\s+(?=[A-Za-zÀ-ỹ])", "", value)
    value = re.sub(r"\s+(?:NT|N[O0])\s*$", "", value, flags=re.I)
    return " ".join(value.split())


def _cell(field: str, value: str | None, blocks: list[Block], currency: str):
    if not blocks or value is None:
        return select(field, [], currency=currency, absent=True)
    ordered = sorted(dict.fromkeys(blocks), key=lambda b: (b.page, b.reading_order))
    cell = select(field, [Candidate(value, tuple(ordered))], currency=currency)
    if field in {"amount", "unit_price"}:
        parsed = _money_value(value)
        if parsed is not None and currency == "VND":
            cell.update(
                value_status="PRESENT",
                normalized_value=parsed,
                normalization={"rule": "printed_vnd_integer", "version": VERSION},
            )
    elif field == "unit":
        cell.update(
            value_status="PRESENT",
            normalized_value=value,
            normalization={"rule": "printed_unit", "version": VERSION},
        )
    return cell


def _build_item(item: _Item, index: int, currency: str):
    if not item.description:
        return None
    support = item.prelude + item.support
    eligible = [row for row in support if not _is_registered(row)]
    money = [b for row in eligible for b in row if _MONEY_SHAPE.fullmatch(b.text.strip())]
    if not money:
        return None
    # Use horizontal position, not the order in which OCR emitted blocks.
    money.sort(key=lambda b: ((b.left + b.right) / 2, b.top))
    amount = money[-1]
    prices = [b for b in money[:-1] if b.block_id != amount.block_id]
    price = prices[-1] if prices else None
    quantity = next(
        (b for row in eligible for b in row if _QUANTITY.fullmatch(b.text.strip())
         and _money_value(b.text) is None),
        None,
    )
    unit = next(
        (b for row in eligible for b in row if _UNIT.fullmatch(b.text.strip())),
        None,
    )
    row = {
        "line_id": f"line-{index:04d}",
        "description": _cell("description", _clean_description(item.description), item.description, currency),
        "unit": _cell("unit", unit.text if unit else None, [unit] if unit else [], currency),
        "quantity": _cell("quantity", quantity.text if quantity else None, [quantity] if quantity else [], currency),
        "unit_price": _cell("unit_price", price.text if price else None, [price] if price else [], currency),
        "amount": _cell("amount", amount.text, [amount], currency),
    }
    # Retail row grouping is heuristic even when each OCR token is confident.
    add_reason(row["description"], "AMBIGUOUS_FORMAT")
    row_metadata(row)
    return row


def extract_retail_table(evidence: dict, *, currency: str) -> dict:
    """Return retail line items and diagnostics; no header/footer extraction.

    ``currency`` is an explicit caller hint, not inferred from OCR or gold. The
    integrator must establish it before accepting monetary values.
    """
    if currency != "VND":
        raise ValueError("Retail table currently supports only explicit VND context")
    pages: dict[int, list[list[Block]]] = {}
    for row in visual_rows(flatten(evidence)):
        pages.setdefault(row[0].page, []).append(row)
    lines = []
    page_diagnostics = []
    for page, rows in pages.items():
        page_line_start = len(lines)
        starts = [i for i, row in enumerate(rows) if any(_money_value(b.text) is not None for b in row)
                  and not _is_total(row) and not _is_registered(row)]
        # A registered price may precede the actual sale-price row; the product
        # description is recovered by looking backward from the first sale row.
        if not starts:
            page_diagnostics.append({"page": page, "status": "NO_PRICE_ROW"})
            continue
        first = starts[0]
        start = first
        for i in range(first - 1, max(-1, first - 3), -1):
            if _is_header(rows[i]) or _is_total(rows[i]):
                break
            if _description_blocks(rows[i]) and not _is_registered(rows[i]):
                start = i
                break
        current = _Item()
        orphan: list[list[Block]] = []
        stop = len(rows)
        for i in range(start, len(rows)):
            row = rows[i]
            if _is_total(row):
                stop = i
                break
            if current.description and not _description_blocks(row):
                existing = [b for prior in current.support for b in prior
                            if _money_value(b.text) is not None]
                next_values = [_money_value(b.text) for b in row]
                next_values = [value for value in next_values if value is not None]
                # A large isolated amount after a complete item is usually the
                # footer total whose label OCR placed on another visual row.
                if len(existing) >= 2 and len(next_values) == 1 and (
                    next_values[0] > 3 * max(_money_value(b.text) for b in existing)
                ):
                    stop = i
                    break
            if _is_header(row):
                continue
            if _is_registered(row) or any(_PRICE_LABEL.match(folded(b.text)) for b in row):
                if current.description:
                    current.support.append(row)
                continue
            desc = _description_blocks(row)
            # A parenthesized size or a weight fragment belongs to the item above.
            continuation = bool(desc) and all(
                b.text.lstrip().startswith("(") or re.fullmatch(r"x?\s*\d+[.,]\d+\s*kg", folded(b.text))
                for b in desc
            )
            if desc and not continuation:
                built = _build_item(current, len(lines) + 1, currency)
                if built:
                    lines.append(built)
                current = _Item(description=desc, prelude=orphan)
                orphan = []
                if any(_money_value(b.text) is not None for b in row):
                    current.support.append(row)
            elif current.description:
                if continuation:
                    current.description.extend(desc)
                current.support.append(row)
            else:
                orphan.append(row)
        built = _build_item(current, len(lines) + 1, currency)
        if built:
            lines.append(built)
        page_diagnostics.append({"page": page, "status": "PARSED", "start_row": start,
                                 "stop_row": stop, "line_count": len(lines) - page_line_start})
    output = {"schema_version": VERSION, "currency_context": currency,
              "line_items": lines, "pages": page_diagnostics}
    validate_retail_table(output, evidence)
    return output


def validate_retail_table(output: dict, evidence: dict) -> None:
    """Check source linkage, row identity and the signed-money cell contract."""
    source = {b.block_id: b for b in flatten(evidence)}
    if output["schema_version"] != VERSION or output["currency_context"] != "VND":
        raise ValueError("Unsupported retail table version or currency")
    line_ids = set()
    for row in output["line_items"]:
        if row["line_id"] in line_ids:
            raise ValueError("Duplicate retail line_id")
        line_ids.add(row["line_id"])
        if any(field not in row for field in LINE_FIELDS):
            raise ValueError("Missing retail line field")
        union = set()
        for field in LINE_FIELDS:
            cell = row[field]
            ids = cell["source_block_ids"]
            if len(ids) != len(set(ids)) or not set(ids) <= source.keys():
                raise ValueError("Invalid retail source_block_ids")
            ordered = sorted((source[i] for i in ids), key=lambda b: (b.page, b.reading_order))
            if ids != [b.block_id for b in ordered]:
                raise ValueError("Retail source IDs are not in reading order")
            expected = "\n".join(b.text for b in ordered) if ordered else None
            if cell["raw_text"] != expected:
                raise ValueError("Retail raw_text differs from OCR evidence")
            if cell["value_status"] == "PRESENT" and cell["normalized_value"] is None:
                raise ValueError("Present retail cell has no normalized value")
            if cell["value_status"] != "PRESENT" and cell["normalized_value"] is not None:
                raise ValueError("Non-present retail cell has a normalized value")
            union.update(ids)
        if set(row["source_block_ids"]) != union:
            raise ValueError("Retail row source IDs do not match cells")

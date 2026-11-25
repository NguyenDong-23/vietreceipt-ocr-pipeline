"""Find the first receipt totals row and the footer below it from OCR boxes."""
from dataclasses import dataclass
import math
import re
from statistics import median
import unicodedata

from rapidfuzz.distance import Levenshtein


PRIMARY = ('tong so', 'tong cong', 'tong tien', 'tong:', 'cong tien hang', 'thanh tien', 'so tien')
SECONDARY = ('thanh toan', 'khach phai tra', 'tien khach tra', 'tien khach dua',
             'tien mat', 'tien tra lai', 'tien thua', 'phu thu', 'chiet khau',
             'giam gia', 'cam on quy khach', 'han hanh phuc vu')


def fold(text):
    text = unicodedata.normalize('NFD', text.lower().replace('đ', 'd'))
    return ''.join(char for char in text if not unicodedata.combining(char))


def compact(text):
    return ''.join(re.findall('[a-z]', fold(text)))


def starts(value, cue):
    target = compact(cue)
    if value.startswith(target):
        return True
    budget = 2 if len(target) >= 7 else (1 if len(target) >= 5 else 0)
    if not budget:
        return False
    return any(Levenshtein.distance(value[:length], target, score_cutoff=budget) <= budget
               for length in range(len(target) - budget, len(target) + budget + 1))


def line_box(line):
    x, y = zip(*line['poly'])
    return min(x), min(y), max(x), max(y)


def line_height(line):
    points = line['poly']
    left = math.dist(points[0], points[3])
    right = math.dist(points[1], points[2])
    return max(1, (left + right) / 2)


def receipt_text_lines(lines):
    """Ignore tall vertical text off to the side of the receipt's horizontal text."""
    if not lines:
        return []

    typical_height = median(map(line_height, lines))
    boxes = [line_box(line) for line in lines]
    vertical = [bottom - top > 3 * max(right - left, typical_height)
                for left, top, right, bottom in boxes]
    horizontal = [box for box, is_vertical in zip(boxes, vertical) if not is_vertical]
    if not horizontal:
        return lines

    left_edge = min(box[0] for box in horizontal)
    right_edge = max(box[2] for box in horizontal)
    return [line for line, (left, _, right, _), is_vertical in zip(lines, boxes, vertical)
            if not is_vertical or (left < right_edge and right > left_edge)]


def row_position(line, slope):
    """Compare boxes in the same row even when the paper is tilted."""
    return sum(y - slope * x for x, y in line['poly']) / len(line['poly'])


def group_rows(lines, *, geometry=None):
    if not lines:
        return [], 1, 0

    if geometry is None:
        typical_height = median(map(line_height, lines))
        long_boxes = [line['poly'] for line in lines
                      if line['poly'][1][0] - line['poly'][0][0] > 2 * typical_height]
        slope = median((p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) for p in long_boxes) if long_boxes else 0
    else:
        typical_height, slope = geometry
    positions = sorted(((row_position(line, slope), line) for line in lines), key=lambda item: item[0])

    groups = []
    for position, line in positions:
        if groups and position - median(y for y, _ in groups[-1]) <= 0.6 * typical_height:
            groups[-1].append((position, line))
        else:
            groups.append([(position, line)])

    rows = []
    for group in groups:
        members = sorted((line for _, line in group), key=lambda line: line_box(line)[0])
        rows.append({'y': median(y for y, _ in group), 'members': members,
                     'text': ' '.join(line['text'] for line in members)})
    return rows, typical_height, slope


def text_from_lines(lines, *, geometry=None):
    return '\n'.join(row['text'] for row in group_rows(lines, geometry=geometry)[0])


def cue_kind(row):
    text = row['text']
    label = compact(text.split(':', 1)[0])
    if not label:
        return None
    if (re.search(r'\d', text) or ':' in text) and any(starts(label, cue) for cue in PRIMARY):
        return 'primary'
    if any(starts(label, cue) for cue in SECONDARY):
        return 'secondary'
    return None


def paddle_footer_cue(row):
    kind = cue_kind(row)
    if kind:
        return kind

    text = row['text']
    number = re.search(r'\d', text)
    label = compact(text[:number.start()] if number else text)
    # A tilted receipt can place the label and amount in different OCR rows.
    budget = 3 if number else 2
    if Levenshtein.distance(label, 'congtienhang', score_cutoff=budget) <= budget:
        return 'primary'
    if number is None:
        return None
    if label == 'tng' and ':' in text:
        return 'primary'

    # A clipped left edge can leave fragments such as “ng cng” or “ng s”.
    clipped = line_box(row['members'][0])[0] <= 2
    if clipped and label in {'ngcng', 'ngcong', 'ngtin', 'ngtien'}:
        return 'primary'
    if re.fullmatch(r'.*:\s*\d{1,3}\s*', text):
        near_quantity_label = Levenshtein.distance(label, 'tongso', score_cutoff=2) <= 2
        if near_quantity_label or (clipped and label in {'ngs', 'ngso', 'ongs', 'ongso'}):
            return 'primary'
    return None


def row_bounds(rows, typical_height, shape, *, through_bottom=False):
    height, width = shape
    boxes = [line_box(line) for row in rows for line in row['members']]
    left, top, right, bottom = zip(*boxes)
    pad_x = 0.30 * typical_height
    pad_y = 0.40 * typical_height
    return [max(0, math.floor(min(left) - pad_x)), max(0, math.floor(min(top) - pad_y)),
            min(width, math.ceil(max(right) + pad_x)),
            height if through_bottom else min(height, math.ceil(max(bottom) + pad_y))]


@dataclass(frozen=True)
class FooterRegion:
    box: list[int]
    method: str
    slope: float
    start_y: float | None

    def contains(self, line):
        return self.start_y is None or row_position(line, self.slope) >= self.start_y


def boxes_overlap(first, second):
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    overlap_width = min(right, other_right) - max(left, other_left)
    overlap_height = min(bottom, other_bottom) - max(top, other_top)
    if overlap_width <= 0 or overlap_height <= 0:
        return False
    smaller_area = min((right - left) * (bottom - top),
                       (other_right - other_left) * (other_bottom - other_top))
    # A crop pass can split one full-image detection into several smaller ones.
    return overlap_width * overlap_height >= 0.2 * smaller_area


def merge_footer_lines(crop_lines, page_lines, region):
    """Keep crop OCR and restore any footer text box only the page scan saw."""
    result = [line for line in receipt_text_lines(crop_lines) if region.contains(line)]
    seen_boxes = [line_box(line) for line in result]
    for line in receipt_text_lines(page_lines):
        if not region.contains(line) or line['score'] < 0.5 or not line['text'].strip():
            continue
        box = line_box(line)
        if not any(boxes_overlap(box, seen) for seen in seen_boxes):
            result.append({**line, 'source_pass': 'full_image'})
            seen_boxes.append(box)
    return result


def locate_footer(lines, shape):
    """Find the totals start and keep all footer content down to the image bottom."""
    rows, typical_height, slope = group_rows(receipt_text_lines(lines))
    if not rows:
        return None

    height, width = shape
    minimum_y = max(0.38 * height, rows[0]['y'])
    cues = [(index, paddle_footer_cue(row)) for index, row in enumerate(rows)]
    candidates = [(index, kind) for index, kind in cues if rows[index]['y'] + slope * width / 2 >= minimum_y]
    primary = next((index for index, kind in candidates if kind == 'primary'), None)
    # A photo of the footer alone can have its first totals row near the top.
    if primary is None:
        primary = next((index for index, kind in cues if kind == 'primary'), None)
    secondary = next((index for index, kind in candidates if kind == 'secondary'), None)
    if primary is not None:
        start, method = primary, 'summary cue'
    elif secondary is not None:
        start, method = secondary, 'payment cue'
    else:
        start, method = max(0, len(rows) - max(2, len(rows) // 4)), 'bottom-quarter fallback'

    start_y = None
    if start > 0:
        # Crop with padding, then ignore the previous row even if tilt puts a box inside the crop.
        previous = max(row_position(line, slope) for line in rows[start - 1]['members'])
        first = min(row_position(line, slope) for line in rows[start]['members'])
        start_y = (previous + first) / 2
    box = row_bounds(rows[start:], typical_height, shape, through_bottom=True)
    return FooterRegion(box, method, slope, start_y)

"""Trim overlapping padding in the existing skew frame; never reclassify text."""
import copy, math
import cv2
import numpy as np
from region_rules import rot, quad, bounds
from boundary_refiner import ink_box

VERSION = 'shared-gap-0.1'

def compact_regions(image, result, pp_blocks, ocr_lines):
    out = copy.deepcopy(result)
    regions = out.get('regions', [])
    if len(regions) < 2 or not ocr_lines:
        return out
    angle = math.radians(out.get('skew_degrees', 0))
    im = image if isinstance(image, np.ndarray) else cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return out
    lines = {o['ocr_id']: o for o in ocr_lines}
    changes, unresolved = [], []
    for previous, current in zip(regions, regions[1:]):
        pq, nq = rot(quad(previous), -angle), rot(quad(current), -angle)
        # Work on existing parallel quadrilaterals only. No extrapolated fitted line.
        if len(pq) != 4 or len(nq) != 4:
            continue
        if max(abs(pq[2,1]-pq[3,1]), abs(nq[0,1]-nq[1,1])) > 2:
            continue
        bottom, top = max(pq[2:,1]), min(nq[:2,1])
        if bottom < top:
            continue
        supports = []
        for region in (previous, current):
            boxes = []
            for ident in region.get('evidence_ocr_ids', []):
                if ident not in lines:
                    continue
                line = lines[ident]
                # Keep the OCR quad by default. Only replace oversized OCR support
                # with detected glyph support when that quad crosses this interface.
                b = bounds(rot(quad(line), -angle))
                if b[1] <= bottom and b[3] >= top:
                    ink = ink_box(im, line)
                    if ink:
                        x,y,a,z = ink
                        b = bounds(rot(np.array([[x,y],[a,y],[a,z],[x,z]]), -angle))
                boxes.append(b)
            # Preserve independent PP text/image attachments missed by OCR.
            for index in region.get('attached_pp_image_indices', []):
                block = next((b for i,b in enumerate(pp_blocks) if b.get('_source_index', i) == index), None)
                if block:
                    boxes.append(bounds(rot(quad(block), -angle)))
            supports.append(boxes)
        if not all(supports):
            unresolved.append(previous['region_id'] + '/' + current['region_id'])
            continue
        low = max(b[3] for b in supports[0])
        high = min(b[1] for b in supports[1])
        # A real, global gap is required: no semantic reassignment, no local
        # samples projected across the width, and no expansion of either region.
        margin = 1.0
        lower, upper = max(low + margin, top), min(high - margin, bottom)
        if upper - lower < 1:
            unresolved.append(previous['region_id'] + '/' + current['region_id'])
            continue
        cut = (lower + upper) / 2
        pq[2:,1] = cut - .5
        nq[:2,1] = cut + .5
        if min(pq[2:,1]) <= max(pq[:2,1]) or min(nq[2:,1]) <= max(nq[:2,1]):
            continue
        for region, q in ((previous,pq),(current,nq)):
            q = rot(q, angle)
            region.update(polygon=q.tolist(), bbox=bounds(q))
        changes.append(dict(previous=previous['region_id'], next=current['region_id'], gap=[low,high], cut=cut))
    out['compact_boundaries'] = dict(version=VERSION, changes=changes, unresolved=unresolved)
    if unresolved:
        out['warnings'] = sorted(set(out.get('warnings', []) + ['BOUNDARY_NO_SAFE_SHARED_GAP']))
    return out

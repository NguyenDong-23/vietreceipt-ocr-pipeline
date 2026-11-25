"""Join single-image layout regions with the Reader's canonical OCR evidence."""
from pathlib import Path
from io import BytesIO
import hashlib
import importlib
import json

import cv2
import numpy as np
from PIL import Image
from image_input import exif_matrix, transform_polygon


def read_with_layout(source_bytes, layout, *, receipt_id, ocr_run_id, reader=None):
    from ai.document_reader.contract import validate_document

    if layout.get('schema_version') != 'tv2-layout-handoff-0.1':
        raise ValueError('Unsupported layout schema')
    source = layout['source']
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != source['sha256']:
        raise ValueError('Layout and source SHA256 differ')
    if layout['page_count'] != 1 or len(layout['pages']) != 1:
        raise ValueError('This image adapter requires exactly one page')
    with Image.open(BytesIO(source_bytes)) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Expected one image frame')
        width, height = image.size
        orientation = int(image.getexif().get(274, 1))
    if (width, height) != (source['encoded_width_px'], source['encoded_height_px']):
        raise ValueError('Encoded source dimensions differ from layout')
    if orientation != source['exif_orientation']:
        raise ValueError('Source EXIF orientation differs from layout')
    matrix = exif_matrix(orientation, width, height)
    upright_width, upright_height = (height, width) if orientation >= 5 else (width, height)
    page = layout['pages'][0]
    if page['page_index'] != 0 or (page['width_px'], page['height_px']) != (width, height):
        raise ValueError('Layout page index or frame dimensions differ from source')
    regions = []
    seen = set()
    for region in page['regions']:
        ident = region['region_id']
        if ident in seen or region['label'] not in {'header', 'table', 'footer'}:
            raise ValueError('Duplicate region ID or unsupported semantic label')
        seen.add(ident)
        polygon = np.asarray(region['polygon_original_pixels'], dtype=float)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError('Expected finite four-corner region polygon')
        if not all(0 <= x <= width and 0 <= y <= height for x, y in polygon):
            raise ValueError('Region polygon exceeds source bounds')
        if not cv2.isContourConvex(polygon.astype(np.float32)):
            raise ValueError('Region polygon must be convex and nondegenerate')
        transformed = np.asarray(transform_polygon(polygon.tolist(), matrix.tolist()), np.float32)
        regions.append(dict(region_id=ident, label=region['label'], polygon=transformed))

    if reader is None:
        reader = importlib.import_module('ai.document_reader.backend').read_document
    document = reader(source_bytes, content_type=layout['media_type'],
                      receipt_id=receipt_id, ocr_run_id=ocr_run_id)
    validate_document(document)
    if document['receipt_id'] != str(receipt_id) or document['ocr_run_id'] != str(ocr_run_id):
        raise ValueError('Reader returned different run identities')
    if len(document['pages']) != 1:
        raise ValueError('Reader page count differs from source/layout')
    evidence = document['pages'][0]['evidence']
    if evidence['image'] != dict(width_px=upright_width, height_px=upright_height):
        raise ValueError('Reader output frame differs from EXIF-normalized source')

    links = []
    for block in evidence['blocks']:
        polygon = np.asarray([[p['x'] * upright_width, p['y'] * upright_height]
                              for p in block['polygon']], np.float32)
        area = abs(cv2.contourArea(polygon))
        matches = []
        overlaps = []
        if area > 0:
            for region in regions:
                intersection, _ = cv2.intersectConvexConvex(polygon, region['polygon'])
                fraction = max(0.0, min(1.0, float(intersection) / area))
                if fraction > 0:
                    overlaps.append(dict(region_id=region['region_id'], label=region['label'], block_area_fraction=fraction))
                if fraction >= 0.5:
                    matches.append(dict(region_id=region['region_id'], label=region['label'],
                                        block_area_fraction=fraction))
        status = 'ASSIGNED' if len(matches) == 1 else ('AMBIGUOUS' if matches else 'UNASSIGNED')
        links.append(dict(page_index=0, block_id=block['block_id'], status=status, matches=matches,
                          overlaps=overlaps, boundary_review_required=(len(matches)!=1 or matches[0]['block_area_fraction']<.99999),
                          text_review_status='NOT_EVALUATED'))
    sidecar = dict(schema_version='reader-layout-links-0.1', source_sha256=digest,
                   receipt_id=document['receipt_id'], ocr_run_id=document['ocr_run_id'],
                   page_count=1, coordinate_frame='EXIF-normalized source pixels',
                   policy='retain all regions covering at least half the OCR block area',
                   blocks=links,
                   counts={status: sum(link['status'] == status for link in links)
                           for status in ['ASSIGNED', 'AMBIGUOUS', 'UNASSIGNED']})
    return document, sidecar


def main():
    import argparse
    from uuid import uuid4

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--layout', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory')
    repo = Path(args.repo).resolve()
    import sys
    sys.path.insert(0, str(repo))
    sys.path.append(str(repo / '.dataset-deps'))
    source = Path(args.source).read_bytes()
    layout = json.loads(Path(args.layout).read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    try:
        document, links = read_with_layout(source, layout, receipt_id=uuid4(), ocr_run_id=uuid4())
        for name, payload in [('document.json', document), ('region-links.json', links)]:
            (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        status = dict(status='COMPLETE', reader_entrypoint='ai.document_reader.backend:read_document',
                      counts=links['counts'])
    except Exception as error:
        status = dict(status='FAILED', error_type=type(error).__name__, error=str(error))
        (output / 'run.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    (output / 'run.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status))


if __name__ == '__main__':
    main()

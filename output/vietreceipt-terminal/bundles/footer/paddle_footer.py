"""Read a receipt footer from its totals row down to the image bottom with PaddleOCR."""
import argparse
from pathlib import Path
from statistics import mean
import sys

from footer_layout import (boxes_overlap, group_rows, line_box, locate_footer, merge_footer_lines,
                           receipt_text_lines, row_position, text_from_lines)
from paddle_runtime import build_engine, capture_paddle_logs, predict_region, read_ocr_image, save_json


ROOT = Path(__file__).resolve().parent
# Keep the 20-image benchmark's layout resize. The footer itself uses original pixels.
RESIZE_FOR_LAYOUT = {'angled_14', 'angled_15', 'lowlight_19', 'lowlight_20'}


def read_box(engine, image, box):
    left, top, right, bottom = box
    return predict_region(engine, image[top:bottom, left:right].copy(), box)


def refine_footer_lines(engine, image, region, lines):
    """Retry weak recovered text and check a large unread margin at the bottom."""
    if not lines:
        return lines

    left, top, right, bottom = region.box
    refined = []
    for line in lines:
        if line.get('source_pass') == 'full_image' and line['score'] < 0.8:
            x1, y1, x2, y2 = line_box(line)
            pad = max(4, round(0.24 * (y2 - y1)))
            box = [max(left, int(x1) - pad), max(top, int(y1) - pad),
                   min(right, round(x2) + pad), min(bottom, round(y2) + pad)]
            if box[0] < box[2] and box[1] < box[3]:
                local = read_box(engine, image, box)['lines']
                candidates = [item for item in local if region.contains(item)
                              and boxes_overlap(line_box(item), (x1, y1, x2, y2))]
                if candidates and mean(item['score'] for item in candidates) >= line['score'] + 0.08:
                    refined.extend({**item, 'source_pass': 'isolated_line'} for item in candidates)
                    continue
        refined.append(line)

    _, typical_height, _ = group_rows(refined)
    deepest = max(line_box(line)[3] for line in refined)
    if bottom - deepest >= 3 * typical_height:
        box = [left, max(top, round(deepest - 2.2 * typical_height)), right, bottom]
        tail = receipt_text_lines(read_box(engine, image, box)['lines'])
        last_position = max(row_position(line, region.slope) for line in refined)
        refined.extend({**line, 'source_pass': 'bottom_strip'} for line in tail
                       if line['score'] >= 0.7
                       and row_position(line, region.slope) > last_position + 0.6 * typical_height)
    return refined


def read_footer(engine, path, max_side=0, show_full=False):
    layout_image, (width, height) = read_ocr_image(path, max_side)
    full = predict_region(engine, layout_image, [0, 0, width, height])
    region = locate_footer(full['lines'], (height, width))

    lines = []
    footer_seconds = 0
    box = None
    geometry = None
    method = 'no OCR rows'
    if region is not None:
        box, method = region.box, region.method
        original = layout_image if layout_image.shape[:2] == (height, width) else read_ocr_image(path)[0]
        footer = read_box(engine, original, box)
        lines = merge_footer_lines(footer['lines'], full['lines'], region)
        # A newly detected block can split an existing line. Keep the original row alignment.
        geometry = group_rows(lines)[1:]
        lines = refine_footer_lines(engine, original, region, lines)
        footer_seconds = footer['wall_s']

    result = {
        'box': box,
        'method': method,
        'text': text_from_lines(lines, geometry=geometry),
        'lines': lines,
        'row_geometry': geometry,
        'layout_input_shape': full['ocr_input_shape'],
        'layout_seconds': full['wall_s'],
        'footer_seconds': footer_seconds,
    }
    if show_full:
        result['full_lines'] = full['lines']
    return result


def print_ocr(lines, *, geometry=None):
    if not lines:
        print('Không đọc được chữ', flush=True)
        return

    rows, _, _ = group_rows(lines, geometry=geometry)
    for row in rows:
        for line in row['members']:
            print(f'{line["text"]} | Tin cậy {line["score"]:.2%}', flush=True)


def score_footer(result, reference, reference_amount):
    from score import bbox_scores, errors, payable, ratio

    counts = errors(reference['text'], result['text'])
    iou, _, _ = bbox_scores(reference['box'], result['box'])
    amount = payable(group_rows(result['lines'], geometry=result.get('row_geometry'))[0])
    return {
        'reference_box': reference['box'],
        'reference_text': reference['text'],
        'iou': iou,
        'cer': ratio(counts['char_edits'], counts['ref_chars']),
        'wer': ratio(counts['word_edits'], counts['ref_words']),
        'payable_vnd': amount,
        'reference_payable_vnd': reference_amount,
        'payable_correct': amount == reference_amount,
        **counts,
    }


def summarize(results):
    from score import ratio

    rows = list(results.values())
    return {
        'evaluated': len(rows),
        'mean_iou': mean(row['iou'] for row in rows),
        'cer': ratio(sum(row['char_edits'] for row in rows), sum(row['ref_chars'] for row in rows)),
        'wer': ratio(sum(row['word_edits'] for row in rows), sum(row['ref_words'] for row in rows)),
        'payable_correct': sum(row['payable_correct'] for row in rows),
    }


def benchmark_jobs(first, last):
    """Load the optional labeled benchmark from the original TestOCR workspace."""
    image_dir = ROOT.parent / '20_anh'
    benchmark_files = ['regions.py', 'score.py', 'reference_totals.json',
                       'reference_a.json', 'reference_b.json', 'reference_c.json']
    if not image_dir.is_dir() or not all((ROOT / name).is_file() for name in benchmark_files):
        raise ValueError('Use --image PATH to run on a receipt. The numbered benchmark needs the original TestOCR data.')

    from regions import load, references

    refs, amounts = references(), load('reference_totals.json')
    images = {int(path.stem[-2:]): path for path in image_dir.glob('*.jpg') if path.stem in refs}
    jobs = []
    for number in range(first, last + 1):
        path = images.get(number)
        if path is None or path.stem not in amounts:
            raise ValueError(f'Missing benchmark image or labels for receipt {number}')
        jobs.append((path, refs[path.stem], amounts[path.stem]))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Path to a receipt image')
    parser.add_argument('--model-dir', type=Path, help='Directory containing both Paddle inference models')
    parser.add_argument('--threads', type=int, default=4, help='PaddleOCR CPU threads')
    parser.add_argument('--output', type=Path, help='Save a JSON report to this path')
    parser.add_argument('--first', type=int, help='First image in the original benchmark (1 to 20)')
    parser.add_argument('--last', type=int, help='Last image in the original benchmark (1 to 20)')
    parser.add_argument('--show-full', action='store_true', help='Print full receipt text for debugging')
    args = parser.parse_args()

    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.image is not None:
        if args.first is not None or args.last is not None:
            parser.error('Choose --image or a benchmark range')
        if not args.image.is_file():
            parser.error(f'Image not found: {args.image}')
        jobs = [(args.image, None, None)]
        output = args.output
    else:
        first = 1 if args.first is None else args.first
        last = 20 if args.last is None else args.last
        if not 1 <= first <= last <= 20:
            parser.error('Expected 1 <= --first <= --last <= 20')
        try:
            jobs = benchmark_jobs(first, last)
        except ValueError as error:
            parser.error(str(error))
        output = args.output or ROOT / 'paddle_footer_result.json'

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        with capture_paddle_logs():
            engine = build_engine(args.threads, model_dir=args.model_dir)

        results = {}
        for path, reference, reference_amount in jobs:
            max_side = 1000 if reference is not None and path.stem in RESIZE_FOR_LAYOUT else 0
            with capture_paddle_logs():
                result = read_footer(engine, path, max_side=max_side, show_full=args.show_full)

            if reference is not None:
                result.update(score_footer(result, reference, reference_amount))
                results[path.stem] = result
                save_json(output, {'requested_images': len(jobs), 'summary': summarize(results), 'images': results})
            elif output is not None:
                save_json(output, result)

            if args.show_full:
                print_ocr(result['full_lines'])
            else:
                print_ocr(result['lines'], geometry=result['row_geometry'])
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

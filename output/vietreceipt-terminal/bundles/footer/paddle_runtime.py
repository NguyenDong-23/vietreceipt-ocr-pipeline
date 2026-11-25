"""Build PaddleOCR and map its text boxes back to the source image."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, TemporaryFile
import time


ROOT = Path(__file__).resolve().parent
WORKSPACE_CACHE = ROOT / '.paddlex'
DETECTION_MODEL = 'PP-OCRv5_server_det'
RECOGNITION_MODEL = 'latin_PP-OCRv5_mobile_rec'


@contextmanager
def capture_paddle_logs():
    """Hide startup noise in the terminal while preserving logs on errors."""
    stderr = sys.stderr
    fd = stderr.fileno()
    stderr.flush()

    def swap_paddlex_stream(source, target):
        for handler in logging.getLogger('paddlex').handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is source:
                handler.setStream(target)

    with TemporaryFile(mode='w+', encoding='utf-8', errors='replace') as logs:
        try:
            with os.fdopen(os.dup(fd), 'wb') as console:
                try:
                    # Capture both native output and Python's separate Windows console stream.
                    os.dup2(logs.fileno(), fd)
                    swap_paddlex_stream(stderr, logs)
                    with redirect_stdout(logs), redirect_stderr(logs):
                        yield
                finally:
                    os.dup2(console.fileno(), fd)
                    swap_paddlex_stream(logs, stderr)
        except BaseException:
            logs.seek(0)
            stderr.write(logs.read())
            raise


def _has_model_files(directory):
    topology = (directory / 'inference.json').is_file() or (directory / 'inference.pdmodel').is_file()
    return topology and all((directory / name).is_file() for name in ('inference.yml', 'inference.pdiparams'))


def _find_model_dir(model_dir):
    if model_dir is not None:
        directory = Path(model_dir).expanduser().resolve()
        if not all(_has_model_files(directory / name) for name in (DETECTION_MODEL, RECOGNITION_MODEL)):
            raise ValueError(f'Model directory {directory} needs complete {DETECTION_MODEL} and {RECOGNITION_MODEL} folders')
        return directory

    for directory in (ROOT / 'models', WORKSPACE_CACHE / 'official_models'):
        if all(_has_model_files(directory / name) for name in (DETECTION_MODEL, RECOGNITION_MODEL)):
            return directory
    return None


def build_engine(threads, model_dir=None):
    """Use local model files when available, otherwise use Paddle's model cache/download."""
    directory = _find_model_dir(model_dir)
    cache = WORKSPACE_CACHE if WORKSPACE_CACHE.is_dir() else ROOT / '.paddlex'
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(cache))
    if directory is not None:
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

    import torch  # Load Torch DLLs before PaddleX on Windows.
    from paddleocr import PaddleOCR

    local_models = {}
    if directory is not None:
        local_models = {
            'text_detection_model_dir': str(directory / DETECTION_MODEL),
            'text_recognition_model_dir': str(directory / RECOGNITION_MODEL),
        }
    return PaddleOCR(
        device='cpu', cpu_threads=threads, enable_mkldnn=False,
        text_detection_model_name=DETECTION_MODEL, text_recognition_model_name=RECOGNITION_MODEL,
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        **local_models,
    )


def read_ocr_image(path, max_side=0):
    """Read a BGR array for Paddle and retain the original image dimensions."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        if max_side and max(width, height) > max_side:
            scale = max_side / max(width, height)
            image.thumbnail((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    return np.asarray(image)[:, :, ::-1].copy(), (width, height)


def predict_region(engine, input_array, box):
    """Recognize the supplied crop and return polygons in original-image pixels."""
    import numpy as np

    start = time.perf_counter()
    x1, y1, x2, y2 = box
    result = engine.predict(input=input_array)[0].json
    payload = result.get('res', result)
    offsets = np.array([x1, y1])
    input_height, input_width = input_array.shape[:2]
    scale = np.array([(x2 - x1) / input_width, (y2 - y1) / input_height])
    texts, scores, polygons = (payload[key] for key in ('rec_texts', 'rec_scores', 'rec_polys'))
    if not len(texts) == len(scores) == len(polygons):
        raise RuntimeError('PaddleOCR returned inconsistent text, confidence and polygon counts')
    lines = [
        {'text': text, 'score': round(float(score), 6), 'poly': (np.asarray(poly) * scale + offsets).tolist()}
        for text, score, poly in zip(texts, scores, polygons)
    ]
    return {
        'box': box,
        'wall_s': round(time.perf_counter() - start, 3),
        'ocr_input_shape': [input_height, input_width],
        'lines': lines,
    }


def save_json(path, data):
    """Write a full JSON report before replacing the previous one."""
    output = NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False, mode='w', encoding='utf-8')
    temp = Path(output.name)
    try:
        with output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write('\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

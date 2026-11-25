"""Production-facing Header adapter: Reader geometry -> VietOCR text -> Header KIE.

No gold input. No automatic substitution of Paddle text into checkpoint output.
Pass existing Reader evidence to avoid detecting/reading the source twice.
"""
import copy
from pathlib import Path

from setup_runtime import configure
configure()
from checkpoint_ocr import CheckpointRecognizer, rerecognize_document, file_hash
from header import extract_header
from ocr import LiveReader


class CheckpointHeaderPipeline:
    def __init__(self, model_dir=None, *, recognizer=None, reader=None):
        self.recognizer = recognizer if recognizer is not None else CheckpointRecognizer(model_dir)
        self.reader = reader

    def process(self, source, *, evidence=None, reader_metadata=None, crop_dir=None):
        source = Path(source)
        if evidence is None:
            if self.reader is None:
                self.reader = LiveReader()
            evidence, reader_metadata = self.reader.read(source)
            reader_calls = 1
        else:
            reader_calls = 0
            pages = evidence.get('pages', [{'page_index':0, 'evidence':evidence}])
            orientation_pages = (reader_metadata or {}).get('pages', [])
            for page in pages:
                if page['evidence']['engine']['name'] == 'pdfium-text':
                    continue
                index = page['page_index']
                if index >= len(orientation_pages) or orientation_pages[index].get('orientation_clockwise') not in (0,90,180,270):
                    raise ValueError('Existing OCR evidence requires reader_metadata with orientation_clockwise for every OCR page. Obtain it from DocumentReader.process_with_metadata; do not guess the crop rotation.')
        original = copy.deepcopy(evidence)
        derived, provenance, changes = rerecognize_document(
            source, evidence, self.recognizer,
            reader_metadata=reader_metadata, crop_dir=crop_dir,
        )
        if evidence != original:
            raise AssertionError("Original Reader evidence changed")
        header = extract_header(derived)
        from ai.kie.v2.cells import add_reason
        changed_ids = {r['block_id'] for r in changes if r['different']}
        for cell in header['fields'].values():
            if changed_ids.intersection(cell['source_block_ids']):
                add_reason(cell, 'OCR_RECOGNIZER_DISAGREEMENT')
        return {
            'header': header,
            'evidence': derived,
            'paddle_evidence': original,
            'reader_metadata': reader_metadata or {},
            'checkpoint_provenance': provenance,
            'recognizer_comparison': changes,
            'selection': {
                'schema_version': 'header-recognizer-selection-1.0',
                'primary_recognizer': 'vietocr-checkpoint',
                'effective_text_source': 'pdfium-text' if not changes and provenance['pdf_text_pages_preserved'] else 'vietocr-checkpoint',
                'checkpoint_sha256': self.recognizer.sha256,
                'source_sha256': file_hash(source),
                'header_ocr_run_id': derived['ocr_run_id'],
                'parent_ocr_run_id': original['ocr_run_id'],
                'reader_calls_in_this_adapter': reader_calls,
                'paddle_text_role': 'reference_only; never substituted into OCR blocks',
                'paddle_fallback': False,
                'uses_gold_for_selection': False,
                'empty_prediction_policy': 'omit_and_record; do_not_fill_using_paddle',
                'checkpoint_failure_policy': 'raise_error; do_not_silently_switch_to_paddle',
                'pdf_text_policy': 'preserve native PDF text; rerecognize OCR pages only',
            },
        }

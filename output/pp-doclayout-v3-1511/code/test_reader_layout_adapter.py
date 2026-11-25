import copy
from io import BytesIO
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / 'vietreceipt'))
sys.path.append(str(ROOT.parent / 'vietreceipt/.dataset-deps'))
from PIL import Image
from ai.ocr.pipeline import build_ocr_result
from reader_layout_adapter import read_with_layout


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.receipt_id = uuid4()
        self.run_id = uuid4()

    def fixture(self, orientation=1):
        image = Image.new('RGB', (80, 120), 'white')
        exif = image.getexif()
        exif[274] = orientation
        stream = BytesIO()
        image.save(stream, format='PNG', exif=exif)
        payload = stream.getvalue()
        layout = dict(schema_version='tv2-layout-handoff-0.1', media_type='image/png', page_count=1,
                      source=dict(sha256=hashlib.sha256(payload).hexdigest(), encoded_width_px=80,
                                  encoded_height_px=120, exif_orientation=orientation),
                      pages=[dict(page_index=0, width_px=80, height_px=120, regions=[
                          dict(region_id='r1', label='header',
                               polygon_original_pixels=[[10, 10], [50, 10], [50, 30], [10, 30]])])])
        return payload, layout

    def document(self, polygon, width=80, height=120):
        evidence = build_ocr_result(dict(rec_texts=['fixture'], rec_scores=[0.9], rec_polys=[polygon]),
                                    receipt_id=self.receipt_id, ocr_run_id=self.run_id,
                                    width_px=width, height_px=height, duration_ms=1)
        evidence['engine'] = dict(name='test-fixture', version='1')
        return dict(schema_version='document-2.0', receipt_id=str(self.receipt_id),
                    ocr_run_id=str(self.run_id), pages=[dict(page_index=0, evidence=evidence)])

    def connect(self, payload, layout, document):
        return read_with_layout(payload, layout, receipt_id=self.receipt_id,
                                ocr_run_id=self.run_id, reader=lambda *args, **kwargs: document)

    def test_retains_original_reader_output(self):
        payload, layout = self.fixture()
        document = self.document([[20, 15], [40, 15], [40, 25], [20, 25]])
        snapshot = copy.deepcopy(document)
        result, links = self.connect(payload, layout, document)
        self.assertIs(result, document)
        self.assertEqual(document, snapshot)
        self.assertEqual(links['counts']['ASSIGNED'], 1)
        self.assertEqual(links['blocks'][0]['matches'][0]['label'], 'header')

    def test_rotated_source_matches_reader_upright_frame(self):
        payload, layout = self.fixture(6)
        document = self.document([[90, 20], [100, 20], [100, 30], [90, 30]], 120, 80)
        _, links = self.connect(payload, layout, document)
        self.assertEqual(links['counts']['ASSIGNED'], 1)

    def test_ambiguous_region_is_retained(self):
        payload, layout = self.fixture()
        extra = copy.deepcopy(layout['pages'][0]['regions'][0])
        extra.update(region_id='r2', label='table')
        layout['pages'][0]['regions'].append(extra)
        document = self.document([[20, 15], [40, 15], [40, 25], [20, 25]])
        _, links = self.connect(payload, layout, document)
        self.assertEqual(links['blocks'][0]['status'], 'AMBIGUOUS')
        self.assertEqual(len(links['blocks'][0]['matches']), 2)

    def test_hash_failure_happens_before_reader(self):
        payload, layout = self.fixture()
        reader = Mock()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            read_with_layout(payload + b'changed', layout, receipt_id=self.receipt_id,
                             ocr_run_id=self.run_id, reader=reader)
        reader.assert_not_called()

    def test_default_dispatch_uses_minh_entrypoint(self):
        payload, layout = self.fixture()
        document = self.document([[20, 15], [40, 15], [40, 25], [20, 25]])
        reader = Mock(return_value=document)
        with patch('reader_layout_adapter.importlib.import_module',
                   return_value=SimpleNamespace(read_document=reader)) as loader:
            read_with_layout(payload, layout, receipt_id=self.receipt_id, ocr_run_id=self.run_id)
        loader.assert_called_once_with('ai.document_reader.backend')
        reader.assert_called_once_with(payload, content_type='image/png',
                                       receipt_id=self.receipt_id, ocr_run_id=self.run_id)

    def test_wrong_reader_frame_rejected(self):
        payload, layout = self.fixture()
        document = self.document([[20, 15], [40, 15], [40, 25], [20, 25]], 81, 120)
        with self.assertRaisesRegex(ValueError, 'frame'):
            self.connect(payload, layout, document)

    def test_multiple_image_pages_rejected(self):
        payload, layout = self.fixture()
        layout['page_count'] = 2
        with self.assertRaisesRegex(ValueError, 'one page'):
            self.connect(payload, layout, {})


if __name__ == '__main__':
    unittest.main()

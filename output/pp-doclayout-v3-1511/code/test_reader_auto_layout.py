import copy,json,sys,tempfile,unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import Mock
import test_reader_layout_adapter as fixtures
from reader_auto_layout import integrate,make_layout,repair_minor_edges
from image_input import prepare_image

class AutoTests(unittest.TestCase):
    setUp=fixtures.AdapterTests.setUp
    fixture=fixtures.AdapterTests.fixture
    document=fixtures.AdapterTests.document
    connect=fixtures.AdapterTests.connect
    def prepared(self,payload):
        p=Path(__file__).resolve().parent/('test-source-'+str(uuid4())+'.png')
        p.write_bytes(payload);self.addCleanup(p.unlink)
        return p,prepare_image(p)

    def test_minor_edge_recovered_without_mutating_gold_or_ocr(self):
        payload,layout=self.fixture();path,prepared=self.prepared(payload)
        document=self.document([[20,15],[52,15],[52,25],[20,25]])
        before=copy.deepcopy(layout);doc_before=copy.deepcopy(document)
        fixed=repair_minor_edges(layout,document,prepared)
        self.assertEqual(fixed['boundary_adjustments'][0]['status'],'EXPANDED_MINOR_EDGE')
        self.assertAlmostEqual(fixed['pages'][0]['regions'][0]['bbox_original_pixels'][2],52,places=4)
        self.assertEqual(layout,before);self.assertEqual(document,doc_before)

    def test_expansion_cannot_eat_neighbor(self):
        payload,layout=self.fixture();path,prepared=self.prepared(payload)
        layout['pages'][0]['regions'].append(dict(region_id='neighbor',label='table',polygon_original_pixels=[[51,10],[70,10],[70,30],[51,30]]))
        doc=self.document([[20,15],[52,15],[52,25],[20,25]])
        fixed=repair_minor_edges(layout,doc,prepared)
        self.assertEqual(fixed['boundary_adjustments'][0]['status'],'WOULD_INCREASE_REGION_OVERLAP')
        self.assertEqual(fixed['pages'][0]['regions'],layout['pages'][0]['regions'])

    def test_integrate_calls_reader_once_and_uses_same_blocks(self):
        payload,layout=self.fixture();path,prepared=self.prepared(payload)
        doc=self.document([[20,15],[40,15],[40,25],[20,25]]);snapshot=copy.deepcopy(doc)
        prediction=dict(source=prepared.source,input_sha256=prepared.source['sha256'],regions_original=[dict(region_id='r1',label='header',polygon=[[10,10],[50,10],[50,30],[10,30]])],pp_checkpoint_sha256='fixture',ocr_provenance=doc['pages'][0]['evidence']['engine'],code_sha256={})
        engine=Mock();engine.predict.return_value=(prediction,prepared.image)
        reader=Mock(return_value=doc)
        result=integrate(path,engine,receipt_id=self.receipt_id,ocr_run_id=self.run_id,reader=reader)
        self.assertEqual(reader.call_count,1);self.assertEqual(doc,snapshot)
        self.assertEqual(engine.predict.call_args.kwargs['ocr_lines'][0]['ocr_id'],doc['pages'][0]['evidence']['blocks'][0]['block_id'])
        self.assertEqual(result['region_evidence']['regions'][0]['source_blocks'],doc['pages'][0]['evidence']['blocks'])

    def test_partial_block_is_flagged_even_when_assigned(self):
        payload,layout=self.fixture();doc=self.document([[20,15],[52,15],[52,25],[20,25]])
        _,links=self.connect(payload,layout,doc)
        self.assertEqual(links['blocks'][0]['status'],'ASSIGNED')
        self.assertTrue(links['blocks'][0]['boundary_review_required'])
        self.assertEqual(links['blocks'][0]['text_review_status'],'NOT_EVALUATED')

if __name__=='__main__':unittest.main()

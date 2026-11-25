"""Reader once -> PP layout + same Reader evidence -> immutable linked document."""
from pathlib import Path
import copy,hashlib,json,mimetypes
import numpy as np
import cv2
from image_input import prepare_image,transform_polygon
from reader_layout_adapter import read_with_layout

def reader_lines(document, prepared):
    from ai.document_reader.contract import validate_document
    validate_document(document)
    if len(document['pages'])!=1 or document['pages'][0]['page_index']!=0:
        raise ValueError('Single image/page only')
    ev=document['pages'][0]['evidence'];w,h=prepared.image.size
    if ev['image']!={'width_px':w,'height_px':h}:
        raise ValueError('Reader frame differs from EXIF-upright source; do not guess rotation')
    return [dict(ocr_id=b['block_id'],text=b['text'],confidence=b['confidence'],
                 polygon=[[p['x']*w,p['y']*h] for p in b['polygon']]) for b in ev['blocks']]

def make_layout(path, prediction):
    s=prediction['source'];w,h=s['encoded_width'],s['encoded_height']
    regions=[]
    for r in prediction['regions_original']:
        poly=r['polygon']
        regions.append(dict(region_id=r['region_id'],label=r['label'],polygon_original_pixels=poly,
            bbox_original_pixels=[min(p[0] for p in poly),min(p[1] for p in poly),max(p[0] for p in poly),max(p[1] for p in poly)],
            polygon_original_normalized=[dict(x=x/w,y=y/h) for x,y in poly]))
    return dict(schema_version='tv2-layout-handoff-0.1',media_type=mimetypes.guess_type(str(path))[0] or 'application/octet-stream',
        page_count=1,annotation_status='PREDICTION_PENDING_REVIEW',
        source=dict(sha256=s['sha256'],encoded_width_px=w,encoded_height_px=h,exif_orientation=s['exif_orientation']),
        pages=[dict(page_index=0,width_px=w,height_px=h,regions=regions)],
        provenance=dict(layout_model='PP-DocLayoutV3',checkpoint_sha256=prediction['pp_checkpoint_sha256'],
            semantic_adapter='PP native + original Reader text and coordinates',ocr=prediction['ocr_provenance'],code_sha256=prediction['code_sha256']))

def repair_minor_edges(layout, document, prepared, max_extension_px=8):
    """Expand only a uniquely >=90%-contained block, at most 8px, no new region overlap."""
    out=copy.deepcopy(layout);regions=out['pages'][0]['regions'];events=[]
    forward=prepared.source['original_to_upright_matrix'];inverse=prepared.source['upright_to_original_matrix']
    polygons=[np.array(transform_polygon(r['polygon_original_pixels'],forward),np.float32) for r in regions]
    w,h=prepared.image.size
    for b in document['pages'][0]['evidence']['blocks']:
        p=np.array([[v['x']*w,v['y']*h] for v in b['polygon']],np.float32);area=abs(cv2.contourArea(p))
        if area<=0:continue
        hits=[max(0,min(1,cv2.intersectConvexConvex(p,q)[0]/area)) for q in polygons]
        eligible=[i for i,f in enumerate(hits) if f>=.9]
        if len(eligible)!=1:continue
        i=eligible[0]
        if hits[i]>=.99999:continue
        q=polygons[i];basis=np.column_stack([q[1]-q[0],q[3]-q[0]])
        if abs(np.linalg.det(basis))<1 or np.linalg.norm(q[0]+q[2]-q[1]-q[3])>1:continue
        uv=(p-q[0])@np.linalg.inv(basis).T
        lo=np.minimum([0,0],uv.min(0));hi=np.maximum([1,1],uv.max(0))
        new=(np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])@basis.T+q[0]).astype(np.float32)
        reason=None
        if np.linalg.norm(new-q,axis=1).max()>max_extension_px:reason='EXCEEDS_EXTENSION_LIMIT'
        elif np.any(new<0) or np.any(new[:,0]>w) or np.any(new[:,1]>h):reason='OUTSIDE_IMAGE'
        elif any(cv2.intersectConvexConvex(new,other)[0]>cv2.intersectConvexConvex(q,other)[0]+.01 for j,other in enumerate(polygons) if j!=i):reason='WOULD_INCREASE_REGION_OVERLAP'
        event=dict(block_id=b['block_id'],region_id=regions[i]['region_id'],before_fraction=hits[i],status=reason or 'EXPANDED_MINOR_EDGE')
        if reason is None:
            poly=transform_polygon(new.tolist(),inverse);r=regions[i];r['polygon_original_pixels']=poly
            r['bbox_original_pixels']=[min(x for x,y in poly),min(y for x,y in poly),max(x for x,y in poly),max(y for x,y in poly)]
            sw,sh=prepared.source['encoded_width'],prepared.source['encoded_height']
            r['polygon_original_normalized']=[dict(x=x/sw,y=y/sh) for x,y in poly];polygons[i]=new
        events.append(event)
    out['boundary_adjustments']=events
    return out

def integrate(path, engine, *, receipt_id, ocr_run_id, reader=None):
    from importlib import import_module
    path=Path(path);source=path.read_bytes();prepared=prepare_image(path)
    reader=reader or import_module('ai.document_reader.backend').read_document
    media_type=mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    document=reader(source,content_type=media_type,receipt_id=receipt_id,ocr_run_id=ocr_run_id)
    if document['receipt_id']!=str(receipt_id) or document['ocr_run_id']!=str(ocr_run_id):raise ValueError('Reader identities differ')
    lines=reader_lines(document,prepared)
    prediction,image=engine.predict(path,ocr_lines=lines,ocr_provenance=document['pages'][0]['evidence']['engine'])
    if prediction['input_sha256']!=hashlib.sha256(source).hexdigest():raise ValueError('Input changed during processing')
    original=make_layout(path,prediction);layout=repair_minor_edges(original,document,prepared)
    result,links=read_with_layout(source,layout,receipt_id=receipt_id,ocr_run_id=ocr_run_id,reader=lambda *a,**kw:document)
    # Full canonical blocks, never crop text to the region or infer KIE values.
    ev={b['block_id']:b for b in document['pages'][0]['evidence']['blocks']}
    bundles=[dict(region_id=r['region_id'],label=r['label'],polygon_original_pixels=r['polygon_original_pixels'],
                  source_blocks=[copy.deepcopy(ev[b['block_id']]) for b in links['blocks'] if any(m['region_id']==r['region_id'] for m in b['matches'])]) for r in layout['pages'][0]['regions']]
    return dict(document=result,layout=layout,layout_before_boundary_guard=original,region_links=links,
                region_evidence=dict(schema_version='reader-region-evidence-0.1',source_sha256=layout['source']['sha256'],receipt_id=result['receipt_id'],ocr_run_id=result['ocr_run_id'],page_index=0,
                    image=document['pages'][0]['evidence']['image'],block_coordinate_frame='normalized 0-1 on EXIF-upright source',region_coordinate_frame='original encoded source pixels',
                    original_to_upright_matrix=prepared.source['original_to_upright_matrix'],text_review_status='NOT_EVALUATED',regions=bundles),prediction=prediction)

def main():
    import argparse,sys
    from uuid import uuid4
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--repo',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('Use a new output directory')
    repo=Path(a.repo).resolve();sys.path.insert(0,str(repo));sys.path.append(str(repo/'.dataset-deps'))
    out.mkdir(parents=True)
    try:
        from pipeline import LayoutPipeline
        result=integrate(a.source,LayoutPipeline(external_ocr=True,model_path=a.model_dir),receipt_id=uuid4(),ocr_run_id=uuid4())
        for name,payload in result.items():(out/(name.replace('_','-')+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        status=dict(status='COMPLETE_NOT_TEXT_QA_VERIFIED',reader_entrypoint='ai.document_reader.backend:read_document',counts=result['region_links']['counts'])
    except Exception as e:
        (out/'run.json').write_text(json.dumps(dict(status='FAILED',error_type=type(e).__name__,error=str(e))),encoding='utf-8');raise
    (out/'run.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status))

if __name__=='__main__':main()

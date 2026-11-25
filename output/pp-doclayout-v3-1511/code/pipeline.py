"""Live single/batch layout engine. Both detectors consume the same upright pixels."""
from pathlib import Path
import sys,os,json,hashlib,time
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent
sys.path.insert(0,str(WS/'pretrained-layout-comparison/runtime'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(WS/'pretrained-layout-comparison/hf-cache')
from image_input import prepare_image
import region_rules,visual_fallback
from PIL import ImageDraw,ImageFont
import torch
from transformers import AutoImageProcessor,AutoModelForObjectDetection
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class LayoutPipeline:
 def __init__(self, *, external_ocr=False, model_path=None):
  start=time.perf_counter();torch.set_num_threads(4);model_path=Path(model_path) if model_path else WS/'pretrained-layout-comparison/PP-DocLayoutV3_safetensors';protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'));self.pp_hash=sha(model_path/'model.safetensors')
  if self.pp_hash!=protocol['pp_checkpoint']['sha256']:raise ValueError('PP checkpoint hash does not match protocol')
  self.processor=AutoImageProcessor.from_pretrained(str(model_path),local_files_only=True)
  self.model=AutoModelForObjectDetection.from_pretrained(str(model_path),local_files_only=True,attn_implementation='eager').eval()
  if external_ocr:self.ocr=None
  else:
   from ocr_engine import ReceiptOCR
   self.ocr=ReceiptOCR()
  if self.ocr is not None and self.ocr.model_sha256!=protocol['ocr_recognizer']['model_sha256']:raise ValueError('OCR checkpoint hash does not match protocol')
  self.model_load_seconds=time.perf_counter()-start

 def predict(self,path, *, ocr_lines=None, ocr_provenance=None):
  start=time.perf_counter();prepared=prepare_image(path);im=prepared.image;bgr=prepared.bgr()
  with torch.inference_mode():raw=self.model(**self.processor(images=im,return_tensors='pt'))
  detections=self.processor.post_process_object_detection(raw,threshold=.25,target_sizes=[im.size[::-1]])[0];blocks=[]
  for i,(score,label,box) in enumerate(zip(detections['scores'],detections['labels'],detections['boxes'])):
   p=detections['polygon_points'][i];blocks.append(dict(native_label=self.model.config.id2label[int(label)],confidence=float(score),bbox=box.tolist(),polygon=p.tolist() if hasattr(p,'tolist') else p))
  if ocr_lines is None and self.ocr is None:raise ValueError('External OCR mode requires Reader lines')
  lines=self.ocr(bgr) if ocr_lines is None else ocr_lines
  result=visual_fallback.apply_visual_fallback(bgr,region_rules.build_regions(blocks,lines,im.width,im.height),blocks,lines)
  modules={'pipeline':Path(__file__),'image_input':ROOT/'image_input.py','ocr_engine':ROOT/'ocr_engine.py','region_rules':Path(region_rules.__file__),'visual_fallback':Path(visual_fallback.__file__)}
  if 'boundary_refiner' in sys.modules:modules['boundary_refiner']=Path(sys.modules['boundary_refiner'].__file__)
  if 'compact_regions' in sys.modules:modules['compact_regions']=Path(sys.modules['compact_regions'].__file__)
  result.update(schema_version='layout-prediction-1.1',input_file=str(Path(path).resolve()),input_sha256=prepared.source['sha256'],source=prepared.source,width=im.width,height=im.height,coordinate_system='pixel_xy_on_exif_normalized_image',regions_original=prepared.regions_in_source_frame(result['regions']),evidence_coordinate_system='pixel_xy_on_exif_normalized_image',pp_native=blocks,ocr_lines=lines,pp_checkpoint_sha256=self.pp_hash,ocr_recognizer_sha256=self.ocr.model_sha256 if ocr_lines is None else None,ocr_provenance=ocr_provenance,code_sha256={k:sha(p) for k,p in modules.items()},annotation_status='PREDICTION_PENDING_REVIEW',training=False,elapsed_seconds=time.perf_counter()-start)
  return result,im

def write_prediction(result,image,dest):
 dest=Path(dest);dest.mkdir(exist_ok=True,parents=True);(dest/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 from preview_render import render_preview
 render_preview(image,result['regions']).save(dest/'preview.jpg',quality=97)

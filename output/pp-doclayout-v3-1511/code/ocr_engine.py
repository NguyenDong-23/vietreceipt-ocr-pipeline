from pathlib import Path
import sys,os,json,hashlib,math,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np,cv2,onnxruntime as ort
from rapidocr import RapidOCR,LangRec
class ReceiptOCR:
 def __init__(self):
  self.det=RapidOCR(params={'Global.use_rec':False,'Global.use_cls':False,'Global.font_path':'C:/Windows/Fonts/arial.ttf','Global.max_side_len':1600,'Global.log_level':'error','Det.model_path':str(ROOT/'models/ch_PP-OCRv4_det_infer.onnx'),'Det.limit_side_len':1600,'Det.limit_type':'max','Cls.model_path':str(ROOT/'models/ch_ppocr_mobile_v2.0_cls_infer.onnx'),'Rec.model_path':str(ROOT/'models/latin_PP-OCRv3_rec_infer.onnx'),'Rec.lang_type':LangRec.LATIN,'EngineConfig.onnxruntime.intra_op_num_threads':2,'EngineConfig.onnxruntime.inter_op_num_threads':1})
  p=ROOT/'models/vi-rec';self.characters=['']+(p/'charset.txt').read_text(encoding='utf-8').splitlines()+[' '];opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;self.rec=ort.InferenceSession(str(p/'recognition.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);self.model_sha256=hashlib.sha256((p/'recognition.onnx').read_bytes()).hexdigest()
 def __call__(self,image):
  im=image if isinstance(image,np.ndarray) else cv2.imdecode(np.fromfile(str(image),dtype=np.uint8),cv2.IMREAD_COLOR)
  if im is None or im.ndim!=3 or im.shape[2]!=3:raise ValueError('OCR requires a readable BGR image')
  out=self.det(im,use_rec=False,use_cls=False);boxes=[] if out.boxes is None else out.boxes;lines=[];crops=[]
  for i,poly in enumerate(boxes):
   p=np.asarray(poly,dtype=np.float32);w=round(max(np.linalg.norm(p[0]-p[1]),np.linalg.norm(p[2]-p[3])));h=round(max(np.linalg.norm(p[0]-p[3]),np.linalg.norm(p[1]-p[2])))
   if min(w,h)<3:continue
   crop=cv2.warpPerspective(im,cv2.getPerspectiveTransform(p,np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)),(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
   if h/w>1.5:crop=np.rot90(crop).copy()
   target=320 if crop.shape[1]/crop.shape[0]<=320/48 else 640;neww=min(target,max(1,math.ceil(48*crop.shape[1]/crop.shape[0])));x=cv2.resize(crop,(neww,48)).astype(np.float32)/255.;x=(x-.5)/.5;pad=np.zeros((3,48,target),np.float32);pad[:,:,:neww]=x.transpose(2,0,1);crops.append((i,p.tolist(),target,pad))
  for target in [320,640]:
   group=[x for x in crops if x[2]==target]
   for k in range(0,len(group),6):
    chunk=group[k:k+6];output=self.rec.run(None,{self.rec.get_inputs()[0].name:np.stack([x[3] for x in chunk])})[0];assert output.shape[-1]==len(self.characters)
    for item,prob in zip(chunk,output):
     idx=prob.argmax(-1);conf=prob.max(-1);tokens=[];scores=[];last=-1
     for token,s in zip(idx,conf):
      token=int(token)
      if token!=0 and token!=last:tokens.append(self.characters[token]);scores.append(float(s))
      last=token
     text=''.join(tokens).strip()
     if text:lines.append(dict(ocr_id='ocr-'+str(item[0]),text=text,confidence=float(np.mean(scores)) if scores else 0,polygon=item[1]))
  return sorted(lines,key=lambda x:sum(p[1] for p in x['polygon'])/4)

def main():
 meta=json.loads((ROOT.parent/'layout-training-draft/dataset-manifest.json').read_text(encoding='utf-8'));out=ROOT/'ocr-cache';out.mkdir(exist_ok=True);engine=ReceiptOCR()
 for r in meta['records']:
  p=out/(r['image_id']+'.json')
  if p.exists():
   saved=json.loads(p.read_text(encoding='utf-8'))
   if saved['image_sha256']==r['image_sha256'] and saved['recognizer_sha256']==engine.model_sha256:continue
  start=time.time();lines=engine(r['image']);p.write_text(json.dumps(dict(image_id=r['image_id'],image_sha256=r['image_sha256'],recognizer_sha256=engine.model_sha256,lines=lines,seconds=time.time()-start),ensure_ascii=False,indent=2),encoding='utf-8');print(r['image_id'],len(lines),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()

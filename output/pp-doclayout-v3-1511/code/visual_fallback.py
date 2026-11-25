"""Conservative geometry fallback for unreadable zones; never synthesizes text."""
import copy,math
import cv2,numpy as np
from region_rules import bounds,quad,rot
VERSION='unreadable-geometry-0.3'
def _apply_unreadable_fallback(image,result,pp_blocks,ocr_lines):
 out=copy.deepcopy(result);tables=[r for r in out['regions'] if r['label']=='table'];footers=[r for r in out['regions'] if r['label']=='footer']
 # Only missing-header cases with independently established table/footer anchors.
 if any(r['label']=='header' for r in out['regions']) or not tables or not footers:return out
 im=image if isinstance(image,np.ndarray) else cv2.imdecode(np.fromfile(str(image),np.uint8),cv2.IMREAD_COLOR)
 if im is None:return out
 height,width=im.shape[:2];angle=math.radians(out.get('skew_degrees',0));tb=bounds(rot(quad(tables[0]),-angle));lastf=footers[-1];fb=bounds(rot(quad(lastf),-angle));heights=[bounds(rot(quad(x),-angle))[3]-bounds(rot(quad(x),-angle))[1] for x in ocr_lines if x.get('confidence',0)>=.4];hm=max(4,float(np.median(heights))) if heights else max(width,height)/50
 support=[]
 for i,p in enumerate(pp_blocks):
  if p.get('native_label') not in ['doc_title','text','table','header','paragraph_title']:continue
  b=bounds(rot(quad(p),-angle))
  if b[1]<tb[1]-hm and min(b[2],tb[2])-max(b[0],tb[0])>0:support.append((i,b))
 if not support and out.get('table_start_reason') not in ['description_and_right_aligned_prices','item_with_pp_table_support']:return out
 miny=min((b[1] for _,b in support),default=hm)-hm;docx=[min([tb[0],fb[0]]+[b[0] for _,b in support])-hm,max([tb[2],fb[2]]+[b[2] for _,b in support])+hm]
 scale=min(1.,900/max(width,height));small=cv2.resize(im,None,fx=scale,fy=scale);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
 # Estimate paper color from light pixels around the already established item table.
 lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB).astype(float);source_table=bounds(quad(tables[0]));x,y,a,b=[int(v*scale) for v in source_table];patch=lab[max(0,y):min(lab.shape[0],b+1),max(0,x):min(lab.shape[1],a+1)].reshape(-1,3)
 if len(patch)<10:return out
 light=patch[patch[:,0]>=np.quantile(patch[:,0],.6)];paper_color=np.median(light,axis=0);color_distance=np.linalg.norm(lab[:,:,1:]-paper_color[1:],axis=2);binary=((color_distance<10)&(np.abs(lab[:,:,0]-paper_color[0])<65)).astype(np.uint8)*255;binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 center=rot(np.array([[(tb[0]+tb[2])/2,(tb[1]+tb[3])/2]]),angle)[0]*scale;selected=[c for c in contours if cv2.pointPolygonTest(c,tuple(map(float,center)),False)>=0]
 if not selected:return out
 paper=max(selected,key=cv2.contourArea)
 if cv2.contourArea(paper)<.12*gray.size:return out
 mask=np.zeros_like(gray);cv2.drawContours(mask,[paper],-1,255,-1);hs=hm*scale;erode=max(3,round(hs*.45));mask=cv2.erode(mask,np.ones((erode,erode),np.uint8));background=cv2.GaussianBlur(gray,(0,0),max(3,min(15,hs*.33)));signal=cv2.subtract(background,gray);ink=((signal>3)&(mask>0)).astype(np.uint8)*255;ink=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((2,2),np.uint8));ink=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((max(2,round(hs*.1)),max(5,round(hs*.55))),np.uint8));components,_=cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);accepted=[]
 for c in components:
  x,y,w,h=cv2.boundingRect(c)
  if w<max(8,hs*.55) or h<max(2,hs*.1) or w*h<hs*hs*.1 or w/max(h,1)>22:continue
  p=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],float)/scale;b=bounds(rot(p,-angle));cx=(b[0]+b[2])/2
  if docx[0]<=cx<=docx[1] and b[1]>=miny:accepted.append(b)
 above=[b for b in accepted if b[3]<tb[1]-hm*.15 and b[2]-b[0]>=hm];below=[b for b in accepted if b[1]>=fb[1]]
 if len(above)<3 or max(b[3] for b in above)-min(b[1] for b in above)<2*hm:return out
 def region_geometry(box):
  x,y,a,b=box;p=rot(np.array([[x,y],[a,y],[a,b],[x,b]],float),angle);p[:,0]=p[:,0].clip(0,width);p[:,1]=p[:,1].clip(0,height);return dict(polygon=p.tolist(),bbox=bounds(p))
 pad=hm*.22;hb=[min(b[0] for b in above)-pad,min(b[1] for b in above)-pad,max(b[2] for b in above)+pad,min(tb[1]-hm*.15,max(b[3] for b in above)+pad)]
 header=dict(region_id='visual-header',label='header',geometry='polygon',**region_geometry(hb),evidence_ocr_ids=[],evidence_pp_indices=[i for i,_ in support],decision_rows=[],origin='visual_geometry_fallback',role_status='INFERRED_NEEDS_REVIEW',content_status='UNREADABLE_OR_REDACTED',pixel_component_boxes=above)
 out['regions'].insert(0,header);out['warnings']=[w for w in out['warnings'] if w!='NO_HEADER_EVIDENCE'];out['warnings'].append('HEADER_INFERRED_FROM_UNREADABLE_BLOCKS')
 if below and max(b[3] for b in below)>fb[3]+hm*.5:
  expanded=[min([fb[0]]+[b[0] for b in below])-pad,fb[1],max([fb[2]]+[b[2] for b in below])+pad,max(b[3] for b in below)+pad];lastf.update(region_geometry(expanded));lastf['origin']='ocr_plus_visual_geometry_fallback';lastf['content_status']='PARTIALLY_READABLE';lastf['role_status']='INFERRED_EXTENSION_NEEDS_REVIEW';lastf['pixel_component_boxes']=below;out['warnings'].append('FOOTER_EXTENDED_TO_UNREADABLE_BLOCKS')
 out['visual_fallback']=dict(version=VERSION,applied=True,method='inside-paper local contrast components + PP overlap + OCR table/footer anchors',recovered_text=False);out['needs_review']=True;out['warnings']=sorted(set(out['warnings']));return out

def apply_visual_fallback(image,result,pp_blocks,ocr_lines):
 from boundary_refiner import refine_boundaries
 from compact_regions import compact_regions
 return compact_regions(image,_apply_unreadable_fallback(image,result,pp_blocks,ocr_lines),pp_blocks,ocr_lines)

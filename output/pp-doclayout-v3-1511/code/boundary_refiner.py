"""Refine overlapping semantic interfaces using image ink, never label coordinates."""
import copy,re,math
import cv2,numpy as np
from region_rules import bounds,quad,norm,footer_reason
VERSION='ink-boundaries-0.2-conservative'

def safe_separator(samples,slope_hint,gap):
 """Find a line inside all observed glyph gaps instead of averaging through a word."""
 limit=math.tan(math.radians(12));best=None
 for slope in np.linspace(max(-limit,slope_hint-.1),min(limit,slope_hint+.1),401):
  low=max(p+gap-slope*x for x,_,p,n in samples);high=min(n-gap-slope*x for x,_,p,n in samples)
  if low>high:continue
  intercept=float(np.clip(np.mean([mid-slope*x for x,mid,_,_ in samples]),low,high))
  cost=sum((slope*x+intercept-mid)**2 for x,mid,_,_ in samples)
  if best is None or cost<best[0]:best=(cost,float(slope),intercept)
 return None if best is None else best[1:]

def ink_box(im,line):
 q=quad(line);raw=bounds(q);x,y,a,b=[int(v) for v in raw];x=max(0,x);y=max(0,y);a=min(im.shape[1],a+1);b=min(im.shape[0],b+1)
 if a-x<4 or b-y<4:return None
 crop=im[y:b,x:a];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);_,bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 if gray.max()-gray.min()<20:return None
 mask=np.zeros_like(bw);cv2.fillPoly(mask,[np.round(q-[x,y]).astype(np.int32)],255);bw=cv2.bitwise_and(bw,mask)
 h=b-y;count,labels,stats,centers=cv2.connectedComponentsWithStats(bw,8);parts=[]
 edge=cv2.subtract(mask,cv2.erode(mask,np.ones((3,3),np.uint8)))>0
 for component,(xx,yy,w,hh,area) in enumerate(stats[1:],1):
  if yy<=0 or yy+hh>=h or xx<=0 or xx+w>=a-x:continue
  if np.any(edge[labels==component]):continue
  if area<max(2,h*h*.0015) or (w/max(1,hh)>8 and hh<h*.2):continue
  parts.append([xx,yy,xx+w,yy+hh,hh,int(area)])
 main=[p for p in parts if p[4]>=h*.22]
 if len(main)<2:return None
 # Oversized OCR quads sometimes include pieces of a neighbouring text row.
 clusters=[]
 for p in sorted(main,key=lambda p:(p[1]+p[3])/2):
  match=next((g for g in clusters if min(max(q[3] for q in g),p[3])-max(min(q[1] for q in g),p[1])>=min(p[4],max(q[4] for q in g))*.25),None)
  if match is None:clusters.append([p])
  else:match.append(p)
 main=max(clusters,key=lambda g:sum(p[5] for p in g))
 if len(main)<2:return None
 top=min(p[1] for p in main);bottom=max(p[3] for p in main)
 selected=[p for p in parts if p[3]>=top-h*.28 and p[1]<=bottom+h*.04]
 return [float(x+min(p[0] for p in selected)),float(y+min(p[1] for p in selected)),float(x+max(p[2] for p in selected)),float(y+max(p[3] for p in selected))]

def _rule_line(im,x0,x1,y0,y1,target,slope_hint):
 x0=max(0,int(x0));x1=min(im.shape[1],int(x1));y0=max(0,int(y0));y1=min(im.shape[0],int(y1))
 if x1-x0<30 or y1-y0<4:return None
 gray=cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
 bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
 edges=cv2.Canny(gray,40,120);lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=max(15,int((x1-x0)*.12)),minLineLength=(x1-x0)*.45,maxLineGap=(x1-x0)*.035)
 if lines is None:return None
 candidates=[];middle=(x0+x1)/2
 for xx,yy,aa,bb in lines[:,0]:
  if abs(aa-xx)<1:continue
  slope=(bb-yy)/(aa-xx)
  if abs(slope)>math.tan(math.radians(12)) or abs(slope-slope_hint)>.09:continue
  intercept=y0+yy-slope*(x0+xx);center=slope*middle+intercept
  if abs(center-target)>(y1-y0)*.38:continue
  # A physical rule spans much of the crop; keep only one close to the semantic gap.
  candidates.append((abs(center-target),slope,intercept))
 if not candidates:return None
 _,slope,intercept=min(candidates);return slope,intercept-1

def refine_boundaries(image,result,pp_blocks,ocr_lines,*,experimental=False):
 # Experimental geometry adjustment caused regressions; disabled by default.
 if not experimental:return copy.deepcopy(result)
 out=copy.deepcopy(result);regions=out.get('regions',[])
 if len(regions)<2:return out
 # Intermediate totals can reopen a table; this separator model assumes a monotonic flow.
 if len(regions)>3:return out
 im=image if isinstance(image,np.ndarray) else cv2.imdecode(np.fromfile(str(image),np.uint8),cv2.IMREAD_COLOR)
 if im is None:return out
 byid={o['ocr_id']:o for o in ocr_lines};cache={}
 def box(ident):
  if ident not in cache:cache[ident]=ink_box(im,byid[ident]) if ident in byid else None
  return cache[ident]
 changes=[]
 for previous,current in zip(regions,regions[1:]):
  if len(previous['polygon'])!=4 or len(current['polygon'])!=4:continue
  prev_ids=list(previous.get('evidence_ocr_ids',[]));next_ids=list(current.get('evidence_ocr_ids',[]))
  before={i:box(i) for i in prev_ids if box(i)};after={i:box(i) for i in next_ids if box(i)}
  if not before or not after:continue
  heights=[b[3]-b[1] for b in list(before.values())+list(after.values())];h=max(4,float(np.median(heights)))
  moved=[]
  if previous['label']=='table' and current['label']=='footer':
   anchors=[(i,b) for i,b in after.items() if footer_reason(norm(byid[i]['text']))]
   if anchors:
    anchor=min(anchors,key=lambda pair:pair[1][1])[1]
    for ident,b in list(after.items()):
     if re.fullmatch(r'[\d\s.,]+',norm(byid[ident]['text'])) and (b[1]+b[3])/2<(anchor[1]+anchor[3])/2-h*.35:
      before[ident]=b;after.pop(ident);moved.append(ident)
  # Geometry overlap alone is not proof of clipped text. Preserve the base
  # quadrilateral unless semantic evidence requires moving a numeric item price.
  if not moved:continue
  if not after:continue
  prev_bottom=max(b[3] for b in before.values());next_top=min(b[1] for b in after.values())
  oldprev=bounds(quad(previous));oldnext=bounds(quad(current))
  if oldprev[3]+2<oldnext[1] and not moved:continue
  previous_window=1.5 if previous['label']=='table' and current['label']=='footer' else .4
  next_window=2.5 if previous['label']=='table' and current['label']=='footer' else 1.1
  before={i:b for i,b in before.items() if b[3]>=prev_bottom-h*previous_window};after={i:b for i,b in after.items() if b[1]<=next_top+h*next_window}
  x0=max(min(b[0] for b in before.values()),min(b[0] for b in after.values()));x1=min(max(b[2] for b in before.values()),max(b[2] for b in after.values()))
  if x1-x0<h*2:continue
  samples=[]
  for x in np.linspace(x0,x1,9):
   pb=min(before.values(),key=lambda b:max(b[0]-x,0,x-b[2]));nb=min(after.values(),key=lambda b:max(b[0]-x,0,x-b[2]))
   # At a shared column choose the last/first ink, not an earlier word on the same page.
   overlaps=[b for b in before.values() if b[0]-h*.3<=x<=b[2]+h*.3]
   if overlaps:pb=max(overlaps,key=lambda b:b[3])
   overlaps=[b for b in after.values() if b[0]-h*.3<=x<=b[2]+h*.3]
   if overlaps:nb=min(overlaps,key=lambda b:b[1])
   if pb[3]<=nb[1]+h*.2:samples.append((x,(pb[3]+nb[1])/2,pb[3],nb[1]))
  if len(samples)<4:continue
  slope,intercept=np.polyfit([p[0] for p in samples],[p[1] for p in samples],1)
  if abs(slope)>math.tan(math.radians(12)):continue
  safe=safe_separator(samples,slope,max(.5,h*.02))
  if safe is None:continue
  slope,intercept=safe
  target=slope*(x0+x1)/2+intercept
  rule=_rule_line(im,x0,x1,min(prev_bottom,next_top)-h*.35,max(prev_bottom,next_top)+h*.35,target,slope)
  if rule is not None:
   rs,rb=rule
   if all(rs*x+rb>=p+.5 and rs*x+rb<=n-.5 for x,_,p,n in samples):slope,intercept=rs,rb
  # Refuse a separator that visibly crosses the trimmed glyph support.
  if any(slope*x+intercept<p-h*.18 or slope*x+intercept>n+h*.18 for x,_,p,n in samples):continue
  pq=copy.deepcopy(previous['polygon']);nq=copy.deepcopy(current['polygon']);gap=max(1,h*.03)
  for index in [2,3]:pq[index][1]=float(np.clip(slope*pq[index][0]+intercept-gap/2,0,im.shape[0]))
  for index in [0,1]:nq[index][1]=float(np.clip(slope*nq[index][0]+intercept+gap/2,0,im.shape[0]))
  if min(pq[2][1],pq[3][1])<=max(pq[0][1],pq[1][1]) or max(nq[0][1],nq[1][1])>=min(nq[2][1],nq[3][1]):continue
  previous.update(polygon=pq,bbox=bounds(np.asarray(pq)));current.update(polygon=nq,bbox=bounds(np.asarray(nq)))
  for ident in moved:
   if ident not in previous['evidence_ocr_ids']:previous['evidence_ocr_ids'].append(ident)
   if ident in current['evidence_ocr_ids']:current['evidence_ocr_ids'].remove(ident)
  changes.append(dict(previous=previous['label'],next=current['label'],line=[float(slope),float(intercept)],rule_detected=rule is not None,reassigned_numeric_ocr_ids=moved,ink_samples=samples))
 if changes:out['boundary_refinement']=dict(version=VERSION,changes=changes);out['warnings']=sorted(set(out['warnings']+['IMAGE_INK_BOUNDARY_REFINED']))
 return out

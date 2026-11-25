"""Rules v0.1. Inputs are PP detections/OCR only; no sample IDs or gold labels."""
import math,re,unicodedata
from difflib import SequenceMatcher
import numpy as np
VERSION='pp-ocr-regions-0.8-summary-alias'
CONFIG={'ocr_min_confidence':.4,'max_skew_degrees':15,'row_overlap':.35,'padding_line_height':.22,'allow_multiple_regions_per_class':True}

def is_camera_watermark(text):
 t=norm(text)
 return bool(re.search(r'\bsamsung\s+(?:triple|quad|dual)\s+camera\b|\bchup bang\s+(?:galaxy|samsung|xiaomi|redmi)|\bshot on\s+(?:samsung|galaxy|redmi|xiaomi|iphone)',t))

def document_kind(lines,height):
 texts=[norm(o['text']) for o in lines]
 if any(re.search(r'\bgiao dich\b.*\btam ngung\b',t) for t in texts):return 'SUSPENDED_TRANSACTION_NO_ITEMS'
 brands=[i for i,t in enumerate(texts) if re.search(r'\b(?:vietcombank|vietinbank|bidv|techcombank|sacombank)\b',t)]
 goods=any(re.search(r'\b(?:ten hang|ten mon|ten san pham|descriptions?|qty)\b',t) for t in texts)
 tokens=set(re.findall(r'\b(?:visa|debit|merchant|napas|card|the|tid|aid|a1d|batch|expiry)\b',' '.join(texts)))
 if brands and not goods and len(tokens)>=2 and min(bounds(quad(lines[i]))[1] for i in brands)<height*.5:return 'CARD_PAYMENT_NO_ITEMS'
 return 'ITEMIZED_DOCUMENT_CANDIDATE'
def norm(text):
 text=text.lower().replace('đ','d');return re.sub(r'\s+',' ',''.join(c for c in unicodedata.normalize('NFD',text) if unicodedata.category(c)!='Mn')).strip()
def bounds(p):return [float(np.min(p[:,0])),float(np.min(p[:,1])),float(np.max(p[:,0])),float(np.max(p[:,1]))]
def rot(p,angle):
 c,s=math.cos(angle),math.sin(angle);return np.asarray(p)@np.array([[c,s],[-s,c]])
def quad(obj):
 if obj.get('polygon'):return np.asarray(obj['polygon'],float)
 x,y,a,b=obj['bbox'];return np.array([[x,y],[a,y],[a,b],[x,b]],float)
def column_hits(text):
 keys=[r'\bten (?:hang|san pham|sp|mon)\b',r'\bdescriptions?\b',r'\bqty\b|\bquantity\b',r'\bdvt\b|\bdon vi\b',r'\bdon gia\b|\bd\.?\s*gia\b|\b(?:unit )?price\b',r'\bthanh tien\b|\bt\.?\s*tien\b|\bamount\b',r'\bgia ban\b',r'\bsl\b|\bs l\b',r'\bcode\b|\bitem\b',r'\brm\b']
 hits=sum(bool(re.search(p,text)) for p in keys)
 if re.search(r'\bs\.?\s*luong\b',text):hits+=1
 if re.search(r'\bddon gia\b',text):hits+=1
 if re.search(r'\bthonh tien\b',text):hits+=1
 words=[w.replace('1','i').replace('0','o') for w in re.findall(r'[a-z01]+',text)]
 if not re.search(r'\bdescriptions?\b',text) and any(SequenceMatcher(None,w,'description').ratio()>=.74 for w in words if len(w)>=5):hits+=1
 if not re.search(r'\bamount\b',text) and any(SequenceMatcher(None,w,'amount').ratio()>=.74 for w in words if len(w)>=5):hits+=1
 if re.search(r'\bu\.?p\b|\bamt\b',text):hits+=1
 return hits
def is_meta(text):
 if re.search(r'\b[h#]?(?:sale|saie)\s*type\b',text):return True
 return bool(re.search(r'\b(?:invoice no|invoice number|date|cashier|phone|address|gst reg|company reg|tax invoice|hoa don|ngay lap|so hoa don|so hd|thu ngan|thoi gian|sale.?type|dia chi|nguoi mua|nguoi ban|mst|ma so thue|nhan vien|khach hang|gio vao|gio ra|dien thoai)\b',text))
def footer_reason(text):
 # The Reader can render the vowel in 'tong' as '&' and drop the vowel in 'so'.
 # Require an entire Vietnamese summary row with a numeric value.
 if re.fullmatch(r't&ng\s*s(?:o)?\s*:?\s*\d{1,3}',text):return 'aggregate_quantity'
 if re.fullmatch(r't&ng\s*cong\s*:?\s*(?:\d{1,3}(?:[., ]\d{3})+|\d+)',text):return 'total'
 # Recover short split/truncated Vietnamese aggregate labels, only before an amount.
 if re.search(r'\d',text):
  prefix=re.sub(r'[^a-z]','',re.split(r'\d',text,1)[0])
  for target,kind in [('tongtien','total'),('thanhtien','total'),('phuthu','payment'),('giamgia','payment')]:
   if len(prefix)>=5 and SequenceMatcher(None,prefix,target).ratio()>=.86:return kind
  if re.match(r'^tong\s+\d',text):return 'total'
 words=re.findall(r'[a-z01]+',text)
 if len(words)>=2 and SequenceMatcher(None,words[0].replace('1','l'),'jumlah').ratio()>=.75 and SequenceMatcher(None,words[1],'item').ratio()>=.75:return 'aggregate_quantity'
 # Correct common OCR substitutions only in lexical anchor tokens, preserve raw evidence.
 text=re.sub(r'\bt[o0][t1il][a@][l1i]\b','total',text)
 token=re.match(r'^([a-z]+)',text)
 if token and re.search(r'\d',text):
  word=token.group(1)
  if len(word)>=6 and SequenceMatcher(None,word,'subtotal').ratio()>=.75:text='subtotal'+text[len(word):]
 if is_meta(text) and ('tax invoice' in text or 'hoa don' in text):return None
 patterns=[('subtotal',r'^(?:\d+\s+)?(?:sub\s*total|sttl)\b'),('total',r'^(?:\d+\s+)?(?:grand\s*total|total(?:\s+excl|\s+incl|\s+and|\s+item)?|tong\s+(?:cong|thanh toan|tien|so)|cong\s+tien\s+hang|phai\s+thanh\s+toan|tien\s+thanh\s+toan)\b'),('tax',r'^(?:thue\s+suat|tien\s+thue|thue\s+gtgt|vat\s*[:0-9]|gst\s+summary|pajak|\d+\s*(?:%|x|z)\s*gst)\b'),('payment',r'^(?:cash\b|change\b|payment\b|paid\b|tien\s+(?:khach|thoi|thua|tra lai|mat)|khach\s+dua|khach\s+thanh toan|chiet\s+khau|discount\b|phi\s+giao\s+hang|no\s+cu\b|tong\s+no\b)'),('aggregate_quantity',r'^(?:qty|quantity)\s*:\s*\d')]
 for name,p in patterns:
  if re.search(p,text):return name
 return None

def build_regions(pp_blocks,ocr_lines,width,height):
 pp_blocks=[dict(p,_source_index=i) for i,p in enumerate(pp_blocks)]
 warnings=[];usable=[x for x in ocr_lines if x.get('confidence',0)>=CONFIG['ocr_min_confidence'] and norm(x.get('text','')) and len(x.get('polygon',[]))>=4]
 if not usable:return dict(rules_version=VERSION,regions=[],rows=[],warnings=['NO_USABLE_OCR'],needs_review=True)
 watermark=[o for o in usable if is_camera_watermark(o['text'])]
 excluded_artifacts=[]
 if watermark:
  cutoff=min(bounds(quad(o))[1] for o in watermark)
  wm=next(o for o in watermark if bounds(quad(o))[1]==cutoff);wb=bounds(quad(wm));wh=wb[3]-wb[1]
  # Camera badges immediately above the watermark text are also outside the receipt.
  badges=[p for p in pp_blocks if p.get('native_label') in ['image','footer'] and cutoff-wh*1.7<=p['bbox'][1]<cutoff and p['bbox'][2]-p['bbox'][0]<(wb[2]-wb[0])*.45 and min(p['bbox'][2],wb[2])-max(p['bbox'][0],wb[0])>0]
  if badges:cutoff=min(p['bbox'][1] for p in badges)
  excluded_artifacts=[o['ocr_id'] for o in usable if bounds(quad(o))[1]>=cutoff]
  usable=[o for o in usable if o['ocr_id'] not in excluded_artifacts]
  pp_blocks=[p for p in pp_blocks if (p['bbox'][1]+p['bbox'][3])/2<cutoff]
  warnings.append('CAMERA_WATERMARK_EXCLUDED')
 if not usable:return dict(rules_version=VERSION,regions=[],rows=[],warnings=warnings+['NO_DOCUMENT_EVIDENCE'],excluded_ocr_ids=excluded_artifacts,needs_review=True)
 kind=document_kind(usable,height)
 non_itemized=kind!='ITEMIZED_DOCUMENT_CANDIDATE'
 if kind=='CARD_PAYMENT_NO_ITEMS':
  brand_top=min(bounds(quad(o))[1] for o in usable if re.search(r'\b(?:vietcombank|vietinbank|bidv|techcombank|sacombank)\b',norm(o['text'])))
  median_height=float(np.median([bounds(quad(o))[3]-bounds(quad(o))[1] for o in usable]));floor=brand_top-median_height
  excluded_artifacts += [o['ocr_id'] for o in usable if bounds(quad(o))[3]<floor]
  usable=[o for o in usable if bounds(quad(o))[3]>=floor]
  pp_blocks=[p for p in pp_blocks if p['bbox'][3]>=floor]
 # A dominant table column provides a generic fence against another receipt in the background.
 table_blocks=[p for p in pp_blocks if p.get('native_label')=='table' and p['bbox'][2]-p['bbox'][0]>=width*.22]
 if table_blocks:
  anchor=max(table_blocks,key=lambda p:(p['bbox'][2]-p['bbox'][0])*(p['bbox'][3]-p['bbox'][1]));x,y,a,b=anchor['bbox'];pad=(a-x)*.18;left,right=x-pad,a+pad
  kept=[o for o in usable if left<=(bounds(quad(o))[0]+bounds(quad(o))[2])/2<=right]
  if kept and len(kept)<len(usable):
   usable=kept;warnings.append('OCR_OUTSIDE_PRIMARY_COLUMN_EXCLUDED')
   def in_column(p):
    x,y,a,b=p['bbox'];return max(0,min(a,right)-max(x,left))/max(1,min(a-x,right-left))>=.35
   pp_blocks=[p for p in pp_blocks if in_column(p)]
 angles=[]
 for line in usable:
  p=quad(line);dx,dy=p[1]-p[0]
  if abs(dx)>2:angles.append(math.atan2(dy,dx))
 angle=float(np.median(angles)) if angles else 0
 if abs(math.degrees(angle))>CONFIG['max_skew_degrees']:angle=0;warnings.append('SKEW_UNCERTAIN')
 transformed=[]
 for b in pp_blocks:
  label=b.get('native_label','').lower()
  if label not in ['table','text','doc_title','paragraph_title','header','footer','number','footnote','vision_footnote','chart']:continue
  transformed.append(dict(label=label,bbox=bounds(rot(quad(b),-angle))))
 hmed=float(np.median([bounds(rot(quad(x),-angle))[3]-bounds(rot(quad(x),-angle))[1] for x in usable]));hmed=max(hmed,2)
 lines=[];excluded=list(excluded_artifacts)
 # OCR in gaps between PP text blocks is allowed within the document's supported text band.
 supported_boxes=[]
 for line in usable:
  b=bounds(rot(quad(line),-angle));cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
  if any(x['bbox'][0]-hmed<=cx<=x['bbox'][2]+hmed and x['bbox'][1]-hmed<=cy<=x['bbox'][3]+hmed for x in transformed):supported_boxes.append(b)
 document_band=[min(b[0] for b in supported_boxes),min(b[1] for b in supported_boxes),max(b[2] for b in supported_boxes),max(b[3] for b in supported_boxes)] if supported_boxes else None
 for line in usable:
  box=bounds(rot(quad(line),-angle));cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2;pad=hmed*.8
  support=not transformed or any(b['bbox'][0]-pad<=cx<=b['bbox'][2]+pad and b['bbox'][1]-pad<=cy<=b['bbox'][3]+pad for b in transformed)
  if not support and document_band and document_band[0]-hmed<=cx<=document_band[2]+hmed and document_band[1]<=cy<=document_band[3]:
   support=True;warnings.append('OCR_IN_PP_GAP')
  if not support and non_itemized and document_band and document_band[0]-hmed<=cx<=document_band[2]+hmed and cy>=document_band[1]:
   support=True;warnings.append('NON_ITEMIZED_FOOTNOTE_OCR_INCLUDED')
  if not support:excluded.append(line['ocr_id']);continue
  lines.append(dict(**line,box=box))
 if not lines:return dict(rules_version=VERSION,regions=[],rows=[],warnings=['NO_PP_SUPPORTED_OCR'],needs_review=True)
 rows=[]
 for line in sorted(lines,key=lambda x:(x['box'][1]+x['box'][3])/2):
  b=line['box'];cy=(b[1]+b[3])/2;matches=[]
  for i,row in enumerate(rows):
   a=row['bbox'];overlap=max(0,min(a[3],b[3])-max(a[1],b[1]));small=min(a[3]-a[1],b[3]-b[1])
   if overlap/max(small,1)>=CONFIG['row_overlap'] and abs((a[1]+a[3])/2-cy)<max(hmed*.7,small*.6):matches.append(i)
  if matches:
   row=rows[matches[-1]];row['lines'].append(line);row['bbox']=[min(row['bbox'][0],b[0]),min(row['bbox'][1],b[1]),max(row['bbox'][2],b[2]),max(row['bbox'][3],b[3])]
  else:rows.append(dict(lines=[line],bbox=list(b)))
 rows.sort(key=lambda x:(x['bbox'][1]+x['bbox'][3])/2)
 for row in rows:
  row['lines'].sort(key=lambda x:x['box'][0]);row['text']=' '.join(x['text'] for x in row['lines']);t=norm(row['text']);row['normalized_text']=t;row['footer_reason']=footer_reason(t);row['column_hits']=column_hits(t);row['is_meta']=is_meta(t);cy=(row['bbox'][1]+row['bbox'][3])/2;cx=(row['bbox'][0]+row['bbox'][2])/2
  if kind=='SUSPENDED_TRANSACTION_NO_ITEMS' and re.match(r'^(?:tri gia|thanh tien)\b',t):row['footer_reason']='total'
  if kind=='CARD_PAYMENT_NO_ITEMS' and re.match(r'^tong',t) and re.search(r'(?:[uv]nd|usd)\s*\d',t):row['footer_reason']='total'
  row['pp_table_support']=any(b['label']=='table' and b['bbox'][0]-hmed<=cx<=b['bbox'][2]+hmed and b['bbox'][1]-hmed<=cy<=b['bbox'][3]+hmed for b in transformed)
  nums=re.findall(r'\d[\d.,]*',t);row['item_evidence']=row['pp_table_support'] and len(nums)>=2 and bool(re.search(r'[a-z]{3}',t)) and not row['is_meta'] and not row['footer_reason'] and '%' not in t
 starts=[i for i,r in enumerate(rows) if r['column_hits']>=2 and not r['footer_reason'] and not r['is_meta']] if not non_itemized else []
 if starts:start=starts[0];start_reason='column_headers'
 else:
  starts=[i for i,r in enumerate(rows) if r['item_evidence']] if not non_itemized else [];start=starts[0] if starts else None;start_reason='item_with_pp_table_support' if starts else 'unresolved'
 if start is None and not non_itemized:
  # Layout detectors can miss an item row or split its description/prices across rows.
  end=next((i for i,r in enumerate(rows) if r['footer_reason']),len(rows))
  previous_meta=max([i for i,r in enumerate(rows[:end]) if r['is_meta']],default=-1)
  left=min(r['bbox'][0] for r in rows);right=max(r['bbox'][2] for r in rows)
  def price_row(row):
   return any(re.fullmatch(r'[\d\s.,]+',norm(l['text'])) and l['box'][0]>=left+.55*(right-left) and re.search(r'\d',l['text']) for l in row['lines'])
  candidates=[]
  for i,row in enumerate(rows[:end]):
   if i<=previous_meta or row['is_meta'] or row['footer_reason'] or not re.search(r'[a-z]{3}',row['normalized_text']):continue
   inline=price_row(row) and (row['pp_table_support'] or i>previous_meta>=0 or len(re.findall(r'\d[\d.,]*',row['normalized_text']))>=2)
   following=any(price_row(rows[j]) and not re.search(r'[a-z]',rows[j]['normalized_text']) and all(not re.search(r'[a-z]',rows[k]['normalized_text']) for k in range(i+1,j)) and not rows[j]['is_meta'] and not rows[j]['footer_reason'] and rows[j]['bbox'][1]-row['bbox'][3]<=4*hmed for j in range(i+1,min(end,i+3)))
   if inline or following:candidates.append(i)
  if candidates:start=candidates[0];start_reason='description_and_right_aligned_prices';warnings.append('TABLE_START_INFERRED_FROM_OCR_PRICE_ROWS')
 if start is None:warnings.append('NON_ITEMIZED_DOCUMENT_NO_TABLE' if non_itemized else 'TABLE_START_UNRESOLVED')
 state='header';footer_started=False;resumable=False
 for i,row in enumerate(rows):
  if start is None:
   if row['footer_reason']:state='footer';footer_started=True
   row['label']=state;row['decision']='no_table_start_'+state;continue
  if i<start:row['label']='header';row['decision']='before_table';continue
  if i==start:state='table'
  # A per-item VAT rate/base/price row is part of the item ledger. A tax summary
  # following subtotal/total remains Footer; do not let one VAT row swallow later items.
  if state=='table' and not footer_started and row['footer_reason']=='tax' and re.match(r"^vat\s*\d+(?:[.,]\d+)?\s*['’]?\s*[%x]",row['normalized_text']) and len(re.findall(r'\d[\d.,]*',row['normalized_text']))>=3:
   row['footer_reason']=None;warnings.append('ITEM_LEVEL_VAT_ROW_IN_TABLE')
  if row['footer_reason']:
   state='footer';footer_started=True;resumable=row['footer_reason']=='subtotal' or bool(re.search(r'\btotal\b.*\bsuppl',row['normalized_text']));row['decision']='footer_anchor:'+row['footer_reason']
  elif state=='footer' and resumable and row['item_evidence']:
   state='table';row['decision']='resume_item_after_aggregate';warnings.append('INTERLEAVED_AGGREGATE_AND_ITEMS')
  else:row['decision']='continue_'+state
  row['label']=state
 if not footer_started:warnings.append('NO_EXPLICIT_FOOTER_ANCHOR')
 groups=[]
 for i,row in enumerate(rows):
  if not groups or groups[-1]['label']!=row['label']:groups.append(dict(label=row['label'],indices=[]))
  groups[-1]['indices'].append(i)
 regions=[]
 doc_bounds=[min(r['bbox'][0] for r in rows),min(r['bbox'][1] for r in rows),max(r['bbox'][2] for r in rows),max(r['bbox'][3] for r in rows)]
 doc_area=max(1,(doc_bounds[2]-doc_bounds[0])*(doc_bounds[3]-doc_bounds[1]))
 for j,g in enumerate(groups):
  included=[rows[i] for i in g['indices']];b=[min(r['bbox'][0] for r in included),min(r['bbox'][1] for r in included),max(r['bbox'][2] for r in included),max(r['bbox'][3] for r in included)];attachments=[]
  # OCR may entirely miss logos/title/address/QR captions. Preserve independent PP evidence
  # within the semantic band's row boundaries instead of making OCR its only geometry source.
  before_y=rows[g['indices'][0]-1]['bbox'][3] if g['indices'][0]>0 else 0
  after_y=rows[g['indices'][-1]+1]['bbox'][1] if g['indices'][-1]<len(rows)-1 else float('inf')
  for index,block in enumerate(pp_blocks):
   label=block.get('native_label');bb=bounds(rot(quad(block),-angle));cx=(bb[0]+bb[2])/2
   semantic=(g['label']=='header' and label in ['header','doc_title','paragraph_title','text']) or (g['label']=='footer' and label in ['text','footer','footnote','vision_footnote'])
   if semantic and bb[1]>=before_y and bb[3]<=after_y and doc_bounds[0]-2*hmed<=cx<=doc_bounds[2]+2*hmed:
    b=[min(b[0],bb[0]),min(b[1],bb[1]),max(b[2],bb[2]),max(b[3],bb[3])];attachments.append(block.get('_source_index',index))
  for index,block in enumerate(pp_blocks):
   if block.get('native_label')!='image' and not (kind=='CARD_PAYMENT_NO_ITEMS' and block.get('native_label')=='chart'):continue
   bb=bounds(rot(quad(block),-angle));cx,cy=(bb[0]+bb[2])/2,(bb[1]+bb[3])/2;area=max(0,bb[2]-bb[0])*max(0,bb[3]-bb[1])
   terminal_footer=g['label']=='footer' and j==len(groups)-1
   if 0<area<.2*doc_area and b[1]-hmed<=cy<=(float('inf') if terminal_footer else b[3]+hmed) and doc_bounds[0]-2*hmed<=cx<=doc_bounds[2]+2*hmed:
    before_y=rows[g['indices'][0]-1]['bbox'][3] if g['indices'][0]>0 else -float('inf')
    after_y=rows[g['indices'][-1]+1]['bbox'][1] if g['indices'][-1]<len(rows)-1 else float('inf')
    if bb[1]>=before_y and bb[3]<=after_y:
     b=[min(b[0],bb[0]),min(b[1],bb[1]),max(b[2],bb[2]),max(b[3],bb[3])];attachments.append(block.get('_source_index',index))
    else:warnings.append('NON_TEXT_ATTACHMENT_CROSSES_BOUNDARY')
  padding=hmed*CONFIG['padding_line_height'];b=[b[0]-padding,b[1]-padding,b[2]+padding,b[3]+padding]
  first,last=g['indices'][0],g['indices'][-1]
  if first>0:
   previous=rows[first-1]['bbox'][3];current=rows[first]['bbox'][1]
   if previous<=current:b[1]=max(b[1],(previous+current)/2)
   else:warnings.append('BOUNDARY_OCR_OVERLAP')
  if last<len(rows)-1:
   current=rows[last]['bbox'][3];nxt=rows[last+1]['bbox'][1]
   if current<=nxt:b[3]=min(b[3],(current+nxt)/2)
   else:warnings.append('BOUNDARY_OCR_OVERLAP')
  poly=rot(np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]),angle);poly[:,0]=poly[:,0].clip(0,width);poly[:,1]=poly[:,1].clip(0,height)
  if b[2]<=b[0] or b[3]<=b[1]:continue
  regions.append(dict(region_id='region-'+str(j+1),label=g['label'],geometry='polygon',polygon=poly.tolist(),bbox=bounds(poly),evidence_ocr_ids=[l['ocr_id'] for r in included for l in r['lines']],decision_rows=g['indices'],attached_pp_image_indices=attachments))
 for missing in ['header','table','footer']:
  if missing=='table' and non_itemized:continue
  if not any(r['label']==missing for r in regions):warnings.append('NO_'+missing.upper()+'_EVIDENCE')
 for row in rows:row['ocr_ids']=[x['ocr_id'] for x in row.pop('lines')]
 return dict(rules_version=VERSION,config=CONFIG,document_kind=kind,skew_degrees=math.degrees(angle),regions=regions,rows=rows,excluded_ocr_ids=excluded,table_start_reason=start_reason,warnings=sorted(set(warnings)),needs_review=True,provenance='PP predictions + OCR only; no gold/sample-specific coordinates')

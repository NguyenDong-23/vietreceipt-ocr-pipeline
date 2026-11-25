"""One explicit image frame for PP, OCR and previews; reversible EXIF geometry."""
from pathlib import Path
from dataclasses import dataclass
import hashlib,copy,io
import numpy as np
from PIL import Image,ImageOps

def exif_matrix(orientation,width,height):
 matrices={1:[[1,0,0],[0,1,0],[0,0,1]],2:[[-1,0,width],[0,1,0],[0,0,1]],3:[[-1,0,width],[0,-1,height],[0,0,1]],4:[[1,0,0],[0,-1,height],[0,0,1]],5:[[0,1,0],[1,0,0],[0,0,1]],6:[[0,-1,height],[1,0,0],[0,0,1]],7:[[0,-1,height],[-1,0,width],[0,0,1]],8:[[0,1,0],[-1,0,width],[0,0,1]]}
 if orientation not in matrices:raise ValueError(f'Unsupported EXIF orientation: {orientation}')
 return np.asarray(matrices[orientation],dtype=float)

def transform_polygon(polygon,matrix):
 p=np.asarray(polygon,dtype=float)
 if p.ndim!=2 or p.shape[1]!=2 or len(p)<3:raise ValueError('Invalid polygon')
 mapped=np.column_stack([p,np.ones(len(p))])@np.asarray(matrix,dtype=float).T
 return (mapped[:,:2]/mapped[:,2,None]).tolist()

def bounds(polygon):return [min(p[0] for p in polygon),min(p[1] for p in polygon),max(p[0] for p in polygon),max(p[1] for p in polygon)]

@dataclass
class PreparedImage:
 image: Image.Image
 source: dict
 def bgr(self):return np.ascontiguousarray(np.asarray(self.image)[:,:,::-1])
 def regions_in_source_frame(self,regions):
  result=[]
  for region in regions:
   poly=transform_polygon(region['polygon'],self.source['upright_to_original_matrix'])
   entry={k:copy.deepcopy(region[k]) for k in ['region_id','label','content_status','role_status','evidence_ocr_ids'] if k in region}
   entry.update(geometry='polygon',polygon=poly,bbox=bounds(poly),coordinate_system='pixel_xy_on_original_encoded_image');result.append(entry)
  return result

def prepare_image(path):
 path=Path(path).resolve();payload=path.read_bytes()
 with Image.open(io.BytesIO(payload)) as raw:
  if getattr(raw,'n_frames',1)!=1:raise ValueError('Multi-page/animated input: export each page as an image first')
  orientation=int(raw.getexif().get(274,1));width,height=raw.size;forward=exif_matrix(orientation,width,height)
  rgb=ImageOps.exif_transpose(raw).convert('RGB');upright=Image.frombytes('RGB',rgb.size,rgb.tobytes())
 expected=(height,width) if orientation>=5 else (width,height);assert upright.size==expected
 source=dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),encoded_width=width,encoded_height=height,exif_orientation=orientation,upright_width=upright.width,upright_height=upright.height,original_to_upright_matrix=forward.tolist(),upright_to_original_matrix=np.linalg.inv(forward).tolist(),geometry_convention='Continuous pixel edges, x in [0,width], y in [0,height]; use pixel centers for raster points')
 return PreparedImage(upright,source)

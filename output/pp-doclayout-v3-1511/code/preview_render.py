"""Antialiased preview only; never changes prediction coordinates."""
from PIL import Image,ImageDraw,ImageFont

def render_preview(image,regions,size=(1000,1500)):
 view=image.convert('RGB').copy();view.thumbnail(size)
 factor=3
 overlay=Image.new('RGBA',(view.width*factor,view.height*factor));d=ImageDraw.Draw(overlay)
 try:
  font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18*factor)
 except OSError:
  font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18*factor)
 for region in regions:
  points=[(x*view.width/image.width*factor,y*view.height/image.height*factor) for x,y in region['polygon']]
  color={'header':'#006bd6','table':'#bd8300','footer':'#dd147b'}[region['label']]
  d.line(points+[points[0]],fill=color,width=2*factor,joint='curve')
  # Stable top-left anchor, not the highest vertex (which may be on the right).
  x,y=points[0];d.text((x,max(0,y-21*factor)),region['label'],font=font,fill=color,stroke_width=factor,stroke_fill='white')
 view=Image.alpha_composite(view.convert('RGBA'),overlay.resize(view.size,Image.Resampling.LANCZOS))
 return view.convert('RGB')

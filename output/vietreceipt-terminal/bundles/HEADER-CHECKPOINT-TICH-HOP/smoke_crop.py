"""Load the actual checkpoint and recognize one user-supplied line crop."""
import argparse
import json
from PIL import Image
from checkpoint_ocr import CheckpointRecognizer

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--crop',required=True)
args=parser.parse_args()
recognizer=CheckpointRecognizer()
with Image.open(args.crop) as image:
    result=recognizer.recognize_crop(image)
print(json.dumps({'checkpoint_sha256':recognizer.sha256,'result':result},ensure_ascii=False))

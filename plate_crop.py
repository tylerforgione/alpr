#imports
import numpy as np
import ultralytics
from pathlib import Path
import cv2

# config
MODEL_PATH = '/Users/tylerforgione/alpr/best.pt'
IMAGE_PATH = Path('alpr/License Plate Recognition/valid/images')
OUTPUT_PATH = Path('alpr/ocr/valid/images')
CONF = 0.05
PADDING = 0.08

OUTPUT_PATH.mkdir
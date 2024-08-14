# Import Dependencies
import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# The goal of this script is to run OCR on the regions of the
# main design where the specs for the machines are

# Some parameters
EPS = 50 # used for clustering distance
minSameples = 1 # Minimum samples needed for group formation
UseGPU = True # Do you (Can you) use a GPU? Runs faster

ocr = easyocr.Reader(['en'], gpu=UseGPU)

def ServerSpecOCR(imgPath=None):
    if imgPath == None:
        # no image so give up
        return None
    
    # Read the image
    image = cv2.imread(imgPath)

    # Run ocr
    baseResult = ocr.readtext(imgPath)
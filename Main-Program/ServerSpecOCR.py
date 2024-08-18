# Import Dependencies
import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import os

# The goal of this script is to run OCR on the regions of the
# main design where the specs for the machines are

# Some parameters
EPS = 150 # used for clustering distance
minSamples = 1 # Minimum samples needed for group formation
UseGPU = True # Do you (Can you) use a GPU? Runs faster

ocr = easyocr.Reader(['en'], gpu=UseGPU)

currentDir= os.path.dirname(__file__)

def ServerSpecOCR(imgPath=None, debug=False):
    if imgPath == None:
        # no image so give up
        return None
    
    imgPath = os.path.join(currentDir, imgPath)
    enhancedPath = os.path.join(currentDir, "temp_stuff", "ENHANCED.jpeg")
    
    # Read the image
    image = cv2.imread(imgPath)

    # Convert to grayscale
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast
    enhanced_img = cv2.addWeighted(grayscaleImage, 1.5, np.zeros_like(grayscaleImage), 0, -75)

    # blur
    enhanced_img = cv2.blur(enhanced_img,(2,2))

    # save the enhanced image
    print(enhancedPath)
    print(cv2.imwrite(enhancedPath, enhanced_img))

    # Run ocr
    baseResult = ocr.readtext(enhancedPath, mag_ratio=2)
    if debug:
        print(baseResult)

    # Extract the centers of bounding boxes
    coordinates = []
    for (boundBox, text, prob) in baseResult:
        (top_left, top_right, bottom_right, bottom_left) = boundBox
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        coordinates.append([center_x, center_y])

    # Convert to numpy array
    coordinates = np.array(coordinates)

    # DBSCAN clustering
    clustering = DBSCAN(eps=EPS, min_samples=minSamples).fit(coordinates)

    # Group the text by the clusters
    clustered_text = {}
    for label, (bbox, text, prob) in zip(clustering.labels_, baseResult):
        if label not in clustered_text:
            clustered_text[label] = []
        clustered_text[label].append((bbox, text, prob))

    # Preview OCR Groups

    # New preview of clusterd text

    # list to store the grouped text as objects
    grouped_text_objects = []

    for cluster in clustered_text.values():
      all_x = []
      all_y = []
      combined_text = ''
      for (bbox, text, prob) in cluster:
          (top_left, top_right, bottom_right, bottom_left) = bbox
          top_left = tuple(map(int, top_left))
          bottom_right = tuple(map(int, bottom_right))
          #cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)
          #cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
          all_x.extend([top_left[0], bottom_right[0]])
          all_y.extend([top_left[1], bottom_right[1]])
          combined_text += ' ' + text

      # Store the combined text of the cluster
      grouped_text_objects.append(combined_text.strip())

      # Draw box around cluster
      min_x, max_x = min(all_x), max(all_x)
      min_y, max_y = min(all_y), max(all_y)
      cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

    # Convert from BGR to RGB
    imgPreview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(grouped_text_objects)

    # Display the image
    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(imgPreview)
        plt.axis('off')
        plt.show()

    # return the groups text object
    return grouped_text_objects[-1]
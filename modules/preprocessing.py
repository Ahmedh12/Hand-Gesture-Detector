#This module is responsible for preprocessing the images before feature extraction

import numpy as np
import cv2


def preprocess_image(img):
    '''
    Binarize the image and fill the holes in the image in place
    @param img: the image to be processed
    @return: the processed image
    '''
    #convert to RGB2YCrCb
    for _ in range(5):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    #extract the Y channel
    img = img[:,:,0]

    #binarize the image
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #fill the holes in the image
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,3), np.uint8))

    return img




#This module is responsible for preprocessing the images before feature extraction

import numpy as np
import cv2

def preprocess_image(img):
    img_t = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,2]

    #binarize the image
    _, img_t = cv2.threshold(img_t, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #fill the holes in the image
    img_t = cv2.morphologyEx(img_t, cv2.MORPH_OPEN, np.ones((5,3), np.uint8))

    #find the contours in the image
    contours, _ = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #threshold the contours based on their area
    contours = [c for c in contours if cv2.contourArea(c) > 1500]

    #draw the contours on a blank image
    img_t = np.zeros_like(img_t)
    cv2.drawContours(img_t, contours, -1, 255, 3)

    return img_t


def preprocess_image_2(img):
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




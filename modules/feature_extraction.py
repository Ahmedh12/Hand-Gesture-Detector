#This module is responsible for extracting features from the binary images
import cv2
from sklearn.decomposition import PCA
import numpy as np
from modules.display_image import display_image
from skimage.feature import hog
import matplotlib.pyplot as plt


def extract_features(image):
    # Extract HOG features
    #HOG Descriptor Parameters:
    #1- Window Size
    #2- Block Size
    #3- Block Stride
    #4- cell Size
    #5- number of bins
    hog = cv2.HOGDescriptor((64,64) , (32,32) , (16,16) , (16,16) , 8 , 0 , 1)
    features = hog.compute(image)
    return features


def extract_features_skimage(image, orientation = 2, pixels_per_cell=(4, 2), cells_per_block=(3, 3)):
    # use hog from skimage
    # return the feature points only
    features = hog(image, orientations= orientation, pixels_per_cell= pixels_per_cell, 
                   cells_per_block= cells_per_block, visualize=False)
    return features

def visualize_HOG(image, orientation = 2, pixels_per_cell=(4, 2), cells_per_block=(3, 3)):
    # use hog from skimage
    _, hog_image = hog(image, orientations= orientation, pixels_per_cell= pixels_per_cell, 
                   cells_per_block= cells_per_block, visualize=True)
    display_image(hog_image)
    display_image(image)


#reduce the feature vector using PCA
def reduce_features(features , n_components=0.95 , whiten=True):
    #reduce the feature vector using PCA
    #PCA Parameters:
    #1- number of components
    #2- whiten
    pca = PCA(n_components = n_components,whiten = whiten)
    reduced_features = pca.fit_transform(features)
    return pca , reduced_features

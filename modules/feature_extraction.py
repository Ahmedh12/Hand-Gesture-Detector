#This module is responsible for extracting features from the binary images
import cv2
from sklearn.decomposition import PCA
import numpy as np

#we are using the HOG feature extractor from Cv2 library

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

#reduce the feature vector using PCA
def reduce_features(features , n_components=0.95 , whiten=True):
    #reduce the feature vector using PCA
    #PCA Parameters:
    #1- number of components
    #2- whiten
    pca = PCA(n_components = n_components,whiten = whiten)
    reduced_features = pca.fit_transform(features)
    return pca , reduced_features
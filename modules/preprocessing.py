#This module is responsible:
#   1-loading the dataset
#   2-Preprocessing the images
#   3-Augmenting the dataset

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset(path):
    images = []
    labels = []
    print ("Loading dataset from " + path)
    for folder in os.listdir(os.path.dirname(__file__) + '\\..\\' + path):
        for cls in os.listdir(os.path.dirname(__file__) + '\\..\\' + path + '\\' + folder):
            for image in os.listdir(os.path.dirname(__file__) + '\\..\\' + path + '\\' + folder + '\\' + cls):
                try:
                    img = cv2.imread(path + '\\' + folder + '\\' + cls + '\\' + image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append((folder, cls))
                except Exception as e:
                    print('Error loading image' , image)
                    print(e)
    print ("Dataset loaded successfully") 
    return images, labels

# View random images from the dataset loaded
def view_random_images(images, labels, num_images=5):
    # View random images from the dataset in a grid
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    fig.tight_layout()
    for i in range(num_images):
        index = random.randint(0, len(images))
        axes[i].imshow(images[index], cmap='gray')
        axes[i].set_title(labels[index])
    plt.show()

# Augment the dataset
def augment_dataset(images, labels, num_images=5):
    augmented_images = []
    augmented_labels = []
    for _ in range(num_images):
        #select a random id for the transformation pipeline
        transformation_id = random.randint(0, 2)
        #select a random image from the dataset
        index = random.randint(0, len(images))
        #perofrm a series of random transformations on the image then add it to the dataset
        img = images[index]
        if transformation_id == 0:
            img = cv2.flip(img, 0)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            augmented_images.append(img)
        elif transformation_id == 1:
            img = cv2.flip(img, 1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            augmented_images.append(img)
        elif transformation_id == 2:
            img = cv2.flip(img, -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            augmented_images.append(img)
        
        augmented_labels.append(labels[index])
    return augmented_images, augmented_labels



#test the function
def main():
    images, labels = load_dataset('dataset')
    print("DataSet Size :" , len(images))
    assert len(images) == len(labels)
    view_random_images(images, labels)
    augmented_images, augmented_labels = augment_dataset(images, labels)
    view_random_images(augmented_images, augmented_labels)


if __name__ == '__main__':
    main()

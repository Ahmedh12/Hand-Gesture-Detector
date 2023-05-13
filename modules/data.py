# Description: This module contains functions to load and augment the dataset

import os
import random
import matplotlib.pyplot as plt
import cv2
from terminaltables import AsciiTable

def load_dataset(path):
    images = []
    labels = []
    print ("Loading dataset from " + path + " Folder...")
    abs_path = r"{}".format(os.path.dirname(__file__) + "\\..\\..\\" + path)
    for folder in os.listdir(abs_path):
        classes = r"{}".format(abs_path + "\\" + folder)
        for cls in os.listdir(classes):
            images_paths = r"{}".format(abs_path + '\\' + folder + '\\' + cls)
            for image_path in os.listdir(images_paths):
                try:
                    img_path = r"{}".format(abs_path + '\\' + folder + '\\' + cls + '\\' + image_path)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (461,260))
                    images.append(img)
                    labels.append((folder, cls))
                except Exception as e:
                    print('Error loading image' , img_path)
                    print(e)
    print ("Dataset loaded successfully") 
    return images, labels


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

# View random images from the dataset loaded
def view_random_images(images, labels, num_images=5):
    # View random images from the dataset in a grid
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    fig.tight_layout()
    for i in range(num_images):
        index = random.randint(0, len(images))
        axes[i].imshow(images[index], cmap='binary')
        axes[i].set_title(labels[index])
    plt.show()


# create a stratified test train val split
def create_stratified_split(images, labels , efc_features , test_size=0.2, val_size=0.2):
    #create a stratified split
    train_images = []
    train_efc_features = []
    train_labels = []
    test_images = []
    test_efc_features = []
    test_labels = []
    val_images = []
    val_efc_features = []
    val_labels = []
    for i in range(len(images)):
        if random.random() < test_size:
            test_images.append(images[i])
            test_efc_features.append(efc_features[i])
            test_labels.append(labels[i])
        elif random.random() < val_size:
            val_images.append(images[i])
            val_efc_features.append(efc_features[i])
            val_labels.append(labels[i])
        else:
            train_images.append(images[i])
            train_efc_features.append(efc_features[i])
            train_labels.append(labels[i])



    return train_images, train_efc_features, train_labels, test_images, test_efc_features, test_labels, val_images, val_efc_features, val_labels


# get the split stats
def get_stats(labels):
    m0,m1,m2,m3,m4,m5 = 0,0,0,0,0,0
    w0,w1,w2,w3,w4,w5 = 0,0,0,0,0,0
    for i in range(len(labels)):
        if(labels[i][0] == 'men'):
            if(labels[i][1] == '0'):
                m0 += 1
            elif(labels[i][1] == '1'):
                m1 += 1
            elif(labels[i][1] == '2'):
                m2 += 1
            elif(labels[i][1] == '3'):
                m3 += 1
            elif(labels[i][1] == '4'):
                m4 += 1
            elif(labels[i][1] == '5'):
                m5 += 1
        elif(labels[i][0] == 'Women'):
            if(labels[i][1] == '0'):
                w0 += 1
            elif(labels[i][1] == '1'):
                w1 += 1
            elif(labels[i][1] == '2'):
                w2 += 1
            elif(labels[i][1] == '3'):
                w3 += 1
            elif(labels[i][1] == '4'):
                w4 += 1
            elif(labels[i][1] == '5'):
                w5 += 1
    return m0,m1,m2,m3,m4,m5,w0,w1,w2,w3,w4,w5

def print_stat_table(split_name, m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5):
    total = m0+m1+m2+m3+m4+m5+w0+w1+w2+w3+w4+w5
    table_data=[
        ['class_count','Men','Women' , 'Total' , 'percentage'],
        ['0',m0,w0 , m0+w0 , round((m0+w0)/total,2)],
        ['1',m1,w1 , m1+w1 , round((m1+w1)/total,2)],
        ['2',m2,w2 , m2+w2 , round((m2+w2)/total,2)],
        ['3',m3,w3 , m3+w3 , round((m3+w3)/total,2)],
        ['4',m4,w4 , m4+w4 , round((m4+w4)/total,2)],
        ['5',m5,w5 , m5+w5 , round((m5+w5)/total,2)],
    ]
    table_obj = AsciiTable(table_data,split_name)
    table_obj.inner_heading_row_border = True
    table = table_obj.table 
    print(table)

def get_split_stats(train_labels, test_labels, val_labels):
    m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5 = get_stats(train_labels)
    print_stat_table("Train Set",m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5)
    m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5 = get_stats(test_labels)
    print_stat_table("Test Set",m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5)
    m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5 = get_stats(val_labels)
    print_stat_table("Val Set",m0, m1, m2, m3, m4, m5, w0, w1, w2, w3, w4, w5)

#test the functions
def main():
    images, labels = load_dataset('dataset')
    print("DataSet Size :" , len(images))
    assert len(images) == len(labels)
    view_random_images(images, labels)
    augmented_images, augmented_labels = augment_dataset(images, labels)
    view_random_images(augmented_images, augmented_labels)


if __name__ == '__main__':
    main()
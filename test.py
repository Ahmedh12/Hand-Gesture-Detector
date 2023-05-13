import os 
import cv2

##########################LOAD DATASET#####################################
def load_dataset(path):
    images = []
    labels = []
    for cls in os.listdir(path):
        images_paths = r"{}".format(path +'\\' + cls)
        for image_path in os.listdir(images_paths):
            try:
                img_path = r"{}".format(path + '\\' + cls + '\\' + image_path)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (461,260))
                images.append(img)
                labels.append(int(cls))
            except Exception as e:
                print('Error loading image' , img_path)
                print(e)
    print ("Dataset loaded successfully") 
    return images, labels


print("Loading dataset...")
images, labels = load_dataset(r"C:\Users\ahmed\Desktop\University Courses Material\pattern\project\testset\testset2")
print("Original Image Dimensions: ", images[0].shape)

##########################PREPROCESSING#####################################

from modules.preprocessing import preprocess_image

print("Preprocessing...")
for i in range(len(images)):
    images[i] , _ = preprocess_image(images[i])

print("Preprocessed Image Dimensions: ", images[0].shape)

##########################FEATURE EXTRACTION#####################################

from modules.feature_extraction import extract_features_skimage
import numpy as np

print("Extracting features...")
features = np.array([])
for image in images:
    if features.size == 0:
        features = extract_features_skimage(image)
    else:
        features = np.vstack((features, extract_features_skimage(image)))

print("Original Features Dimensions: ", features.shape)

##########################Feature Reduction#####################################

from pickle import load

print("Feature Reduction...")
pca = load(open(r"C:\Users\ahmed\Desktop\University Courses Material\pattern\project\Hand-Gesture-Detector\models\SVM_0.69\pca4_n_components_0.5.sav", 'rb'))
features = pca.transform(features)

print("Reduced Features Dimensions: ", features.shape)

##########################CLASSIFICATION#####################################

print("Classification...")
SVM_model = load(open(r"C:\Users\ahmed\Desktop\University Courses Material\pattern\project\Hand-Gesture-Detector\models\SVM_0.69\svm_model4_best_acc_0.69.sav", 'rb'))

predictions = SVM_model.predict(features)
##########################Result Reporting#####################################
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report

print("Accuracy:\n", accuracy_score(labels, predictions) , "\n\n")
print("Confusion Matrix:\n", confusion_matrix(labels, predictions) , "\n\n")
print("Classification Report:\n", classification_report(labels, predictions) , "\n\n")

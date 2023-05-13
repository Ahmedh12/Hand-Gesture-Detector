import time
import cv2
from modules.preprocessing import preprocess_image
from modules.feature_extraction import extract_features_skimage, reduce_features


def load_image(image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (461,260))
    return image


def test_func(folder_name, test_num, model=None, pca=None):

    results_txt = open("out/results.txt", "w")
    time_txt = open("out/time.txt", "w")

    for index in range(1, test_num + 1):

        image_name = folder_name + "/" + str(index) + '.png'
        img = load_image(image_name)

        start_time = time.time()

        img, _ = preprocess_image(img)
        features = extract_features_skimage(img)
        reduced_features = pca.transform(features.reshape(1, -1))
        prediction = model.predict(reduced_features)

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)

        results_txt.write(str(prediction[0]))
        time_txt.write(str(elapsed_time))

        if index != test_num:
            results_txt.write("\n")
            time_txt.write("\n")

    results_txt.close()
    time_txt.close()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import cv2
from PIL import Image
from collections import Counter
from sklearn import svm
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

"""
Extracting features from images and classification them

#### This script made by Shir Shabat ####
"""

def palette_perc(k_cluster):
    """
    The function returns the main colors in the image.
    And also takes into account their quantity.
    :param k_cluster: clusters
    :return: main colors in image
    """
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = dict(sorted(perc.items()))
    step = 0
    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)
    return palette


def histo_according_to_channels(image):
    """
    This function performs a histogram on each channel.
    :param image:
    :return:
    """
    dataFrame = pd.DataFrame()
    histogram_red, bin_edges = np.histogram(image[:, :, 0], bins=256)  # red
    dataFrame['red_channel'] = histogram_red

    histogram_green, bin_edges = np.histogram(image[:, :, 1], bins=256)  # green
    dataFrame['green_channel'] = histogram_green

    histogram_blue, bin_edges = np.histogram(image[:, :, 2], bins=256)  # blue
    dataFrame['blue_channel'] = histogram_blue
    return dataFrame


def data_frame_features(data_frame, is_pattern):
    """
    This function extract features using pandas package.
    Using median inorder to bring out the most common color
    :param data_frame: histogram of the channels - RGB.
    :param is_pattern: the colors are pattern
    :return: features
    """
    data_frame = data_frame.cumsum()
    data_frame = data_frame.apply(lambda x: x / x.max(), axis=0)  # normalized

    med_axis_1 = []
    if is_pattern:
        med_axis_1 = data_frame.median(axis=1).values.tolist()
    med_axis_0 = data_frame.median(axis=0).values.tolist()
    mean = data_frame.mean(axis=0).values.tolist()
    res = med_axis_0 + mean + med_axis_1
    return res


def extract_features(image):
    """
    This function extract features from image.
    At first step, we resize the image to fit the other images.
    At second step, we extract the features.
    :param image: image of one of the following categories: forest, sun or sky.
    :return: features
    """
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    clt = KMeans(n_clusters=7)
    clt_1 = clt.fit(image.reshape(-1, 3))

    df1 = histo_according_to_channels(image)
    df2 = histo_according_to_channels(palette_perc(clt_1))

    res1 = data_frame_features(df1, False)
    res2 = data_frame_features(df2, True)
    return res1 + res2


def load_data(type):
    path = "images\\" + type + "\data"
    return ["images\\" + type + "\data\\" + f for f in listdir(path) if isfile(join(path, f))]


def getY(forest, sky, sun):
    y_1 = [0] * len(forest)
    y_2 = [1] * len(sky)
    y_3 = [2] * len(sun)
    y = y_1 + y_2 + y_3
    return y


def getX(area1, area2, area3):
    x = np.concatenate((area1, area2, area3))
    return x


def get_x_y(lst):
    test_1, test_2, test_3 = lst
    x = getX(test_1, test_2, test_3)
    y = getY(test_1, test_2, test_3)
    return x, y


def linearSVC(x_train, y_train, x_test, y_test):
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("y_pred: " + str(y_pred) + " and y_test: " + str(y_test))
    print("x_test: " + str(x_test))
    print("Linear SVC Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("linearSVC confusion: ")

    metrics.plot_confusion_matrix(clf, x_test, y_test)
    plt.title("LinearSVC algorithm")
    plt.show()
    modelPickle = open('svc_pickle_file', 'wb')
    pickle.dump(clf, modelPickle)


def create_train_test():
    train_set, test_set = [], []
    forest_path_images, sun_path_images, sky_path_images = load_data("forest"), load_data("sun"), load_data("sky")

    train_forest, test_forest = train_test_split(forest_path_images, random_state=True)
    train_set.append(train_forest), test_set.append(test_forest)

    train_sun, test_sun = train_test_split(sun_path_images, random_state=True)
    train_set.append(train_sun), test_set.append(test_sun)

    train_sky, test_sky = train_test_split(sky_path_images, random_state=True)
    train_set.append(train_sky), test_set.append(test_sky)

    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            train_set[i][j] = extract_features(train_set[i][j])
        # print("-----------------------------------------")
    for i in range(len(test_set)):
        for j in range(len(test_set[i])):
            test_set[i][j] = extract_features(test_set[i][j])
        # print("-----------------------------------------")

    train_data = get_x_y(train_set)
    test_data = get_x_y(test_set)
    return train_data, test_data


def resize_image(image_path):
    image = Image.open(image_path)

    format = image_path[-3:]
    if format != "jpg":
        image = image.convert('RGB')
    image = image.resize((256, 256))
    return image


# if __name__ == '__main__':
#     train_data, test_data = create_train_test()
#     linearSVC(train_data[0], train_data[1], test_data[0], test_data[1])


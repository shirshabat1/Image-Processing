
import features as f
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import dataframe_image as dfi
import pandas as pd
"""
Instructions- 
Please enter a valid image path to the working directory and then press the run button.

#### This script made by Shir Shabat ####
"""

def predict_image(image_path):
    x_test = [f.extract_features(image_path)]
    loaded_model = pickle.load(open('svc_pickle_file', 'rb'))
    result = loaded_model.predict(x_test)
    if result == 0:
        print("The picture is probably a forest")
    elif result == 1:
        print("The picture is probably a sun")
    else:
        print("The picture is probably a sky")

def create_features_for_unseen_data(forest_path_images, sun_path_images, sky_path_images):
    validation = []

    validation.append(forest_path_images)
    validation.append(sun_path_images)
    validation.append(sky_path_images)
    for i in range(len(validation)):
        for j in range(len(validation[i])):
            validation[i][j] = f.extract_features(validation[i][j])
    validation_data = f.get_x_y(validation)
    return validation_data


def load_data(type):
    path = "images\\tests\\" + type
    return ["images\\tests\\" + type + "\\" + f for f in listdir(path) if isfile(join(path, f))]


def show_performance():
    forest_path_images, sun_path_images, sky_path_images = load_data("forest"), load_data("sun"), load_data("sky")
    x_validation, y_validation = create_features_for_unseen_data(forest_path_images, sun_path_images, sky_path_images)

    clf = pickle.load(open('svc_pickle_file', 'rb'))
    metrics.plot_confusion_matrix(clf, x_validation, y_validation, normalize='true', values_format='.0%', display_labels=["forest", "sun", "sky"])
    plt.title("LinearSVC algorithm")
    x_labels = clf.predict(x_validation)
    data = {'y_Actual': y_validation, 'y_Predicted': x_labels}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

    df['y_Actual'] = df['y_Actual'].map({0 : 'Forest', 1: 'Sun', 2: 'Sky'})
    df['y_Predicted'] = df['y_Predicted'].map({0 : 'Forest', 1: 'Sun', 2: 'Sky'})
    dfi.export(df, 'dataframe.png')
    plt.show()







#     ########################### MY TESTS ############################
if __name__ == '__main__':
    #     # forest = 0
    #     # sun = 1
    #     # sky = 2
    show_performance()



# if __name__ == '__main__':
#     # Enter a image from one of the following categories: sun, sky or forest.
#     if len(sys.argv) < 2:
#         print("Please enter a valid path")
#     else:
#         predict_image(sys.argv[1])
#
#
from PIL import Image, ImageEnhance
import os.path
IMAGE_NUMBER = 10
"""
Adding images in different brightness tones, in order to increase the number of images.

#### This script made by Shir Shabat ####
"""

def adjust_image_brightness(im, name, im_number):
    """
    This function will give us a brightened and darkened image and then- save it.
    :param image: image
    """
    # image brightness enhancer
    enhancer = ImageEnhance.Brightness(im)

    # brightens the image
    im_output = enhancer.enhance(1.5)
    im_output.save("images/" + name + "/data/" + name + str(im_number) + "_brightened.jpg")

    # darkens the image
    im_output = enhancer.enhance(0.5)
    im_output.save("images/" + name + "/data/" + name + str(im_number) + "_darkened.jpg")



def resize_image(image_path, name):
    image = Image.open(image_path)
    format = image_path[-3:]
    if format != "jpg":
        image = image.convert('RGB')
    image = image.resize((256, 256))
    image.save(name + ".jpg")


def load_data():
    for i in range(10):
        m = i + 1
        forest_file = "forest" + str(m) + ".jpg"
        sun_file = "sun" + str(m) + ".jpg"
        sky_file = "sky" + str(m) + ".jpg"
        if os.path.isfile(forest_file):
            resize_image(forest_file, "forest" + str(m) + "_reshape")
        else:
            resize_image("forest" + str(m) + ".png", "forest" + str(m) + "_reshape")
        if os.path.isfile(sun_file):
            resize_image(sun_file, "sun" + str(m) + "_reshape")
        else:
            resize_image("sun" + str(m) + ".png", "sun" + str(m) + "_reshape")
        if os.path.isfile(sky_file):
            resize_image(sky_file, "sky" + str(m) + "_reshape")
        else:
            resize_image("sky" + str(m) + ".png", "sky" + str(m) + "_reshape")
    forest_path_images, sun_path_images, sky_path_images = [], [], []
    for i in range(10):
        m = i + 1
        forest_path_images.append("forest" + str(m) + "_reshape.jpg")
        sun_path_images.append("sun" + str(m) + "_reshape.jpg")
        sky_path_images.append("sky" + str(m) + "_reshape.jpg")
    return forest_path_images, sun_path_images, sky_path_images


def check_if_valid_path(type):
    for i in range(1, IMAGE_NUMBER + 1):
        cur_path = "images/" + type + "/original/" + type + str(i) + ".jpg"
        place_to_save = "images/" + type + "/data/" + type + str(i) + "_reshape"
        if os.path.isfile(cur_path):

            resize_image(cur_path, place_to_save)
        else:
            resize_image(type + str(i) + ".png", place_to_save)


def load_original_images(type, directory, type_of_image):
    arr = []
    for i in range(1, IMAGE_NUMBER + 1):
        arr.append("images/" + type + "/"+ directory+ "/" + type + str(i) + type_of_image + ".jpg")
    return arr



def create_more_data(type):
    directory = "reshape_image"

    images = load_original_images(type, directory, "_reshape")
    for i in range(1, len(images) + 1):
        im = Image.open(images[i-1])
        adjust_image_brightness(im, type, i)


if __name__ == '__main__':
    create_more_data("sky")
    create_more_data("sun")
    create_more_data("forest")

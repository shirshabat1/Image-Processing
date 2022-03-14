import re
import os, itertools, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.draw import line

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, \
    AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import color
from scipy import ndimage

HEIGHT = 0
WIDTH = 1
GRAY_SCALE = 1
NORMALIZED = 255
FILTER_SIZE = 3
MIN_SIGMA = 0
MAX_SIGMA = 0.2
PATCH_SIZE = 24
NUM_CHANNELS = 48
PATCH_SIZE_DEBLURRING = 16
NUM_CHANNELS_DEBLURRING = 32
NUM_CHANNELS_SUPER = 54
PATCH_SIZE_SUPER = 20
LIST_KERNEL = 7
denoise_num_res_blocks = 7
deblur_num_res_blocks = 7
super_resolution_num_res_blocks = 7


def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    height = crop_size[HEIGHT]
    width = crop_size[WIDTH]
    images = dict()
    # using generator as asked
    while True:
        source_batch = []
        target_batch = []
        choice = random.choices(filenames, k=batch_size)  # random choice for a list
        for i in range(batch_size):
            image = choice[i]
            if image not in images:
                images[image] = read_image(image, GRAY_SCALE)
            clean_image = images[image]
            patched_height = np.random.randint(0, np.shape(clean_image)[0] - height * 3)
            patched_width = np.random.randint(0, np.shape(clean_image)[1] - width * 3)
            im_clean_batch = clean_image[patched_height: patched_height + height * 3,
                             patched_width: patched_width + width * 3]
            image_corrupted = corruption_func(im_clean_batch)
            im_cor_batch = image_corrupted[height: (height * 2), width: (2 * width)] - 0.5
            im_clean_batch = im_clean_batch[height: (2 * height), width: (2 * width)] - 0.5
            source_batch.append(im_cor_batch)
            target_batch.append(im_clean_batch)
        yield np.array(source_batch).reshape((batch_size, height, width, 1)), \
              np.array(target_batch).reshape((batch_size, height, width, 1))


def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    step_1 = Conv2D(num_channels, (FILTER_SIZE, FILTER_SIZE), padding='same')(input_tensor)
    step_2 = Activation('relu')(step_1)
    step_3 = Conv2D(num_channels, (FILTER_SIZE, FILTER_SIZE), padding='same')(step_2)
    step_4 = Add()([input_tensor, step_3])
    final_output = Activation('relu')(step_4)
    return final_output


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    input = Input(shape=(height, width, 1))
    network = Conv2D(num_channels, (FILTER_SIZE, FILTER_SIZE), padding='same')(input)
    network = Activation('relu')(network)
    for i in range(num_res_blocks):
        network = resblock(network, num_channels)
    output = Conv2D(1, (FILTER_SIZE, FILTER_SIZE), padding='same')(network)
    network = Add()([input, output])
    network_model = Model(inputs=input, outputs=network)
    return network_model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    np.random.shuffle(images)
    div_images = np.round(0.8 * len(images)).astype(np.int64)
    training_set = images[:div_images]
    validation_set = images[div_images:]
    crop_size = (model.input_shape[1], model.input_shape[2])
    training = load_dataset(training_set, batch_size, corruption_func, crop_size=crop_size)
    validation = load_dataset(validation_set, batch_size, corruption_func, crop_size=crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    num_valid_samples = num_valid_samples // batch_size
    model.fit_generator(training, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=validation,
                        validation_steps=num_valid_samples, use_multiprocessing=True)


def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    hight = np.shape(corrupted_image)[0]
    width = np.shape(corrupted_image)[1]
    a = Input(shape=(hight, width, 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    shifted_corrupted = (corrupted_image.reshape(1, hight, width, 1)) - 0.5
    restored_im = new_model.predict(shifted_corrupted)[0][:, :, 0] + 0.5
    restored_im = np.clip(restored_im, 0, 1).astype(np.float64)
    return restored_im


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    var = np.random.uniform(min_sigma, max_sigma)
    gauss_noise = np.random.normal(0, var, np.shape(image))
    image_denoising = image + gauss_noise
    image_denoising = np.round(image_denoising * NORMALIZED) / NORMALIZED
    image_denoising = np.clip(image_denoising, 0, 1).astype(np.float64)
    return image_denoising


def learn_denoising_model(denoise_num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param denoise_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    images = images_for_denoising()

    def corruption_func(image):
        return add_gaussian_noise(image, MIN_SIGMA, MAX_SIGMA)

    if quick_mode:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    else:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 100, 100, 10, 1000
    model = build_nn_model(PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, denoise_num_res_blocks)
    train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    image_kernel = motion_blur_kernel(kernel_size, angle)
    res = convolve(image, image_kernel)
    return res


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    motion_blur = add_motion_blur(image, kernel_size, angle)
    motion_blur = np.round(motion_blur * NORMALIZED) / NORMALIZED
    return np.clip(motion_blur, 0, 1).astype(np.float64)


def learn_deblurring_model(deblur_num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param deblur_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """

    def corruption_func(im):
        return random_motion_blur(im, [LIST_KERNEL])

    images = images_for_deblurring()
    if quick_mode:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    else:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 100, 100, 10, 1000
    model = build_nn_model(PATCH_SIZE_DEBLURRING, PATCH_SIZE_DEBLURRING, NUM_CHANNELS_DEBLURRING, deblur_num_res_blocks)
    train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model


def super_resolution_corruption(image):
    """
    Perform the super resolution corruption
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    options = [2, 3, 4]
    size_zoom = np.random.choice(options)
    new_im = zoom(image, 1 / size_zoom)
    x = np.arange(0, np.shape(image)[0])
    y = np.arange(0, np.shape(image)[1])
    cols, rows = np.meshgrid(y, x)
    new_im = zoom(new_im, size_zoom)
    new_im = ndimage.map_coordinates(new_im, [rows, cols], order=1, prefilter=False)
    np.reshape(new_im, (np.shape(image)[0], np.shape(image)[1]))
    return new_im


def learn_super_resolution_model(super_resolution_num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param super_resolution_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """

    def corruption_func(im):
        return super_resolution_corruption(im)

    images = images_for_super_resolution()
    if quick_mode:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    else:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 7, 250, 7, 1000
    model = build_nn_model(PATCH_SIZE_SUPER, PATCH_SIZE_SUPER, NUM_CHANNELS_SUPER, super_resolution_num_res_blocks)
    train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model


def effect_of_depth():
    """
    Use the num_res_blocks parameter to test models of different depths for both of the above tasks.
    Produce a plot of the mean square error on the validation set with respect to the number of residual blocks (from 1 to 5),
    and save to the path "./depth_plot_denoise.png" and "./depth_plot_deblur.png"
    """
    lst_deblur = []
    lst_denoise = []
    colors = ['darkblue', 'lightblue', 'darkgreen', 'gold', 'orange']
    for i in range(5):
        model = learn_denoising_model(i + 1)
        lst_denoise.append(model.history.history['val_loss'])

    num_epochs = [i for i in range(1, len(lst_denoise[0]) + 1)]
    for i in range(5):
        plt.plot(num_epochs, lst_denoise[i], color=colors[i], label=str(i + 1) + " res blocks")
    plt.legend()
    plt.title("Denoising Model")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig("depth_plot_denoise.png")
    plt.close()
    for i in range(5):
        model = learn_deblurring_model(i + 1)
        lst_deblur.append(model.history.history['val_loss'])


    num_epochs = [i for i in range(1, len(lst_deblur[0]) + 1)]
    for i in range(5):
        plt.plot(num_epochs, lst_deblur[i], color=colors[i], label=str(i + 1) + " res blocks")
    plt.legend()
    plt.title("Deblurring Model")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig("depth_plot_deblur.png")
    plt.close()


def deep_prior_restore_image(corrupted_image):
    """
    Implementation of the paper on deep prior image restoration for images of size 64x64
    :param corrupted_image: the 64 by 64 corrupted image
    :return: a restored image
    """
    pass

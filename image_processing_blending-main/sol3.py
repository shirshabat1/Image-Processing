import imageio
import numpy as np
from imageio import imwrite
from scipy import ndimage
from scipy import signal
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

GRAY_SCALE = 1
NORMALIZED = 255.0
BASE = [1, 1]
SECOND_PIXEL_IN_ROW = 2
MIN_DIM = 16


def read_image(filename, representation):
    """
    Reading an image into a given representation
    :param filename:  the filename of an image on disk (could be grayscale or RGB)
    :param representation: - representation code, either 1 or 2
    :return: the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    image = imageio.imread(filename)
    image = image.astype(np.float64)
    image = image / NORMALIZED
    if representation == GRAY_SCALE:
        image = rgb2gray(image)
    return image


def reduce_conv(image, filter_conv):
    """
    this function reduces the image size by using convolution with filter
    :param image: a grayscale image with double values in [0, 1]
    :param filter_conv: gaussian filter
    :return: sub sample of every second pixel in every row of the image after the reduce
    """
    reduce_image = ndimage.filters.convolve(image, filter_conv)
    reduce_image = ndimage.filters.convolve(reduce_image, filter_conv.T)
    # sub sample
    reduce_image = reduce_image[::SECOND_PIXEL_IN_ROW, ::SECOND_PIXEL_IN_ROW]
    return reduce_image


def expand(image, filter_conv):
    """
    this function expands the image size by padding and after that  using convolution with filter
    :param image: a grayscale image with double values in [0, 1]
    :param filter_conv: gaussian filter
    :return: sub sample of every second pixel in every row of the image after the reduce
    """
    row_padded = 2 * np.shape(image)[0]
    col_padded = 2 * np.shape(image)[1]
    padded_image = np.zeros((row_padded, col_padded))
    padded_image[::SECOND_PIXEL_IN_ROW, ::SECOND_PIXEL_IN_ROW] = image
    filter_conv = 2 * filter_conv  # to maintain constant brightness
    padded_image = ndimage.filters.convolve(padded_image, filter_conv, mode='constant')
    padded_image = ndimage.filters.convolve(padded_image, filter_conv.T, mode='constant')
    return padded_image


def create_filter(filter_size):
    """
    this function create a gaussian filter according to the filter size
    :param filter_size: odd number
    :return:  a gaussian filter
    """
    filter_vec = BASE
    filter_size = max(2, filter_size)
    for i in range(filter_size - 2):
        filter_vec = signal.convolve(filter_vec, BASE)
    filter_vec = filter_vec / (np.sum(filter_vec))  # normalized
    return np.reshape(filter_vec, (1, filter_size))


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    this function constructs a Gaussian pyramid
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar)
    :return: resulting pyramid pyr, filter vec
    """
    pyr = [im]  # original Image
    filter_vec = create_filter(filter_size)
    im_reduce = im
    for i in range(max_levels - 1):
        im_reduce = reduce_conv(im_reduce, filter_vec)
        if np.shape(im_reduce)[0] < MIN_DIM or np.shape(im_reduce)[1] < MIN_DIM:
            break  # im is too small
        pyr.append(im_reduce)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    this function constructs a Laplacian pyramid
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar)
    :return: resulting pyramid pyr, filter vec
    """
    pyr = []
    pyrGaussian, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    # for i in range(max_levels-1):
    for i in range(len(pyrGaussian) - 1):
        res = pyrGaussian[i] - expand(pyrGaussian[i + 1], filter_vec).astype(np.float64)
        pyr.append(res)
    pyr.append(pyrGaussian[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    the reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: Laplacian pyramid
    :param filter_vec: filter
    :param coeff:  a python list
    :return:
    """
    for i in range(len(coeff)):
        lpyr[i] = lpyr[i] * coeff[i]
    image = lpyr[-1]
    for i in range(len(coeff) - 2, -1, -1):
        image = expand(image, filter_vec) + lpyr[i]
    return image


def render_pyramid(pyr, levels):
    """
    Built the image by concatenating all the pyramid levels.
    :param pyr:is either a Gaussian or Laplacian pyramid
    :param levels: is the number of levels to present in the result ≤ max_levels
    :return: the big image
    """
    levels = min(len(pyr), levels)
    col = 0
    row = np.shape(pyr[0])[0]
    for i in range(levels):
        minVal, maxVal = np.min(pyr[i]), np.max(pyr[i])
        pyr[i] = (pyr[i] - minVal) / (maxVal - minVal)
        col += np.shape(pyr[i])[1]
    res = np.zeros((row, col))
    col = 0
    for i in range(levels):
        res[:np.shape(pyr[i])[0], col:np.shape(pyr[i])[1] + col] = pyr[i]
        col += np.shape(pyr[i])[1]
    return res


def display_pyramid(pyr, levels):
    """
     Display the stacked pyramid image
    :param pyr:is either a Gaussian or Laplacian pyramid
    :param levels: is the number of levels to present in the result ≤ max_levels
    :return: the stacked pyramid image
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: input grayscale image to be blended
    :param im2: input grayscale image to be blended
    :param mask: – is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
    of im1 and im2 should appear in the resulting im_blend
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian
    pyramids.
    :param filter_size_im:  the size of the Gaussian filter
    :param filter_size_mask: the size of the Gaussian filter
    :return: blended image from the Laplacian pyramid
    """
    # 1. Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively.
    pyr1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    pyr2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    # 2. Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64).
    mask = mask.astype(np.float64)
    mask_pye, mask_filter = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    levels = min(len(pyr1), len(pyr2), max_levels)
    l_out = []
    for i in range(levels):
        # according to the formula: L_out[k]= G_m[k]*L_1[k] + (1-G_m[k])*L_2[k]
        l_out.append(np.multiply(mask_pye[i], pyr1[i]) + np.multiply(1 - mask_pye[i], pyr2[i]))
    coeff = np.ones(levels, dtype=int)
    return laplacian_to_image(l_out, filter_vec1, coeff).clip(0, 1)


def plot_images(im1, im2, mask, blended_im, titleNeeded):
    """
    plot all images
    :param im1: first image
    :param im2: second image
    :param mask: mask image
    :param blended_im: blended image
    :param titleNeeded: boolean
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    imgplot = plt.imshow(im1)

    ax = fig.add_subplot(222)
    imgplot = plt.imshow(im2)
    if titleNeeded:
        ax1.set_title('My Balcony')
        ax.set_title('China')
    ax = fig.add_subplot(223)
    imgplot = plt.imshow(blended_im)
    ax = fig.add_subplot(224)
    imgplot = plt.imshow(mask, cmap="gray")
    plt.show()


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    blend my balcony and chine view.
    **The photos were taken by me ***
    :return: image1, image2, mask, im_blend
    """
    image_mask = read_image(relpath("externals/mask_room.jpg"), 1)
    image_mask = np.round(image_mask)
    image_mask = image_mask.astype(np.bool)

    image1 = read_image(relpath("externals/china_photo.jpg"), 2)
    image2 = read_image(relpath("externals/china_view.JPG"), 2)

    image_blend = np.zeros(image1.shape).astype(np.float64)
    red = pyramid_blending(image1[:, :, 0], image2[:, :, 0], image_mask, 9, 9, 7)
    image_blend[:, :, 0] = red
    green = pyramid_blending(image1[:, :, 1], image2[:, :, 1], image_mask, 9, 9, 7)
    image_blend[:, :, 1] = green
    blue = pyramid_blending(image1[:, :, 2], image2[:, :, 2], image_mask, 9, 9, 7)
    image_blend[:, :, 2] = blue
    plot_images(image1, image2, image_mask, image_blend, True)
    return image1, image2, image_mask, image_blend


def blending_example2():
    """
    blend Anna and corona-mask
    :return: image1, image2, mask, image_blend
    """
    image_mask = read_image(relpath("externals/pic_1_mask_black_frozen.jpg"), 1)
    image_mask = np.round(image_mask)
    image_mask = image_mask.astype(np.bool)
    image1 = read_image(relpath("externals/pic_1_frozen.jpg"), 2)
    image2 = read_image(relpath("externals/pic_1_woman.jpg"), 2)
    image_blend = np.zeros(image1.shape).astype(np.float64)
    red = pyramid_blending(image1[:, :, 0], image2[:, :, 0], image_mask, 9, 9, 7)
    image_blend[:, :, 0] = red
    green = pyramid_blending(image1[:, :, 1], image2[:, :, 1], image_mask, 9, 9, 7)
    image_blend[:, :, 1] = green
    blue = pyramid_blending(image1[:, :, 2], image2[:, :, 2], image_mask, 9, 9, 7)
    image_blend[:, :, 2] = blue
    plot_images(image1, image2, image_mask, image_blend, False)
    return image1, image2, image_mask, image_blend


if __name__ == '__main__':
    blending_example1()
    blending_example2()

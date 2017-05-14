import cv2
import numpy as np
import matplotlib.pyplot as plt


# Prompt for limited number of options
def promptForInputCategorical(message, options):
    """
    Prompts for user input with limited number of options (not used in this project)
    :param message: Message displayed to the user
    :param options: limited number of options. 
    Prompt will repeat until one of provided options typed correctly
    :return: user response
    """
    response = ''

    options_list = ', '.join(options)

    while response not in options:
        response = input('{} ({}): '.format(message, options_list))

    return response


def promptForInt(message):
    """
    Prompting for Integer input
    :param message: Informative message when prompting for integer input
    :return: integer input
    """
    result = None

    while result is None:
        try:
            result = int(input(message))
        except ValueError:
            pass
    return result


def promptForFloat(message):
    """
    Prompting for Float
    :param message: Informative message when prompting for float input
    :return: integer input
    """
    result = None

    while result is None:
        try:
            result = float(input(message))
        except ValueError:
            pass
    return result


def putThrs(img, low, high):
    """
    Was used at the stage when determining thresholds for binarization
    """
    cv2.putText(img=img,
                text='low threshold: {}, high threshold: {})'.format(low, high),
                org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=3)


def putText(img, text, origin=(100, 100), scale=1.0, color=(255, 0, 0), thickness=2):
    """
    Wrapper for OpenCV putText()
    :param img: 
    :param text: 
    :param origin: 
    :param scale: 
    :param color: 
    :param thickness: 
    :return: 
    """
    cv2.putText(img=img,
                text=text,
                org=origin,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scale,
                color=color,
                thickness=thickness)


def drawRect(img, lx, ly, rx, ry, color=(0, 255, 0), thickness=2):
    """
    Wrapper for OpenCV rectangle
    :param img: 
    :param lx: 
    :param ly: 
    :param rx: 
    :param ry: 
    :param color: 
    :param thickness: 
    :return: 
    """
    cv2.rectangle(img=img, pt1=(lx, ly), pt2=(rx, ry), color=color, thickness=thickness)


def funcSpace(argSpace, fitParams):
    """
    Creates a space of quadratic function f(y) = ay^2 + by + c values given a space of variables
    :param argSpace: space of variables, may be a single value
    :param fitParams: 
    :return: space of function values
    """
    a = fitParams[0]
    b = fitParams[1]
    c = fitParams[2]

    return a * (argSpace ** 2) + b * argSpace + c


def curvature(fitParams, variable, scale=1):
    """
    :param fitParams: 2nd order polynomial params (a, b, c in f(y) = ay^2 + by + c). Passing just a tuple of
    'a' and 'b' is enough
    :param variable: the point where curvature being evaluated (passing 'linspace' should return an array of curvatures
    for a given linspace.
    :param scale: number of units per pixel
    :return: value of curvature in units
    """
    a = fitParams[0]
    b = fitParams[1]

    return ((1 + (2 * a * variable * scale + b) ** 2) ** 1.5) / np.absolute(2 * a)


def plot(img, figsize=(12, 12), title=None, axis='off', cmap=None):
    """
    Wrapper for matplotlib.pyplot imshow. Used for jupyter notebook
    :param img: 
    :param figsize: 
    :param title: 
    :param axis: 
    :param cmap: 
    :return: 
    """
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.axis(axis)
    if cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)


def timeStamp():
    import datetime
    now = datetime.datetime.now()
    y = now.year
    d = now.day
    mo = now.month
    h = now.hour
    m = now.minute
    s = now.second

    return '{}_{}_{}_{}_{}_{}'.format(y, mo, d, h, m, s)


def drawBoxes(img, bBoxes, color=(0, 255, 0), thickness=4):
    """
    Universal bounding box painter, regardless of bBoxes format 
    :param img: image of interest
    :param bBoxes: list of bounding boxes.
    :param color: 
    :param thickness: 
    :return: 
    """
    for bBox in bBoxes:

        bBox = np.array(bBox)
        bBox = bBox.reshape(bBox.size)

        cv2.rectangle(img=img, pt1=(bBox[0], bBox[1]), pt2=(bBox[2], bBox[3]),
                      color=color, thickness=thickness)


# By Vivek Yadav: https://github.com/vxy10/ImageAugmentation
def transform_image(img, ang_range, shear_range, trans_range, brightness=False):
    """
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.
    A Random uniform distribution is used to generate different parameters for transformation
    """

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_m = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, rot_m, (cols, rows))
    img = cv2.warpAffine(img, trans_m, (cols, rows))
    img = cv2.warpAffine(img, shear_m, (cols, rows))

    # Brightness
    if brightness:
        img = augment_brightness(img)

    return img


# Idea by Vivek Yadav: https://github.com/vxy10/ImageAugmentation
def augment_brightness(image):
    hsv_img = hsv(image)

    random_brightness = np.random.uniform(0.75, 1.25)

    def clamp(a):
        return min(255, a * random_brightness)

    vfunc = np.vectorize(clamp)

    hsv_img[:, :, 2] = vfunc(hsv_img[:, :, 2])

    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


def change_colorspace(x, new_color_space, ch_to_heq=None):
    if ch_to_heq is None:
        return cv2.cvtColor(x, new_color_space)
    else:
        x = cv2.cvtColor(x, new_color_space)

        x = hist_eq(x, ch_to_heq)

        return x


def hls(x, ch_to_heq=None):
    return change_colorspace(x, cv2.COLOR_RGB2HLS, ch_to_heq)


def hsv(x, ch_to_heq=None):
    return change_colorspace(x, cv2.COLOR_RGB2HSV, ch_to_heq)


def yuv(x, ch_to_hec=None):
    return change_colorspace(x, cv2.COLOR_RGB2YUV, ch_to_hec)


def hist_eq(x, ch_to_heq=None):
    """
    Histogram equalization. Equalizes each channel separately.
    :param x: 
    :param ch_to_heq: 
    :return: 
    """
    if ch_to_heq is None:
        ch_to_heq = range(x.shape[2])

    for i in ch_to_heq:
        x[:, :, i] = cv2.equalizeHist(x[:, :, i])

    return x


def colorHeatMap(heatMapMono, cmap=cv2.COLORMAP_HOT):
    """
    Makes an RGB version of the 1-channel heatMap
    :param heatMapMono: 
    :param cmap: The color map of choice
    :return: RGB heatMap
    """
    heatMapInt = cv2.equalizeHist(heatMapMono.astype(np.uint8))
    heatColor = cv2.applyColorMap(heatMapInt, cmap)
    heatColor = cv2.cvtColor(heatColor, code=cv2.COLOR_BGR2RGB)

    return heatColor


def rgbImage(imageFileName, resize=False):
    """
    Opens image as RGB with OpenCV
    :param imageFileName: 
    :param resize: Halves width and height if True
    :return: RGB image
    """
    image = cv2.imread(imageFileName)

    if resize:
        image = cv2.resize(src=image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flipImage(image):
    """
    Horizontal flip with OpenCV
    :param image: 
    :return: Horizontally-flipped image
    """
    return cv2.flip(image, 1)

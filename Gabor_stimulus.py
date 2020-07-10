import numpy as np
import cv2
import os


def gabor_function(r, center, sig, th, fr):
    th = np.deg2rad(th)
    e = np.exp(-((r[0] - center[0]) ** 2 + (r[1] - center[1]) ** 2) / (2 * (sig * sig)))
    z = (r[0] - center[0]) * np.sin(th) + (r[1] - center[1]) * np.cos(th)
    c = np.cos(2 * fr * np.pi * z)
    return e * c


def make_gabordist(size, sig, th, fr, ic=None):
    # ic: indicated center

    sig = np.float64(sig)
    th = np.float64(th)
    fr = np.float64(fr)

    gabordist = np.zeros(size)

    if ic is not None:
        center = ic
    else:
        center = [size[0] / 2, size[1] / 2]

    for x in range(size[0]):
        for y in range(size[1]):
            gabordist[x][y] = gabor_function([x, y], center, sig, th, fr)

    return gabordist


def fit_imageformat(image):
    image += 1
    image /= 2
    image *= 255
    image = image.astype('uint8')

    csize = [image.shape[0], image.shape[1], 3]
    colorimage = np.ndarray(csize)

    colorimage[:, :, 0] = image
    colorimage[:, :, 1] = image
    colorimage[:, :, 2] = image

    return colorimage


def make_video(sw, size, sig, th, fr, delta):
    filename = 'moveG_' + str(sw) + '_' + str(delta) + '_' + str(sig) + '_' + str(th) + '_' + str(fr)
    os.makedirs(filename)

    if sw == 0:  # rotation
        var = 'th'
        center = None
        ran = int(360 * 3 / delta) + 1
    elif sw == 1:  # move to left from right
        var = 'center[1]'
        delta = -delta
        center = [size[0] / 2, size[1] / 2 + size[1] * 1.5]
        ran = int(abs(size[1] * 3 / delta)) + 1

    with open(filename + '/test_list.txt', mode='w') as f:
        for n in range(ran):
            image = make_gabordist(size, sig, th, fr, center)
            image = fit_imageformat(image)
            exec (var + '+=' + str(delta))

            cv2.imwrite(filename + '/' + str(n) + '.jpg', image)
            f.write(filename + '/' + str(n) + '.jpg\n')

    return 0

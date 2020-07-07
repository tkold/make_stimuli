import numpy as np
import cv2
from precalc import Gabor_stimulus as Gs
from precalc import p_util as util
from precalc import Geometric_image as GI

size = [120, 160]  # Vertical x side

save = 1
show = 0

gabor = 0
load = 0
gi =1

if gabor == 1:

    sig = 20.0
    th = 90.0
    fr = 0.50

    while fr < 10:
        filename = 'G_' + str(sig) + '_' + str(th) + '_' + str(fr)

        showimage = Gs.make_gabordist(size, sig, th, fr)
        showimage = Gs.fit_imageformat(showimage)
        print(np.sum(showimage / (size[0] * size[1] * 3)))

        if save == 1: cv2.imwrite(filename + '.jpg', showimage)  # Output
        if show == 1: util.cvshow(showimage)  # Show
        if save == 1: np.savetxt(filename + '.csv', showimage[:, :, 0])
        util.make_stimulus_dir(filename + '.jpg', filename, 20)
        fr += 0.1

if load == 1:
    filename = 'Fraser_B'
    util.make_stimulus_dir(filename + '.jpg', filename, 20)

if gi==1:


    dc=0.5
    cpara = 0.5
    while cpara < 35+dc:
        filename=GI.Geometric_image(size, 'circle', [0,0,cpara])
        util.make_stimulus_dir(filename + '.png', filename, 20)
        cpara += dc

if gi == 2:
    dc=0.05
    cpara = 0.05
    while cpara < 0.5+dc:
        filename = GI.Geometric_image(size, 'spiral', [50, -500, 0.1, cpara, 4])
        util.make_stimulus_dir(filename + '.png', filename, 20)
        cpara += dc
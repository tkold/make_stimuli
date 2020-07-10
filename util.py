import os
import shutil
import cv2

def make_stimulus_dir(image, folder, image_num):
    os.makedirs(folder)
    with open(folder + '/test_list.txt', mode='w') as f:
        for n in range(image_num):
            copy = folder + '/' + str(n) + '.jpg'
            shutil.copyfile(image, copy)
            f.write(folder + '/' + str(n) + '.jpg\n')


def resize(originalpath, savepath, size):
    img = cv2.imread(originalpath)
    img2 = cv2.resize(img , (size[1], size[0]))
    cv2.imwrite(savepath, img2)
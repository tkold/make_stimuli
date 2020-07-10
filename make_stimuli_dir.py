import os
import shutil
import util as util
frame_number=20
imagepath='.'

def make_stimulus_dir(image, folder, image_num):
    os.makedirs(folder)
    with open(folder + '/test_list.txt', mode='w') as f:
        for n in range(image_num):
            copy = folder + '/' + str(n) + '.jpg'
            shutil.copyfile(image, copy)
            f.write(folder + '/' + str(n) + '.jpg\n')



for dir in os.listdir(imagepath):
    if '.jpg' in dir:
        util.make_stimulus_dir(dir, dir.replace('.jpg',''), frame_number)
    elif '.png' in dir:
        util.make_stimulus_dir(dir, dir.replace('.png',''), frame_number)
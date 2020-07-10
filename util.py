import cv2

def resize(originalpath, savepath, size):
    img = cv2.imread(originalpath)
    img2 = cv2.resize(img , (size[1], size[0]))
    cv2.imwrite(savepath, img2)
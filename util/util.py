import os
import cv2
import numpy as np
import shutil
from natsort import natsorted


def cvshow(showimage):
    cv2.imshow('image', showimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_folderlist(folder):
    follist = list()
    for n in natsorted(os.listdir(folder)):
        if os.path.isdir(folder + '/' + n):
            follist.append(n)
    for n in follist:
        for m in natsorted(os.listdir(folder + '/' + n)):
            if os.path.isdir(folder + '/' + n + '/' + m):
                follist.append(n + '/' + m)
    return follist


def make_foldertree(rootpath, folder_list):
    if not os.path.exists(rootpath): os.makedirs(rootpath)

    for f in range(len(folder_list)):
        if not os.path.exists(rootpath + '/' + folder_list[f]):
            os.makedirs(rootpath + '/' + folder_list[f])


def Prime_factorization(ch, hor=1, ver=1):
    pnlist = [2, 3, 5]  # prime number
    npn = [0, 0, 0]  # number of prime numbers

    for n in range(len(pnlist)):
        num = pnlist[n]
        while (ch % num) == 0:
            ch /= num
            npn[n] += 1
    if ch != 1:
        pnlist.append(ch)
        npn.append(1)

    for n in reversed(range(len(pnlist))):
        if npn[n] == 0:
            del npn[n]
            del pnlist[n]

    cmblist = list()
    ln = 1
    for n in range(len(pnlist)):
        ln *= npn[n] + 1
    for n in range(ln):
        cmblist.append([1 * hor, 1 * ver])

    nnpn = map(lambda n: n + 1, npn)
    for p in range(len(pnlist)):

        if p != len(pnlist) - 1:
            lr = np.prod(nnpn[p + 1:len(npn)])  # large repeat
        else:
            lr = 1

        if p != 0:
            sr = np.prod(nnpn[0:p])  # small repeat
        else:
            sr = 1

        for nlr in range(lr):
            rlr = np.prod(nnpn[p + 1:len(npn)])
            for n in range(nnpn[p]):
                for nsr in range(sr):
                    nn = nlr * (nnpn[p] * sr) + n * sr + nsr
                    cmblist[nn][0] = cmblist[nn][0] * (pnlist[p] ** (npn[p] - n))
                    cmblist[nn][1] = cmblist[nn][1] * (pnlist[p] ** (n))
    och = cmblist[0][0]  # optimize hor
    ocv = cmblist[0][1]  # optimize ver
    for n, m in cmblist:
        if not 'md' in locals():
            och = n / hor
            ocv = m / ver
            md = abs(n - m)
        elif abs(n - m) < md:
            och = n / hor
            ocv = m / ver
            md = abs(n - m)

    return och, ocv

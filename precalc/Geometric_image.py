import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import p_util as util

def trans(r, theta):

    xt = np.cos(theta) * r[0] - np.sin(theta) * r[1]
    yt = np.sin(theta) * r[0] + np.cos(theta) * r[1]

    return [xt, yt]

def graph_mdf(size):
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)

    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)

    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.axes().set_aspect('equal')

    plt.ylim(-size[0] / 2, size[0] / 2)
    plt.xlim(-size[1] / 2, size[1] / 2)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

def Geometric_image(size, geo, para):
    fig = plt.figure()
    lim = np.sqrt( (size[0]/2)**2 + (size[1]/2)**2 ) * 1.1
    graph_mdf(size)

    if geo=='circle':
        reso = 50000
        r=para[0]
        dr=para[1]
        ddr=para[2]

        ri=0
        ylist=list()
        while r < lim:
            dr+= ddr
            r+= dr

            x = np.zeros(reso)
            i = 0
            for t in np.linspace(-np.pi/2, np.pi/2,reso):
                x[i]=np.sin(t)*r
                i+=1

            y = np.zeros(reso)
            for i in range(len(x)):
                y[i] = np.sqrt(r**2 - x[i]**2)
            ylist.append(y)
            plt.plot(x, ylist[ri], color='black', lw=7)
            ri+=1

            ylist.append(-y)
            plt.plot(x, ylist[ri], color='black', lw=7)
            ri+=1


        filename= geo+'_'+str(para[0])+'_'+str(para[1])+'_'+str(para[2])+'.png'

    elif geo=='spiral':
        a=para[0]
        tf=para[1]
        dt=para[2]
        b=para[3]
        ns=para[4]

        ri=0
        xlist=list()
        ylist=list()

        tt = 2 * np.pi / ns  # trans theta
        for n in range(ns):
            t = tf
            x = y = np.zeros(0)
            r = 0
            while r < lim:
                r = a * (np.e**(b*t))
                xi = r * np.cos(t)
                yi = r * np.sin(t)
                rt = trans([xi, yi], tt*(n))
                x = np.insert(x, 0, rt[0])
                y = np.insert(y, 0, rt[1])
                t += dt

            xlist.append(x)
            ylist.append(y)
            plt.plot(xlist[ri], ylist[ri], color='black', lw=7)
            ri+=1

        filename = geo+'_'+str(para[0])+'_'+str(para[1])+'_'+str(para[2])+'_'+str(para[3])+'_'+str(para[4])+'.png'

    fig.savefig('original.png')
    util.resize('original.png', filename, size)
    return filename.replace('.png','')
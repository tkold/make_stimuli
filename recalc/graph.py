import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from util import util as util
from util import dotmdf as dotmdf
from datetime import datetime

def graph_plot(path, defi=0):
    leg=0
    folderlist=util.get_folderlist(path)
    for folder in folderlist:
        for dir in natsorted(os.listdir(path+'/'+folder)):
            if '.csv' in dir:
                datapath=path+'/'+folder+'/'+dir

                fig = plt.figure()

                if os.path.exists(path+'/condition.csv'):
                    data = np.loadtxt(datapath)
                    with open(path+'/condition.csv', 'r') as f:
                        lines = f.readlines()#lines = f.readline().strip('\n').split(',')
                        del lines[:3]
                        for i in range(len(lines)):
                            lines[i]=lines[i].split(',')[0]
                        print('No',dir.replace('.csv', ''))
                        plt.title(dir.replace('.csv', ''))

                else:
                    data = np.loadtxt(datapath, skiprows = 1)
                    with open(datapath) as f:
                        lines = f.readline().split(',')
                        for i in range(len(lines)):
                            lines[i]=lines[i].replace('\n','').replace('# ','')
                        plt.title(dir.replace('.csv', '') + ' ' + ','.join(lines))

                cmap=matplotlib.colors.XKCD_COLORS.items()

                plt.xlabel('Channel')
                plt.ylabel(dir.replace('.csv', ''))
                ax = fig.add_subplot(111)

                ydata=data[:,1:data.shape[1]]
                if np.min(ydata) > 0:
                    plt.ylim(0, np.max(ydata))
                elif np.max(ydata) > 0:
                    ax.plot( [np.min(data[:, 0]), np.max(data[:, 0])], [0, 0], color='black')
                elif not np.isnan(np.min(ydata)):
                    plt.ylim(np.min(ydata), 0)
                
                for yn in range(1, data.shape[1]):
                    ax.plot(data[:, 0], data[:, yn], '-o', label=lines[yn - 1], color=cmap[yn][1], linewidth=0.1, markersize=8)

                if ydata.shape[0]<10 and os.path.exists(path + '/condition.csv'):
                    plt.xticks(np.arange(0, ydata.shape[0], 1))
                fig.savefig(datapath.replace('.csv','.png'))

                if  leg==0:
                    if not os.path.exists(path + '/condition.csv'):
                        for idx, (color, rgb) in enumerate(matplotlib.colors.XKCD_COLORS.items()):
                            ax.plot(data[:, 0], data[:, 0], '-o', label='ch'+str(idx), color=rgb, linewidth=0.1, markersize=8)
                    plt.legend().get_frame().set_alpha(1)
                    fig.savefig(path + "/legend.png", bbox_inches='tight')
                    leg=1
                plt.close()

    if '_node' in dir:
        dotmdf.make_cgimage(path, footer=dir.split('_')[-1].replace('.csv',''),defi=defi)
    else:
        dotmdf.make_cgimage(path,defi=defi)

def plot_image(data, savepath, title, subtitle=None, nor=None):

    row, col = util.Prime_factorization(data.shape[0], data.shape[1], data.shape[2])

    if nor==None: data = data / np.max(np.abs(data))
    else: data = data / nor

    if int((col * data.shape[2]) / 50)==0 or int((row * data.shape[1]) / 50)==0:
        cor=10
    elif int((col * data.shape[2]) / 50)==1 or int((row * data.shape[1]) / 50)==1:
        cor=5
    elif int((col * data.shape[2]) / 50)<5 or int((row * data.shape[1]) / 50)<5:
        cor=3
    else:
        cor=1
    fig = plt.figure(figsize=(int((col * data.shape[2])*cor / 50), int((row * data.shape[1])*cor / 50)))
    #plt.rcParams["font.family"] = "Times New Roman"
    marg = 0.1
    plt.subplots_adjust(wspace=marg, hspace=marg)
    plt.subplots_adjust(left=0, right=1, bottom=0.01, top=0.9)

    num = 0
    while num < row * col:
        plt.subplot(row, col, num + 1)
        plt.imshow(data[num, :, :], vmin=-1, vmax=1,  cmap='seismic')
        if not subtitle==None:
            plt.title(subtitle[num], fontsize=int(data.shape[2]*cor / 15), pad=2)
        plt.axis('off')
        num += 1
    plt.suptitle(title, fontsize=int(col * data.shape[2]*cor / 15))
    fig.savefig(savepath + '.png')
    plt.close()

def compare_channel(savepath ,explist, nodename, channel):
    savepath=savepath+'/'+nodename.split('/')[0]+'/ch'+str(channel)
    if not os.path.exists(savepath): os.makedirs(savepath)
    savepath=savepath+'/'+nodename.split('/')[-1]

    imglist=list()
    mx=mn=0
    e=0
    for exp in explist:
        img=np.load(exp + '/' + nodename + '.npy')[int(channel)]
        if e==0:
            imglist=np.ndarray([len(explist),img.shape[0],img.shape[1]])
        imglist[e,:,:]=img
        e+=1
    titlelist=list()
    for ex in explist:
        titlelist.append(ex.split('/')[1])
    plot_image(imglist, savepath, nodename+'_'+str(channel), subtitle=titlelist)

def evaluated_channel(rootpath,nodename, channellist, switch, x=None):
    path=rootpath+'/'+nodename
    swlist=['','_nodeactsum.csv','_nodesupsum.csv']

    nodedata  = np.loadtxt(path + swlist[switch])

    y=np.ndarray([nodedata.shape[0], len(channellist)+1])
    if x is None:
        x=np.arange(nodedata.shape[0])
    y[:,0]=x
    for yn in range(1,y.shape[1]):
        y[:,yn]=nodedata[:,int(channellist[yn-1])]
    return y
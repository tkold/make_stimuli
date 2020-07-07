import os
import numpy as np
from natsort import natsorted
from datetime import datetime
from util import util as util

def loadlist(list):
    output=dict()
    with open(list, mode='r') as f:
        txt = f.read().split('\n')
        for node in txt:
            if len(node)!=0:
                node=node.split(',')
                title=node.pop(0)
                subtitle=str()
                for i in reversed(range(len(node))):
                    node[i]=node[i].replace(' ','')
                    if not node[i].isdigit():
                        subtitle=','+node[i] + subtitle
                        del node[i]
                title+=subtitle
                output[title]=node

    return output


class comnode():
    def __init__(self, node):
        self.node = node

    def changerate(self):
        y=np.ndarray(self.node.shape[1])
        for r in range(self.node.shape[1]):
            mx = np.max(self.node[:,r])
            mn = np.min(self.node[:,r])
            if not np.isnan(mx) and not np.isnan(mn):
                if np.abs(mx) > np.abs(mn):
                    y[r] = (mx - mn) / mn
                else:
                    y[r] = (mn - mx) / mx
            else:
                y[r] = np.NaN
        return y

    def max(self):
        y=np.ndarray(self.node.shape[1])
        for r in range(self.node.shape[1]):
            y[r] = np.max(self.node[:, r])
        return y

    def min(self):
        y = np.ndarray(self.node.shape[1])
        for r in range(self.node.shape[1]):
            y[r] = np.min(self.node[:, r])
        return y

    def ave(self):
        y = np.ndarray(self.node.shape[1])
        for r in range(self.node.shape[1]):
            y[r] = np.average(self.node[:, r])
        return y

def evaluate(nodepath, switch):

    if switch==1:
        asnoder = comnode(np.loadtxt(nodepath + '_nodeactsum.csv')).changerate()
        mxnodemx = comnode(np.loadtxt(nodepath + '_nodemax.csv')).max()
        y=np.ndarray([asnoder.shape[0]])
        for c in range(asnoder.shape[0]):
            if abs(asnoder[c]) > 0.1 and mxnodemx[c] > 0.1:
                y[c] = 1
            else:
                y[c] =-1

    elif switch==2:
        ssnoder = comnode(np.loadtxt(nodepath + '_nodesupsum.csv')).changerate()
        mnnodemn = comnode(np.loadtxt(nodepath + '_nodemin.csv')).min()
        y=np.ndarray([ssnoder.shape[0]])
        for c in range(ssnoder.shape[0]):
            if abs(ssnoder[c]) > 0.1 and mnnodemn[c] < -0.1:
                y[c] = 1
            else:
                y[c] =-1

    elif switch==3:
        asnoder = comnode(np.loadtxt(nodepath + '_nodeactsum.csv')).changerate()
        mxnodemx = comnode(np.loadtxt(nodepath + '_nodemax.csv')).max()

        ssnoder = comnode(np.loadtxt(nodepath + '_nodesupsum.csv')).changerate()
        mnnodemn = comnode(np.loadtxt(nodepath + '_nodemin.csv')).min()

        y=np.ndarray([asnoder.shape[0]])
        for c in range(asnoder.shape[0]):
            if abs(asnoder[c]) > 0.1 and mxnodemx[c] > 0.1:
                y[c] =-1
            elif abs(ssnoder[c]) > 0.1 and mnnodemn[c] < -0.1:
                y[c] =-1
            else:
                y[c] = 1

    elif switch==4:
        mxnodemx = comnode(np.loadtxt(nodepath + '_nodemax.csv')).max()
        mnnodemn = comnode(np.loadtxt(nodepath + '_nodemin.csv')).min()

        y=np.ndarray([mxnodemx.shape[0]])
        for c in range(mxnodemx.shape[0]):
            if mxnodemx[c] > 0.1 or mnnodemn[c] < -0.1:
                y[c] = 1
            else:
                y[c] =-1

    return y

def evaluate_g(nodepath1, nodepath2, switch):

    if switch==1:
        rate=0.1
        mxavnode1 = comnode(np.loadtxt(nodepath1 + '_nodemax.csv')).ave()
        mxavnode2 = comnode(np.loadtxt(nodepath2 + '_nodemax.csv')).ave()
        mnavnode1 = comnode(np.loadtxt(nodepath1 + '_nodemin.csv')).ave()
        mnavnode2 = comnode(np.loadtxt(nodepath2 + '_nodemin.csv')).ave()
        y=np.ndarray([mxavnode1.shape[0]])
        for c in range(mxavnode1.shape[0]):
            ma1=mxavnode1[c]
            ma2=mxavnode2[c]
            if ma1 > 0.1 or ma2 > 0.1:
                if abs((ma1-ma2)/ma2) > rate or abs((ma2-ma1)/ma1) > rate:
                    y[c]=1
                    continue

            mn1=mnavnode1[c]
            mn2=mnavnode2[c]
            if mn1 < -0.1 or mn2 < -0.1:
                if abs((ma1-ma2)/ma2) > rate or abs((ma2-ma1)/ma1) > rate:
                    y[c]=1
                else:
                    y[c]=0
            else:
                y[c]=0


    return y



def listup(anapath, switch, secondpath=None, cor=1):
    anapath='analysis/'+anapath
    folderlist=util.get_folderlist(anapath)
    path=anapath + '/Applicable_list_' + str(switch) + '.csv'
    if secondpath is not None: path=path.replace('.csv','_with_'+secondpath+'.csv')
    with open(path, mode='w') as f:

        for fol in folderlist:
            if not 'Applicable' in fol:
                nodelist=list()
                for dir in os.listdir(anapath + '/' + fol):
                    if ('.csv' in dir) and (not 'error' in dir):
                        dir=dir.replace('_' + dir.split('_')[-1], '')
                        if not dir in nodelist: nodelist.append(dir)
                for node in nodelist:
                    f.write( fol + '/' + node)
                    if secondpath is None:
                        ev = evaluate( anapath + '/' + fol + '/' + node, switch)
                    else:
                        ev = evaluate_g( anapath + '/' + fol + '/' + node, 'analysis/'+ secondpath + '/' + fol + '/' + node, switch)

                    nn=0
                    ev*=cor
                    for e in ev:
                        if e==1: f.write(', ' + str(nn))
                        nn+=1

                    f.write('\n')
    return path

def complist(list1path, list2path, listup='dif'):
    list1 = loadlist(list1path)
    list2 = loadlist(list2path)
    list1root=list1path
    list1path=list1path.split('/')[1] + '-' + list1path.split('/')[2].replace('.csv','')
    list2path=list2path.split('/')[1] + '-' + list2path.split('/')[2].replace('.csv','')

    if listup == 'dif':
        l1='analysis/complist/' + list1path + '_not_in_' +  list2path + '.csv'
        with open(l1, mode='w') as f:
            for key in list1.keys():
                f.write(key)
                numlist=list2[key]
                for n in list1[key]:
                    if not n in numlist:
                        f.write(',' + n)
                f.write('\n')

        l2='analysis/complist/' + list2path + '_not_in_' +  list1path + '.csv'
        with open(l2, mode='w') as f:
            for key in list1.keys():
                f.write(key)
                numlist=list1[key]
                for n in list2[key]:
                    if not n in numlist:
                        f.write(',' + n)
                f.write('\n')
        return [l1,l2]

    elif listup == 'eq':
        l1=list1root.replace('.csv','') + '_is_in_' +  list2path + '.csv'
        with open(l1, mode='w') as f:
            for key in list1.keys():
                f.write(key)
                numlist=list1[key]
                for n in list2[key]:
                    if n in numlist:
                        f.write(',' + n)
                f.write('\n')
        return [l1]
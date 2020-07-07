import os
import numpy as np
from natsort import natsorted
from datetime import datetime
from util import util as util

def check_whether_same():
    #chack whether absmin and min are same
    root='analysis'
    anaroot=root+'/'+os.listdir(root)[0]
    folderlist=util.get_folderlist(anaroot)

    with open(anaroot+'/check_whether_same_'+str(datetime.now().strftime('%B%d  %H:%M:%S')), mode='w') as f:
        #f.write(huname+'\n')
        for folder in folderlist:
            folder=anaroot+'/'+folder

            for dir in natsorted(os.listdir(folder)):
                if 'nodeabsmin.csv' in dir:
                    dir=folder+'/'+dir
                    absmindata = np.loadtxt(dir)
                    mindata    = np.loadtxt(dir.replace('absmin', 'min'))

                    if not np.allclose(absmindata, mindata):
                        f.write(dir + '\n')

def del_node_params(experiment):
    experiment=experiment+'/nodedata'
    folderlist=util.get_folderlist(experiment)

    for fol in folderlist:
        for dir in os.listdir(experiment + '/' + fol):
            dirpath=experiment + '/' + fol + '/' + dir
            if os.path.isfile(dirpath) and ('.csv in dir'):
                os.remove(dirpath)
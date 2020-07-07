import os
import time
import argparse
import numpy as np
from datetime import datetime
from natsort import natsorted
from main import run_PredNet as r
from recalc import node_analysis as na
from recalc import r_util

parser = argparse.ArgumentParser(description='PredNet_UI')
parser.add_argument('--runPN', '-rpn', default=1,  type=int, help='whether PredNet')
parser.add_argument('--condition_sw', '-sw', default=1,  type=int, help='condition')
parser.add_argument('--fit_imagenum', '-fimn', default=1,  type=int, help='fit image number')
parser.add_argument('--recalcpara', '-rcp', default=0,  type=int, help='recaluclation parameter')
parser.set_defaults(test=False)
args = parser.parse_args()


if args.recalcpara==1: args.runPN = 0

def analyzes(savedir, tl, startt):
    with open(savedir + '/runtime.txt', mode='a') as f:
        f.write("%s\n" % tl[len(tl) - 1])

        if args.recalcpara == 1:
            r_util.del_node_params(savedir)

        if args.runPN==1:
            tl.append(['end', time.time() - startt - tl[len(tl) - 1][1]])
            f.write("%s\n" % tl[len(tl) - 1])

        experiment = list()
        for dir in natsorted(os.listdir(savedir + '/act')):
            experiment.append(savedir + '/act/' + dir)

        print('start_get_nodedata')
        na.get_nodedata(experiment, savedir)
        tl.append(['output error time series', time.time() - startt - tl[len(tl) - 1][1]])
        f.write("%s\n" % tl[len(tl) - 1])

        tl.append(['total', time.time() - startt])
        f.write("%s\n" % tl[len(tl) - 1])

    return 0

if args.runPN==1:
    if args.condition_sw==0:
        path='condition/test.txt'
    elif args.condition_sw==1:
        path='condition/time_series.txt'

    imagelist=natsorted(os.listdir('.'))

    with open(path, mode='r') as f:
        line = f.readline().strip()
        while line:
            s=line.split(' ')
            if len(s)==2:
                if s[1].isdigit():exec(s[0] + '=int(s[1])')
                else: exec(s[0] + '=s[1]')
            else: exec(s[0] + '=None')
            line = f.readline().strip()

    for image in imagelist:
        if os.path.isdir(image) and os.path.exists(image+'/test_list.txt'):
            images=image+'/test_list.txt'
            if args.fit_imagenum==1: input_len=len(os.listdir(image))-1
            tl=list()#time list
            savedir=image+'_'+str(datetime.now().strftime('%B%d  %H:%M:%S'))
            startt=time.time()

            tl.append([image+'start',time.time()-startt])
            print(image+'_start')
            prediction_error=r.run_PredNet(images, sequences, gpu, root, initmodel, resume, \
                          size, channels, offset, input_len, ext, bprop, save, period, test, savedir)

            savedir = 'runs/' + savedir
            np.savetxt(savedir+'/prediction_error.csv',prediction_error)
            analyzes(savedir, tl, startt)
else:
    for savedir in natsorted(os.listdir('runs')):
        savedir = 'runs/' + savedir
        with open(savedir + '/runtime.txt', mode='a') as f:
            f.write("addtional analysis\n")
        print('start_analysis',savedir)
        tl=list()#time list
        startt=time.time()
        tl.append(['analysis start',time.time()-startt])
        analyzes(savedir, tl, startt)
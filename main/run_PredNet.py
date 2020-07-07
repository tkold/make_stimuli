import os
from datetime import datetime
import numpy as np
from main import node_monitor as nm
from PIL import Image
from main import save_nodedata as sn

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import chainer.computational_graph as c
import net

def load_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples

def read_image(path, size, offset):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    top = offset[1] + (image.shape[1]  - size[1]) / 2
    left = offset[0] + (image.shape[2]  - size[0]) / 2
    bottom = size[1] + top
    right = size[0] + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image /= 255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

def prepare_PredNet(savedir):

    if not os.path.exists('runs'):
        os.makedirs('runs')

    save_root = 'runs/' + savedir
    os.makedirs(save_root+'/models')  # made models
    os.makedirs(save_root+'/images')  # output images
    os.makedirs(save_root+'/act')     # node activation

    return save_root


def run_PredNet(images='', sequences='', gpu=-1, root='.', initmodel='', resume='', size='160,120', \
             channels='3,48,96,192', offset='0,0', input_len=50, ext=10, bprop=20, save=10000, period=1000000, test=None, savedir=datetime.now().strftime('%B%d  %H:%M:%S')):

    if (not images) and (not sequences):
        print('Please specify images or sequences')
        exit()

    # make folder
    save_root=prepare_PredNet(savedir)

    #save condition
    with open(save_root + '/run_condition.txt', mode='w') as f:
        f.write('images '    + str(images)    + '\n')
        f.write('sequences ' + str(sequences) + '\n')
        f.write('gpu '       + str(gpu)       + '\n')
        f.write('root '      + str(root)      + '\n')
        f.write('initmodel ' + str(initmodel) + '\n')
        f.write('resume '    + str(resume)    + '\n')
        f.write('size '      + str(size)      + '\n')
        f.write('channels '  + str(channels)  + '\n')
        f.write('offset '    + str(offset)    + '\n')
        f.write('input_len ' + str(input_len) + '\n')
        f.write('ext '       + str(ext)       + '\n')
        f.write('bprop '     + str(bprop)     + '\n')
        f.write('save '      + str(save)      + '\n')
        f.write('period '    + str(period)    + '\n')
        f.write('test '      + str(test)      + '\n')

    size = size.split(',')
    for i in range(len(size)):
        size[i] = int(size[i])
    channels = channels.split(',')
    for i in range(len(channels)):
        channels[i] = int(channels[i])
    offset = offset.split(',')
    for i in range(len(offset)):
        offset[i] = int(offset[i])

    if gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu >= 0 else np

    # Create Model
    prednet = net.PredNet(size[0], size[1], channels)
    model = L.Classifier(prednet, lossfun=mean_squared_error)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # GPU or CPU
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
        print('Running on a GPU')
    else:
        print('Running on a CPU')

    # Init/Resume
    if initmodel:
        print('Load model from', initmodel)
        serializers.load_npz(initmodel, model)
    if resume:
        print('Load optimizer state from', resume)
        serializers.load_npz(resume, optimizer)

    # prepare loadlist
    if images: sequencelist = [images]
    else:      sequencelist = load_list(sequences, root)

    prediction_error=np.empty(0)
    # run PredNet
    if test == True:
        for seq in range(len(sequencelist)):
            imagelist = load_list(sequencelist[seq], root)
            prednet.reset_state()
            loss = 0
            batchSize = 1
            x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
            y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
            for i in range(0, len(imagelist)):
                print('frameNo:' + str(i))
                x_batch[0] = read_image(imagelist[i], size, offset)
                if i!=0:
                    prediction_error=np.append(prediction_error,np.sum(abs(x_batch[0].copy() - chainer.cuda.to_cpu(model.y.data[0].copy()))))
                loss += model(chainer.Variable(xp.asarray(x_batch)),
                              chainer.Variable(xp.asarray(y_batch)))
                if gpu >= 0: model.to_cpu()
                if ((i % save) == 0 and i!=0) or (save == 1 and i==0):
                    sn.save_nodedata(model, save_root+'/act/image_'+str(i))
                    #nm.node_monitor(model, save_root+'/network/image_'+str(i))
                if gpu >= 0: model.to_gpu()
                loss.unchain_backward()
                loss = 0
                if gpu >= 0: model.to_cpu()
                write_image(x_batch[0].copy(), save_root+'/images/test_' + str(i) + 'x.png')
                write_image(model.y.data[0].copy(), save_root+'/images/test_' + str(i) + 'y_0.png')
                if gpu >= 0: model.to_gpu()

                if i == 0 or (input_len > 0 and i % input_len != 0):
                    continue
                if gpu >= 0: model.to_cpu()
                x_batch[0] = model.y.data[0].copy()
                if gpu >= 0: model.to_gpu()
                for j in range(ext):
                    print('extended frameNo:' + str(j + 1))
                    loss += model(chainer.Variable(xp.asarray(x_batch)),
                                  chainer.Variable(xp.asarray(y_batch)))
                    loss.unchain_backward()
                    loss = 0
                    if gpu >= 0: model.to_cpu()
                    write_image(model.y.data[0].copy(), save_root+'/images/test_' + str(i) + 'y_' + str(j + 1) + '.jpg')
                    x_batch[0] = model.y.data[0].copy()
                    if gpu >= 0: model.to_gpu()
                prednet.reset_state()
    else:
        logf = open('log.txt', 'w')
        count = 0
        seq = 0
        while count < period:
            imagelist = load_list(sequencelist[seq], root)
            prednet.reset_state()
            loss = 0

            batchSize = 1
            x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
            y_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
            if len(imagelist) == 0:
                print("Not found images.")
                break
            x_batch[0] = read_image(imagelist[0], size, offset);
            for i in range(1, len(imagelist)):
                y_batch[0] = read_image(imagelist[i], size, offset);
                loss += model(chainer.Variable(xp.asarray(x_batch)),
                              chainer.Variable(xp.asarray(y_batch)))

                print('frameNo:' + str(i))
                if (i + 1) % bprop == 0:
                    model.zerograds()
                    loss.backward()
                    loss.unchain_backward()
                    loss = 0
                    optimizer.update()
                    if gpu >= 0: model.to_cpu()
                    write_image(x_batch[0].copy(), save_root+'/images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'x.png')
                    write_image(model.y.data[0].copy(),
                                save_root+'/images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'y.png')
                    write_image(y_batch[0].copy(), save_root+'/images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'z.png')
                    if gpu >= 0: model.to_gpu()
                    print('loss:' + str(float(model.loss.data)))
                    logf.write(str(i) + ', ' + str(float(model.loss.data)) + '\n')

                if (count % save) == 0:
                    print('save the model')
                    serializers.save_npz(save_root+'/models/' + str(count) + '.model', model)
                    print('save the optimizer')
                    serializers.save_npz(save_root+'/models/' + str(count) + '.state', optimizer)

                x_batch[0] = y_batch[0]
                count += 1

            seq = (seq + 1) % len(sequencelist)

    return prediction_error

if __name__== "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='PredNet')
    parser.add_argument('--images', '-i', default='', help='Path to image list file')
    parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-r', default='.',
                        help='Root directory path of sequence and image files')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--size', '-s', default='160,120',
                        help='Size of target images. width,height (pixels)')
    parser.add_argument('--channels', '-c', default='3,48,96,192',
                        help='Number of channels on each layers')
    parser.add_argument('--offset', '-o', default='0,0',
                        help='Center offset of clipping input image (pixels)')
    parser.add_argument('--input_len', '-l', default=50, type=int,
                        help='Input frame length fo extended prediction on test (frames)')
    parser.add_argument('--ext', '-e', default=10, type=int,
                        help='Extended prediction on test (frames)')
    parser.add_argument('--bprop', default=20, type=int,
                        help='Back propagation length (frames)')
    parser.add_argument('--save', default=10000, type=int,
                        help='Period of save model and state (frames)')
    parser.add_argument('--period', default=1000000, type=int,
                        help='Period of training (frames)')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    run_PredNet(args.images, args.sequences, args.gpu, args.root, args.initmodel, args.resume, args.size, \
                args.channels, args.offset, args.input_len, args.ext, args.bprop, args.save, args.period, args.test)
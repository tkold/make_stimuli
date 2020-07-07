import chainer
import os
import numpy as np

from tb_chainer import NodeName, graph

def save_dL(path,name1,node,name2=None): # save_decompose_LSTM
    if name2 is not None:
        savep=path+name1+'_'+node.creator_node.__class__.__name__+'_'+name2
    else:
        savep=path+name1+'_'+node.creator_node.__class__.__name__
    np.save(savep, chainer.cuda.to_cpu(node.data[-1, ...]))

def decompose_LSTM(node, path, Layer):
    save_dL(path,'R(h+1)', node,'o0,c0')#o0 * c0

    # o
    oo=node.creator_node.inputs[0]#o0 = sigmoid o1, input o
    save_dL(path,'o0', oo)
    oo=oo.creator_node.inputs[0]#o1 = o2 + c_o
    save_dL(path,'o1', oo, '+c_o')
    co=oo.creator_node.inputs[1]#c_o
    save_dL(path,'c_o', co)
    NTc=co.creator_node.inputs[0]#NoneType c
    save_dL(path,'c', NTc)
    oo=oo.creator_node.inputs[0]#o2 = o3 + h_o
    save_dL(path,'o2', oo, '+h_o')
    ho=oo.creator_node.inputs[1]#h_o
    save_dL(path,'h_o', ho)
    NTh=ho.creator_node.inputs[0]#NoneType h
    save_dL(path,'h', NTh)
    oo=oo.creator_node.inputs[0]#o3 = (x_o0+x_o1) or (x_o0)

    #c
    cc=node.creator_node.inputs[1]#c0 = tanh c1, input c
    save_dL(path,'c0', cc)
    cc=cc.creator_node.inputs[0]#c1 = c2 + f0
    save_dL(path,'c1', cc, '+f0')
    ff=cc.creator_node.inputs[1]#Input f0
    cc=cc.creator_node.inputs[0]#c2 = c3 * i0
    save_dL(path,'c2', cc, '*i0')
    ii=cc.creator_node.inputs[1]#Input i0
    cc=cc.creator_node.inputs[0]#c3 = tanh c4
    save_dL(path,'c3', cc)
    cc=cc.creator_node.inputs[0]#c4 = c5 + h_c
    save_dL(path,'c4', cc, '+h_c')
    hc=cc.creator_node.inputs[1]#h_c
    save_dL(path,'h_c', hc)
    cc=cc.creator_node.inputs[0]#c5 = (x_c0+x_c1) or (x_c0)

    #f
    save_dL(path,'f0', ff, '*NTc')#f0 = f1 * NTc
    ff=ff.creator_node.inputs[0]#f1 = sigmoid f2
    save_dL(path,'f1', ff)
    ff=ff.creator_node.inputs[0]#f2 = f3 + c_f
    save_dL(path,'f2', ff, '+c_f')
    cf=ff.creator_node.inputs[1]#c_f
    save_dL(path,'c_f', cf)
    ff=ff.creator_node.inputs[0]#f3 = f4 + h_f
    save_dL(path,'f3', ff, '+h_f')
    hf=ff.creator_node.inputs[1]#h_f
    save_dL(path,'h_f', hf)
    ff=ff.creator_node.inputs[0]#f4 = (x_f0+x_f1) or (x_f0)

    #i
    save_dL(path,'i0', ii)#i0 = sigmoid i1
    ii=ii.creator_node.inputs[0]#i1 = i2 + c_i
    save_dL(path,'i1', ii, '+c_i')
    ci=ii.creator_node.inputs[1]#c_i
    save_dL(path,'c_i', ci)
    ii=ii.creator_node.inputs[0]#i2 = i3 + h_i
    save_dL(path,'i2', ii, '+h_i')
    hi=ii.creator_node.inputs[1]#h_i
    save_dL(path,'h_i', hi)
    ii=ii.creator_node.inputs[0]#i3 = (x_i0+x_i1) or (x_i0)

    if Layer==0:
        save_dL(path, 'i3', ii, '+x_i1')
        save_dL(path, 'f4', ff, '+x_f1')
        save_dL(path, 'c5', cc, '+x_c1')
        save_dL(path, 'o3', oo, '+x_o1')

        i1 = ii.creator_node.inputs[1] # x_i1
        f1 = ff.creator_node.inputs[1] # x_f1
        c1 = cc.creator_node.inputs[1] # x_c1
        o1 = oo.creator_node.inputs[1] # x_o1
        save_dL(path, 'x_i1', i1)
        save_dL(path, 'x_f1', f1)
        save_dL(path, 'x_c1', c1)
        save_dL(path, 'x_o1', o1)
        ii = ii.creator_node.inputs[0] # x_i0
        ff = ff.creator_node.inputs[0] # x_f0
        cc = cc.creator_node.inputs[0] # x_c0
        oo = oo.creator_node.inputs[0] # x_o0

    save_dL(path, 'x_i0', ii)
    save_dL(path, 'x_f0', ff)
    save_dL(path, 'x_c0', cc)
    save_dL(path, 'x_o0', oo)

def get_nodename(node, root, node_name):
    path = root+'/'+node_name.name(node).split('/')[0]
    if not os.path.exists(path): os.mkdir(path)
    path = path+'/'
    if 'ConvLSTM' in node_name.name(node):
        if node.creator_node.__class__.__name__=='Mul':
            if node.creator_node.inputs[0].creator_node.__class__.__name__=='Sigmoid' and node.creator_node.inputs[1].creator_node.__class__.__name__=='Tanh':
                if 'R_Layer3' in path :Layer=1
                else: Layer=0
                decompose_LSTM(node, path, Layer)
    else:
        n=0
        sa=-1
        while sa < 0 :
            spath = path + str(node.creator_node.__class__.__name__) + '_' + str(n)
            if not os.path.exists(spath+'.npy'):
                np.save(spath, chainer.cuda.to_cpu(node.data[-1, ...]))
                sa=1
            n+=1
    return 0

def save_nodedata(model,save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    g=graph.build_computational_graph([model.y])
    node_name = NodeName(g.nodes)
    for n in g.nodes:
        if isinstance(n, chainer.variable.VariableNode) and \
          not isinstance(n._variable(), chainer.Parameter) and n.data is not None:
            get_nodename(n, save_root, node_name)
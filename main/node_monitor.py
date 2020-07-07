import chainer.computational_graph as c

from chainer import function_node
from chainer import variable
import numpy as np
import chainer
import os
import sys
from tb_chainer import graph
from recalc import node_analysis as na

# Note: docstrings must be updated when changing these default values.
_var_style = {'shape': 'box', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}

class DotNode(object):
    """Node of the computational graph, with utilities for dot language.
    This class represents a node of computational graph,
    with some utilities for dot language.
    Args:
        node: :class: `VariableNode` object or :class: `FunctionNode` object.
        attribute (dict): Attributes for the node.
        show_name (bool): If `True`, the `name` attribute of the node is added
            to the label. Default is `True`.
    """

    def __init__(self, node, path, attribute=None, show_name=True):
        assert isinstance(node, (variable.VariableNode,
                                 function_node.FunctionNode))
        self.node = node
        self.id_ = id(node)
        self.path = path
        if isinstance(node, variable.VariableNode):
            if show_name and node.name is not None:
                self.attribute = {'label': node.label}
                self.attribute['label'] = '{}: {}'.format(
                    node.name, self.attribute['label'])
            else:
                self.attribute = {'label': '', 'image': str(self.id_)+'.png'}
            self.attribute.update({'shape': 'oval'})
        else:
            self.attribute = {'label': node.label}
            self.attribute.update({'shape': 'box'})
        if attribute is not None:
            self.attribute.update(attribute)

    @property
    def label(self):
        """The text that represents properties of the node.
        Returns:
            string: The text that represents the id and attributes of this
                node.
        """
        if isinstance(self.node, variable.VariableNode) and self.node.data is not None and self.node.name is None:
            cnode = chainer.cuda.to_cpu(self.node.data[-1, ...])
            np.save(self.path+str(self.id_), cnode)
            na.plot_image(cnode, self.path+str(self.id_), str(self.id_))
        attributes = ['%s="%s"' % (k, v) for (k, v)
                      in self.attribute.items()]
        return '%s [%s];' % (str(self.id_), ','.join(attributes))

class ComputationalGraph(object):

    """Class that represents computational graph.
    .. note::
        We assume that the computational graph is directed and acyclic.
    Args:
        nodes (list): List of nodes. Each node is either
             :class:`VariableNode` object or :class:`FunctionNode` object.
        edges (list): List of edges. Each edge consists of pair of nodes.
        variable_style (dict or `'default'`): Dot node style for variable.
            If the special value ``'default'`` is specified, the default
            configuration will be used.
        function_style (dict or `default`): Dot node style for function.
            If the special value ``'default'`` is specified, the default
            configuration will be used.
        rankdir (str): Direction of the graph that must be
            TB (top to bottom), BT (bottom to top), LR (left to right)
            or RL (right to left).
        remove_variable (bool): If ``True``, :class:`VariableNode`\\ s are
            removed from the resulting computational graph. Only
            :class:`FunctionNode`\\ s are shown in the output.
        show_name (bool): If ``True``, the ``name`` attribute of each node is
            added to the label of the node. Default is ``True``.
    .. note::
       The default configuration for ``variable_style`` is
       ``{'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}`` and
       the default configuration for ``function_style`` is
       ``{'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}``.
    .. note::
        The default behavior of :class:`~chainer.ComputationalGraph` has been
        changed from v1.23.0, so that it ouputs the richest representation of
        a graph as default, namely, styles are set and names of functions and
        variables are shown. To reproduce the same result as previous versions
        (<= v1.22.0), please specify `variable_style=None`,
        `function_style=None`, and `show_name=False` explicitly.
    """

    def __init__(self, nodes, edges, path, variable_style='default',
                 function_style='default', rankdir='TB',
                 remove_variable=False, show_name=True):
        # If `variable_style` and `function_style` is explicitly set to None,
        # use legacy (Chainer v1.22.0) style for backward compatibility.
        if variable_style is None:
            variable_style = {}
        elif variable_style == 'default':
            variable_style = dict(_var_style)

        if function_style is None:
            function_style = {}
        elif function_style == 'default':
            function_style = dict(_func_style)

        self.nodes = nodes
        self.edges = edges
        self.path = path
        self.variable_style = variable_style
        self.function_style = function_style
        if rankdir not in ('TB', 'BT', 'LR', 'RL'):
            raise ValueError('rankdir must be in TB, BT, LR or RL.')
        self.rankdir = rankdir
        self.remove_variable = remove_variable
        self.show_name = show_name

    def _to_dot(self):
        """Converts graph in dot format.
        `label` property of is used as short description of each node.
        Returns:
            str: The graph in dot format.
        """
        ret = 'digraph graphname{rankdir=%s;' % self.rankdir

        if self.remove_variable:
            self.nodes, self.edges = _skip_variable(self.nodes, self.edges)

        for node in self.nodes:
            assert isinstance(node, (variable.VariableNode,
                                     function_node.FunctionNode))
            if isinstance(node, variable.VariableNode)and \
             not isinstance(node._variable(), chainer.Parameter):
                if not self.remove_variable:
                    ret += DotNode(
                        node, self.path, self.variable_style, self.show_name).label

        drawn_edges = []
        for edge in self.edges:
            head, tail = edge
            if (isinstance(tail, variable.VariableNode) and
                    isinstance(head, function_node.FunctionNode)):
                head_attr = self.variable_style
                tail_attr = self.variable_style
                tail_node = DotNode(tail, self.path, head_attr, self.show_name)
                for input in head.inputs:
                    if isinstance(input, variable.VariableNode)and \
                      not isinstance(input._variable(), chainer.Parameter)and\
                      hasattr(input,'name_scope'):
                        if tail.name_scope.split('/')[0]==input.name_scope.split('/')[0]:
                            head_node = DotNode(input, self.path, head_attr, self.show_name)
                            edge = (head_node.id_, tail_node.id_)
                            if edge in drawn_edges:
                                continue
                            ret += '%s -> %s;' % edge
                            drawn_edges.append(edge)
        ret += '}'
        return ret

    def dump(self, format='dot'):
        """Dumps graph as a text.
        Args:
            format(str): The graph language name of the output.
            Currently, it must be 'dot'.
        Returns:
            str: The graph in specified format.
        """
        if format == 'dot':
            return self._to_dot()
        raise NotImplementedError('Currently, only dot format is supported.')

def _skip_variable(nodes, edges):
    func_edges = []
    for edge_i, edge in enumerate(edges):
        head, tail = edge
        if isinstance(head, variable.VariableNode):
            if head.creator_node is not None:
                head = head.creator_node
            else:
                continue
        if isinstance(tail, variable.VariableNode):
            for node in nodes:
                if isinstance(node, function_node.FunctionNode):
                    for input_var in node.inputs:
                        if input_var is tail:
                            tail = node
                            break
                    if isinstance(tail, function_node.FunctionNode):
                        break
            else:
                continue
        func_edges.append((head, tail))
    return nodes, func_edges

def node_monitor(model, path, switch=1):
    #switch.. 0:show CG 1:show CG constructed by only nodeimage
    g=graph.build_computational_graph([model.y])

    unit_list=list()
    node_list=list()
    edge_list=list()

    os.makedirs(path) # node activation
    path+='/'

    for node in g.nodes:
        if hasattr(node,'name_scope'):
            uname=node.name_scope.split('/')[0]
            if uname not in unit_list:
                unit_list.append(uname)
                node_list.append(list())
                edge_list.append(list())
            node_list[unit_list.index(uname)].append(node)
    for edge in g.edges:
        if hasattr(edge[0],'name_scope') and hasattr(edge[1],'name_scope'):
            uname=[edge[0].name_scope.split('/')[0], edge[1].name_scope.split('/')[0]]
            if uname[0]==uname[1]:
                edge_list[unit_list.index(uname[0])].append(edge)


    for nn in range(len(unit_list)):
        hpath=path + unit_list[nn]
        if not os.path.exists(hpath):
            os.mkdir(hpath)
        hpath+='/'
        if switch==0: cg = c.ComputationalGraph(node_list[nn], edge_list[nn])
        else:         cg =   ComputationalGraph(node_list[nn], edge_list[nn],hpath)
        with open(hpath + unit_list[nn] + '.dot', 'w') as o:
            o.write(cg.dump())

    return 0

def organization_npyfile(experiment):

    for imagefolder in os.listdir(experiment+'/network'):
        if imagefolder!='images':
            for unitfolder in os.listdir(experiment+'/network/'+imagefolder):
                unitdir=experiment+'/network/'+imagefolder+'/'+unitfolder
                actunitdir=experiment+'/act/'+imagefolder+'/'+unitfolder
                folderlist=list()
                folderlist.append(actunitdir)
                actnodelist=list()
                while folderlist:
                    for dir in os.listdir(folderlist[0]):
                        dirpath=folderlist[0]+'/'+dir
                        if os.path.isdir(dirpath):
                            folderlist.append(dirpath)
                        elif  '.npy' in dir:
                            actnodelist.append(dirpath)
                    del folderlist[0]

                netnodelist=list()
                for file in os.listdir(unitdir):
                    filepath=unitdir+'/'+file
                    if  '.npy' in file:
                        netnodelist.append(filepath)

                with open(experiment+'/network/'+imagefolder + '/correspondence_table.txt', mode='a') as f:
                    while netnodelist:
                        for n in range(len(actnodelist)):
                            netnode=np.load(netnodelist[0])
                            actnode=np.load(actnodelist[n])
                            if netnode.shape==actnode.shape:
                                if np.allclose(netnode, actnode):
                                    f.write( netnodelist[0].split('/')[5] + ' = ' + actnodelist[n] + '\n')
                                    os.remove(netnodelist[0])
                                    del netnodelist[0]
                                    del actnodelist[n]
                                    break
                            if n==len(actnodelist)-1:
                                print(netnodelist[0] + ' is not in actnodelist.')
                                sys.exit(1)

                print(experiment, imagefolder, unitfolder, len(actnodelist))
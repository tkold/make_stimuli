import os
import util as util
import shutil
from PIL import Image

def make_general_graph(netroot):
    with open(netroot + '/correspondence_table.txt') as f:
        lines = f.readlines()
    dic = {}
    for l in range(len(lines)):
        txtline=lines[l].split('/')
        txt=txtline[len(txtline)-1].replace('.npy', '')
        txt=txt.replace('\n', '')
        dic[txtline[0].replace('.npy = runs', '')] = txt
    folderlist=util.get_folderlist(netroot)
    dotroot=netroot+'/general_graph'
    os.makedirs(dotroot)

    for f in range(len(folderlist)):
        folder=folderlist[f]
        dotfolder=dotroot+'/'+folder
        folder   =netroot+'/'+folder
        os.makedirs(dotfolder)
        flist=folder.split('/')
        with open(folder + '/' + flist[len(flist)-1]+'.dot', mode='r') as f:
            dotline = f.readline()
        for dir in os.listdir(folder):
            if '.png' in dir:
                nodeid=dir.replace('.png', '')
                dotline = dotline.replace('\"'+nodeid+'.png\"','\"'+dic[nodeid]+'.png\"')
                dotline = dotline.replace(nodeid, '\"'+dic[nodeid]+'\"')
        with open(dotfolder + '/' + flist[len(flist)-1]+'.dot', mode='w') as f:
            f.write(dotline)

def make_cgimage(rootpath, subfol=None, footer=None, chilfol=0, defi=0):

    netroot = '/network'
    if subfol is not None: 
        netroot  +=  '/' + subfol
    if not os.path.exists(rootpath + netroot): os.makedirs(rootpath + netroot)
    if subfol is not None:
        rootpath +=  '/'   + subfol
        netroot   =  '/..' + netroot
    if chilfol == 1:
        netroot   =  '/..' + netroot


    folder_list=util.get_folderlist(rootpath)
    
    for huname in folder_list:
        hunamefol=huname.split('/')[0]
        if os.path.exists('ComputationalGraph/' + hunamefol + '/' + hunamefol + '.dot'):
            dotdir = rootpath + '/' + huname
            

            with open('ComputationalGraph/' + hunamefol + '/' + hunamefol + '.dot') as f:
                dotline = f.readline()
            if chilfol==1 or defi==1:
                nodelist=list()
                for image in dotline.split('image=\"'):
                    nodelist.append(image.split('\"')[0])
                del nodelist[0]

                dirlist = os.listdir(dotdir)
                txt1 = 'rankdir=TB'
                txt2 = '\" [shape=\"box\",fillcolor=\"#E0E0E0\",style=\"filled\",label=\"'
                wn=1
                for node in nodelist:
                    if wn and node in dirlist:
                        w, h = Image.open(dotdir + '/' + node).size
                        dotline = dotline.replace(txt1, 'node [ height = ' + str(int(h/100)) + ', width = ' + str(int(w/100)) + ' ]; '+txt1)
                        wn=0
                    if not node in dirlist:
                        dotline = dotline.replace(',image=\"' + node + '\"', '')
                        ntxt='\"' + node.replace('.png','') + txt2
                        dotline = dotline.replace(ntxt, ntxt + node.split('/')[-1].replace('.png',''))
            if footer is not None:
                dotline = dotline.replace('.png\"', '_' + footer + '.png\"')
            with open(dotdir + '/' + hunamefol + '.dot', mode='w') as f:
                f.write(dotline)

            statement = 'cd \'' + dotdir + '\'&& '
            statement += 'dot -Tpng ' + hunamefol + '.dot -o \'..' + netroot + '/' + huname.replace('/', '_') + '.png\''
            os.system(statement)

  

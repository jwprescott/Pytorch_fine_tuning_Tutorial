# annotate_cxr8.py
# Annotate NIH CXR8 dataset. Many labels in the original dataset are inaccurate.

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Button
import time

data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative/train/1_images_infection'
out_dir = '/home/ubuntu/Desktop/torch-cxr8/relabeled_images'
label_file = '/home/ubuntu/Desktop/torch-cxr8/relabeled.csv'

# annotations
# 1: pneumonia
# 2: no pneumonia
# 3: maybe pneumonia (atelectasis, effusion, combination, etc)

files = os.listdir(data_dir)
dx = ('Pneumonia','Not pneumonia')
dxdict = dict.fromkeys(files)

img = Image.open(os.path.join(data_dir,files[0]))
    
fig = plt.figure(figsize=(15,15))
plt.axis('off')
l = plt.imshow(img)

axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.02, 0.7, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, dx)


def hzfunc(label):
    hzlist = ('Pneumonia','Not pneumonia')
    #ydata = hzdict[label]
    #l.set_ydata(ydata)
    #plt.draw()
radio.on_clicked(hzfunc)

class Index(object):
    ind = 0

    def next(self, event):
        currDx = radio.value_selected
        currFile = files[self.ind]
        dxdict[currFile] = currDx
        self.ind += 1
        newFile = files[self.ind]
        newDx = dxdict[newFile]
        img = Image.open(os.path.join(data_dir,newFile))
        l.set_data(img)
        if newDx == None:
            radio.set_active(0)
        else:
            radio.set_active(dx.index(newDx))
        plt.title(files[self.ind])
        plt.draw()

    def prev(self, event):
        currDx = radio.value_selected
        currFile = files[self.ind]
        dxdict[currFile] = currDx
        self.ind -= 1
        newFile = files[self.ind]
        newDx = dxdict[newFile]
        img = Image.open(os.path.join(data_dir,newFile))
        l.set_data(img)
        if newDx == None:
            radio.set_active(0)
        else:
            radio.set_active(dx.index(newDx))
        plt.title(files[self.ind])
        plt.draw()

callback = Index()
axprev = plt.axes([0.02, 0.6, 0.1, 0.075])
axnext = plt.axes([0.13, 0.6, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()
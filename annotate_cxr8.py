# annotate_cxr8.py
# Annotate NIH CXR8 dataset. Many labels in the original dataset are inaccurate.

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Button
import time
import csv

data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative/train/1_images_infection'
out_dir = '/home/ubuntu/Desktop/torch-cxr8/relabeled_images'
label_file = '/home/ubuntu/Desktop/torch-cxr8/relabeled.csv'

# annotations
# 1: pneumonia
# 2: no pneumonia
# 3: maybe pneumonia (atelectasis, effusion, combination, etc)

files = os.listdir(data_dir)
dx = ('None','Pneumonia','Not pneumonia')

if os.path.exists(label_file):
    with open(label_file, 'rt') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        dxdict = dict(reader)
        for key,value in dxdict.items():
            if value == '':
                dxdict[key] = None            
else:
    dxdict = dict.fromkeys(files)


fig = plt.figure(figsize=(15,15))
plt.axis('off')
img = Image.open(os.path.join(data_dir,files[0]))
l = plt.imshow(img)
plt.title(files[0])

axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.02, 0.7, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, dx)


def hzfunc(label):
    hzlist = ('None','Pneumonia','Not pneumonia')
    #ydata = hzdict[label]
    #l.set_ydata(ydata)
    #plt.draw()
radio.on_clicked(hzfunc)

class Index(object):
    ind = 0

    def next(self, event):
        currDx = radio.value_selected
        currFile = files[self.ind]
        if currDx == 'None':
            dxdict[currFile] = None
        else:
            dxdict[currFile] = currDx        
        self.ind += 1
        newFile = files[self.ind]
        newDx = dxdict[newFile]
        img = Image.open(os.path.join(data_dir,newFile))
        l.set_data(img)
        l.axes.set_title(files[self.ind])
        if newDx == None:
            radio.set_active(0)
        else:
            radio.set_active(dx.index(newDx))        
        plt.draw()

    def prev(self, event):
        currDx = radio.value_selected
        currFile = files[self.ind]
        if currDx == 'None':
            dxdict[currFile] = None
        else:
            dxdict[currFile] = currDx
        self.ind -= 1
        newFile = files[self.ind]
        newDx = dxdict[newFile]
        img = Image.open(os.path.join(data_dir,newFile))
        l.set_data(img)
        l.axes.set_title(files[self.ind])
        if newDx == None:
            radio.set_active(0)
        else:
            radio.set_active(dx.index(newDx))        
        plt.draw()
        
    def save(self, event):
        with open(label_file, 'wt') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Image Index","Finding Labels"])
            for key, value in dxdict.items():
                writer.writerow([key, value])
        

callback = Index()
axprev = plt.axes([0.02, 0.6, 0.1, 0.075])
axnext = plt.axes([0.13, 0.6, 0.1, 0.075])
axsave = plt.axes([0.02, 0.5, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
bsave = Button(axsave, 'Save')
bsave.on_clicked(callback.save)

plt.show()
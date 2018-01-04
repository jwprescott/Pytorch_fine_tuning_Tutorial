# annotate_cxr8.py
# Annotate NIH CXR8 dataset. Many labels in the original dataset are inaccurate.

import os
from PIL import Image
import matplotlib.pyplot as plt
import time

data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative/train/1_images_infection'
out_dir = '/home/ubuntu/Desktop/torch-cxr8/relabeled_images'
label_file = '/home/ubuntu/Desktop/torch-cxr8/relabeled.csv'

# annotations
# 1: pneumonia
# 2: no pneumonia
# 3: maybe pneumonia (atelectasis, effusion, combination, etc)

files = os.listdir(data_dir)
for file in files[0:1]:
#for file in os.listdir(data_infection_dir):
    img = Image.open(os.path.join(data_dir,file))
    #plt.axis('off')
    plt.pause(0.001)
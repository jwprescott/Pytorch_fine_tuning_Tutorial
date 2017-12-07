# From CXR8 dataset, find images labeled with infiltration or pneumonia (infection)
# and those that aren't labeled as such. Then convert corresponding images to RGB
# and save as jpeg to infection/not_infection directories, for use as input for
# retraining neural network.

# TODO: Redownload images_008, because original download was corrupted:
# https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

# TODO: Separate subsets into training and validation, don't include same patient
# in pneumonia and not pneumonia diagnostic directories

import os
import sys
import csv
from shutil import copyfile
from shutil import rmtree
from PIL import Image
import numpy as np

DATA_DIR = "/home/ubuntu/Desktop/CXR8"
OUT_DIR = "/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative"

# Clean out, recreate output directories
if os.path.exists(os.path.join(OUT_DIR,'images_infection')):
    rmtree(os.path.join(OUT_DIR,'images_infection'))
os.makedirs(os.path.join(OUT_DIR,'images_infection'))

if os.path.exists(os.path.join(OUT_DIR,'images_not_infection')):
    rmtree(os.path.join(OUT_DIR,'images_not_infection'))
os.makedirs(os.path.join(OUT_DIR,'images_not_infection'))


datafile = open(os.path.join(DATA_DIR,'Data_Entry_2017.csv'),'rt')

reader = csv.reader(datafile,delimiter=',')
_ = next(reader)   #skip header

count_total = 0
count_infection = 0
count_not_infection = 0
sum_m = 0
sum_std = 0
num_cases = 1000
l_infection = []
l_not_infection = []
for row in reader:
    flag_infection = 0
    for dx in row[1].split('|'):
        #if dx == 'Pneumonia' or dx == "Infiltration":
        if dx == 'Pneumonia':
            if count_infection < num_cases:
                l_infection.append(row[0])
                print(row[0] + ", " + dx)
                count_infection = count_infection + 1
                if os.path.isfile(os.path.join(DATA_DIR,'images',row[0])):
                    img = Image.open(os.path.join(DATA_DIR,'images',row[0]))
                    rgbimg = Image.new("RGB", img.size)
                    rgbimg.paste(img)
                    rgbimg.save(os.path.join(OUT_DIR,'images_infection',row[0][:-4]+'.jpg'))
                    #copyfile(os.path.join(DATA_DIR,"images",row[0]),
                            #os.path.join(OUT_DIR,"images_infection",row[0]))
                else:
                    print("File " + row[0] + " does not exist")
                flag_infection = 1
            break
    if not flag_infection:
        if dx == 'No Finding':
            if count_not_infection < num_cases:
                l_not_infection.append(row[0])
                count_not_infection = count_not_infection + 1
                if os.path.isfile(os.path.join(DATA_DIR,'images',row[0])):
                    img = Image.open(os.path.join(DATA_DIR,'images',row[0]))
                    rgbimg = Image.new("RGB", img.size)
                    rgbimg.paste(img)
                    rgbimg.save(os.path.join(OUT_DIR,'images_not_infection',row[0][:-4]+'.jpg'))
                    #copyfile(os.path.join(DATA_DIR,"images",row[0]),
                            #os.path.join(OUT_DIR,"images_not_infection",row[0]))
                else:
                    print("File " + row[0] + " does not exist")
    if count_infection >= num_cases and count_not_infection >= num_cases:
        break
    
    ## calculate mean, std (wasteful image load but whatever)
    #img = Image.open(os.path.join(DATA_DIR,'images',row[0]))
    #rgbimg = Image.new("RGB", img.size)
    #rgbimg.paste(img)
    #m = np.mean(rgbimg,axis=(0,1))/255
    #std = np.std(rgbimg,axis=(0,1))/255
    #sum_m = sum_m + m
    #sum_std = sum_std + std
    #count_total = count_total + 1

datafile.close()

#total_m = sum_m / count_total
#total_std = sum_std / count_total

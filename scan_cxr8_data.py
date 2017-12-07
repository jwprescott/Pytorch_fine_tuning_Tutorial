# From CXR8 dataset, find images labeled with infiltration or pneumonia (infection)
# and those that aren't labeled as such. Then convert corresponding images to RGB
# and save as jpeg to infection/not_infection directories, for use as input for
# retraining neural network.

import os
import sys
import csv
from shutil import copyfile
from shutil import rmtree
from PIL import Image

DATA_DIR = "/home/ubuntu/Desktop/CXR8"

# Clean out, recreate output directories
if os.path.exists(os.path.join(DATA_DIR,'images_infection')):
    rmtree(os.path.join(DATA_DIR,'images_infection'))
os.makedirs(os.path.join(DATA_DIR,'images_infection'))

if os.path.exists(os.path.join(DATA_DIR,'images_not_infection')):
    rmtree(os.path.join(DATA_DIR,'images_not_infection'))
os.makedirs(os.path.join(DATA_DIR,'images_not_infection'))


datafile = open(os.path.join(DATA_DIR,'Data_Entry_2017.csv'),'rb')

reader = csv.reader(datafile,delimiter=',')
reader.next()   #skip header

count_infection = 0
count_not_infection = 0
num_cases = 100
l_infection = []
l_not_infection = []
for row in reader:
    flag_infection = 0
    for dx in row[1].split('|'):
        if dx == 'Pneumonia' or dx == "Infiltration":
            if count_infection < num_cases:
                l_infection.append(row[0])
                print row[0] + ", " + dx
                count_infection = count_infection + 1
                img = Image.open(os.path.join(DATA_DIR,'images',row[0]))
                rgbimg = Image.new("RGB", img.size)
                rgbimg.paste(img)
                rgbimg.save(os.path.join(DATA_DIR,'images_infection',row[0][:-4],'.jpg'))
                #copyfile(os.path.join(DATA_DIR,"images",row[0]),
                         #os.path.join(DATA_DIR,"images_infection",row[0]))
                flag_infection = 1
            break
    if not flag_infection:
        if count_not_infection < num_cases:
            l_not_infection.append(row[0])
            count_not_infection = count_not_infection + 1
            img = Image.open(os.path.join(DATA_DIR,'images',row[0]))
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            rgbimg.save(os.path.join(DATA_DIR,'images_not_infection',row[0][:-4],'.jpg'))
            #copyfile(os.path.join(DATA_DIR,"images",row[0]),
                     #os.path.join(DATA_DIR,"images_not_infection",row[0]))
    if count_infection >= 100 and count_not_infection >= 100:
        break

datafile.close()

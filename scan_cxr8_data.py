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
import random

DATA_DIR = "/home/ubuntu/Desktop/CXR8"
OUT_DIR = "/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative"

# Clean out, recreate output directories
if os.path.exists(os.path.join(OUT_DIR,'images_infection')):
    rmtree(os.path.join(OUT_DIR,'images_infection'))
os.makedirs(os.path.join(OUT_DIR,'train/images_infection'))
os.makedirs(os.path.join(OUT_DIR,'val/images_infection'))
os.makedirs(os.path.join(OUT_DIR,'test/images_infection'))

if os.path.exists(os.path.join(OUT_DIR,'images_not_infection')):
    rmtree(os.path.join(OUT_DIR,'images_not_infection'))
os.makedirs(os.path.join(OUT_DIR,'train/images_not_infection'))
os.makedirs(os.path.join(OUT_DIR,'val/images_not_infection'))
os.makedirs(os.path.join(OUT_DIR,'test/images_not_infection'))

datafile = open(os.path.join(DATA_DIR,'Data_Entry_2017.csv'),'rt')

reader = csv.reader(datafile,delimiter=',')
_ = next(reader)   #skip header

count_total = 0
count_infection = 0
count_not_infection = 0
sum_m = 0
sum_std = 0
num_cases = 10
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

####################################################
# Separate images into train, validation, and test datasets, without patient overlap
#

train_percent = 0.93
val_percent = 0.05
test_percent = 0.02

#train_percent = 0.8
#val_percent = 0.1
#test_percent = 0.1

file_infection = []
file_not_infection = []
pt_infection = []
pt_not_infection = []

flag_infection = 0

datafile = open(os.path.join(DATA_DIR,'Data_Entry_2017.csv'),'rt')

reader = csv.reader(datafile,delimiter=',')
_ = next(reader)   #skip header

debug_count = 0
for row in reader:
    flag_infection = 0
    for dx in row[1].split('|'):
        if dx == 'Pneumonia':
            file_infection.append(row[0])
            pt_infection.append(row[3])
            flag_infection = 1
            break
    if not flag_infection:
        file_not_infection.append(row[0])
        pt_not_infection.append(row[3])
    debug_count = debug_count + 1
    #if debug_count > 1000:
        #break

datafile.close()

# Find unique patients in datasets, intersection between datasets
unique_pt_infection = list(set(pt_infection))
unique_pt_not_infection = list(set(pt_not_infection))

# Patients who have images without infection who also have images with infection
intersect_pt_infection_not_infection = list(set.intersection(set(pt_infection),set(pt_not_infection)))

# Patients who only have images without infection
pt_not_infection_only = list(set(unique_pt_not_infection).symmetric_difference(unique_pt_infection))

num_infection_pt_train = round(len(unique_pt_infection) * train_percent)
num_infection_pt_val = round(len(unique_pt_infection) * val_percent)
num_infection_pt_test = len(unique_pt_infection) - num_infection_pt_train - num_infection_pt_val

# Percent of pneumonia images versus not pneumonia images
percent_infection = len(file_infection) / (len(file_infection) + len(file_not_infection))

# Only a small percent of patients have images with pneumonia (~ 1%), so make sure
# these patients are adequately distributed between train, validation, and test sets
infection_pt_train = unique_pt_infection[0:num_infection_pt_train]
infection_pt_val = unique_pt_infection[num_infection_pt_train:
                                           num_infection_pt_train + num_infection_pt_val]
infection_pt_test = unique_pt_infection[-(num_infection_pt_test):]


num_not_infection_pt_train = round(len(pt_not_infection_only) * train_percent)
num_not_infection_pt_val = round(len(pt_not_infection_only) * val_percent)
num_not_infection_pt_test = len(pt_not_infection_only) - num_not_infection_pt_train - num_not_infection_pt_val

not_infection_pt_train = pt_not_infection_only[0:num_not_infection_pt_train]
not_infection_pt_val = pt_not_infection_only[num_not_infection_pt_train:
                                           num_not_infection_pt_train +
                                           num_not_infection_pt_val]
not_infection_pt_test = pt_not_infection_only[-(num_not_infection_pt_test):]


pt_infection_image_file_train = []
pt_not_infection_image_file_train = []
pt_infection_image_file_val = []
pt_not_infection_image_file_val = []
pt_infection_image_file_test = []
pt_not_infection_image_file_test = []

for p in infection_pt_train:    
    for i, x in enumerate(pt_infection):
        if x == p:
            pt_infection_image_file_train.append(file_infection[i])
    for i, x in enumerate(pt_not_infection):
        if x == p:
            pt_not_infection_image_file_train.append(file_not_infection[i])
            
for p in not_infection_pt_train:    
    for i, x in enumerate(pt_not_infection):
        if x == p:
            if not file_not_infection[i] in pt_not_infection_image_file_train:
                pt_not_infection_image_file_train.append(file_not_infection[i])
                
for p in infection_pt_val:    
    for i, x in enumerate(pt_infection):
        if x == p:
            pt_infection_image_file_val.append(file_infection[i])
    for i, x in enumerate(pt_not_infection):
        if x == p:
            pt_not_infection_image_file_val.append(file_not_infection[i])
            
for p in not_infection_pt_val:    
    for i, x in enumerate(pt_not_infection):
        if x == p:
            if not file_not_infection[i] in pt_not_infection_image_file_val:
                pt_not_infection_image_file_val.append(file_not_infection[i])
                
for p in infection_pt_test:    
    for i, x in enumerate(pt_infection):
        if x == p:
            pt_infection_image_file_test.append(file_infection[i])
    for i, x in enumerate(pt_not_infection):
        if x == p:
            pt_not_infection_image_file_test.append(file_not_infection[i])
            
for p in not_infection_pt_test:    
    for i, x in enumerate(pt_not_infection):
        if x == p:
            if not file_not_infection[i] in pt_not_infection_image_file_test:
                pt_not_infection_image_file_test.append(file_not_infection[i])


print("Number of patients")
print("Infection (Train, Validation, Test): " + str(num_infection_pt_train) +
      ", " + str(num_infection_pt_val) + ", " + str(num_infection_pt_test))
print("Not infection (Train, Validation, Test): " + str(num_not_infection_pt_train) +
      ", " + str(num_not_infection_pt_val) + ", " + str(num_not_infection_pt_test))
print("")
print("Number of images")
print("Infection (Train, Validation, Test): " + str(len(pt_infection_image_file_train)) +
      ", " + str(len(pt_infection_image_file_val)) + ", " +
      str(len(pt_infection_image_file_test)))
print("Not infection (Train, Validation, Test): " + 
      str(len(pt_not_infection_image_file_train)) + ", " + 
      str(len(pt_not_infection_image_file_val)) + ", " + 
      str(len(pt_not_infection_image_file_test)))
print("")

#Number of patients
#Infection (Train, Validation, Test): 888, 48, 19
#Not infection (Train, Validation, Test): 27828, 1496, 599

#Number of images
#Infection (Train, Validation, Test): 1276, 56, 21
#Not infection (Train, Validation, Test): 103245, 5170, 2352


# TODO: Make sure no overlap in patients between train, validation, and test sets,
# in addition to making sure not file overlap.
# for example:
# if set.intersection(set(infection_pt_train),set(infection_pt_test)):
#       print("The same patient is in both train and test sets")

                
for f in pt_infection_image_file_train:
    img = Image.open(os.path.join(DATA_DIR,'images',f))
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(os.path.join(OUT_DIR,'train/images_infection',f[:-4]+'.jpg'))
    
for f in pt_not_infection_image_file_train:
    img = Image.open(os.path.join(DATA_DIR,'images',f))
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(os.path.join(OUT_DIR,'train/images_not_infection',f[:-4]+'.jpg'))
    
for f in pt_infection_image_file_val:
    img = Image.open(os.path.join(DATA_DIR,'images',f))
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(os.path.join(OUT_DIR,'val/images_infection',f[:-4]+'.jpg'))
    
for f in pt_not_infection_image_file_val:
    img = Image.open(os.path.join(DATA_DIR,'images',f))
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(os.path.join(OUT_DIR,'val/images_not_infection',f[:-4]+'.jpg'))
    
for f in pt_infection_image_file_test:
    img = Image.open(os.path.join(DATA_DIR,'images',f))
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(os.path.join(OUT_DIR,'test/images_infection',f[:-4]+'.jpg'))
    
for f in pt_not_infection_image_file_test:
    img = Image.open(os.path.join(DATA_DIR,'images',f))
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg.save(os.path.join(OUT_DIR,'test/images_not_infection',f[:-4]+'.jpg'))
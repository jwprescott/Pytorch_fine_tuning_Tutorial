
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

from pytorch_logger import Logger

import io
import requests
from PIL import Image
from torch.nn import functional as F
import cv2

from sklearn.metrics import roc_curve, auc

# Resize, normalize for training and validation
# Mean and standard deviation calculated from 5862 images from CXR8 dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative_small_set_unbalanced'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

class_weights = torch.FloatTensor([0.9,0.1])

#if use_gpu:
    #class_weights = class_weights.cuda()
    

#def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
def train_model(model, criterion, optimizer, num_epochs=25):

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()   # for lr_scheduler.StepLR
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                #weights = torch.FloatTensor([class_weights[x] for x in labels]).unsqueeze(1)
                weights = torch.FloatTensor([class_weights[x] for x in labels])
                #labels = labels.float().unsqueeze(1)
                #labels = labels.unsqueeze(1)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    weights = weights.cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                criterion.weight = weights

                # forward
                outputs = model(inputs)
                #_, preds = torch.max(outputs.data, 1)
                outputs_sig = F.sigmoid(outputs).squeeze()
                #outputs_sig = 1 - outputs_sig
                preds = torch.round(outputs_sig).long()
                #preds = (1 - torch.round(outputs_sig)).long()
                loss = criterion(outputs_sig, labels.float())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds==labels).long().data.cpu().numpy()[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
            ## Update optimizer learning rate
            #if phase == 'val':
                #scheduler.step(epoch_loss, epoch)
                
            # TODO: Save checkpoints
            
                
            ##============ TensorBoard logging ============#
            ## (1) Log the scalar values
            #if phase == 'val':
                #info = {
                    #'loss': epoch_loss,
                    #'accuracy': epoch_acc
                    #}
                
                #for tag, value in info.items():
                    #logger.scalar_summary(tag, value, epoch)
                    
                ## (2) Log values and gradients of the parameters (histogram)
                #for tag, value in model_ft.named_parameters():
                    #tag = tag.replace('.', '/')
                    #logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                    #logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
                        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.densenet121(pretrained=True)
num_ftrs = model_ft.classifier.in_features	# for densenet
model_ft.classifier = nn.Linear(num_ftrs, 1) # for densenet

if use_gpu:
    model_ft = model_ft.cuda()
   
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

optimizer_ft = optim.Adam(model_ft.parameters())

## Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', factor=0.1,
                                                  #patience=7, min_lr=0.5e-6)

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       #num_epochs=50)
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=50)


# Debugging code to run from command line after train_model fails
phase = 'train'
class_weights = torch.FloatTensor([0.8,0.2])
data = next(iter(dataloaders[phase]))
inputs, labels = data
#weights = torch.FloatTensor([class_weights[x] for x in labels]).unsqueeze(1)
weights = torch.FloatTensor([class_weights[x] for x in labels])
inputs = Variable(inputs.cuda())
labels = Variable(labels.cuda())
weights = weights.cuda()
criterion.weight = weights
outputs = model_ft(inputs)
outputs_sig = F.sigmoid(outputs).squeeze()
preds = (1-torch.round(outputs_sig)).long()
loss = criterion(outputs_sig, labels.float())
running_loss = 0.0
running_corrects = 0
running_labels = np.array([])
running_preds = np.array([])
running_outputs_sig = np.array([])
running_loss += loss.data[0]
running_corrects += torch.sum(preds==labels).long().data.cpu().numpy()[0]
#running_labels.append(labels.long().data.cpu().numpy())
#running_preds.append(preds.data.cpu().numpy())
#running_outputs_sig.append(outputs_sig.data.cpu().numpy())
running_labels = np.concatenate([running_labels,labels.long().data.cpu().numpy()])
running_preds = np.concatenate([running_preds,preds.data.cpu().numpy()])
running_outputs_sig = np.concatenate([running_outputs_sig,outputs_sig.data.cpu().numpy()])
epoch_loss = running_loss / dataset_sizes[phase]
epoch_acc = running_corrects/dataset_sizes[phase]
print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

# Debug: Calculate validation accuracy for data with pretrained densenet model
model_ft = models.densenet121(pretrained=True)
num_ftrs = model_ft.classifier.in_features	# for densenet
model_ft.classifier = nn.Linear(num_ftrs, 1) # for densenet
model_ft = model_ft.cuda()
phase = 'val'
model_ft.train(False)
data = next(iter(dataloaders[phase]))
inputs, labels = data
weights = torch.FloatTensor([class_weights[x] for x in labels])
inputs = Variable(inputs.cuda())
labels = Variable(labels.cuda())
weights = weights.cuda()
criterion.weight = weights
outputs = model_ft(inputs)
outputs_sig = F.sigmoid(outputs).squeeze()
preds = torch.round(outputs_sig).long()
loss = criterion(outputs_sig, labels.float())
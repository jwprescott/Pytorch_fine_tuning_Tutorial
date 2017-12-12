# License: BSD
# Author: Sasank Chilamkurthy

# from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Start tensorboard from bash command line: tensorboard --logdir='./logs' --port=6006

# Best performance so far: use ImageNet mean, std; random horizontal flip for training
# images, loss rate 0.1 per 7 epochs, 20 epochs, pneumonia vs no finding

# TODO: Output all class activation maps for validation set
# TODO: Calculate ROC curves and area under ROC

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


plt.ion()   # interactive mode

# Resize, normalize for training and validation
# Mean and standard deviation calculated from 5862 images from CXR8 dataset
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomSizedCrop(224),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # CXR8 mean, std
        #transforms.Normalize([0.518, 0.518, 0.518], [0.231, 0.231, 0.231])
        # ImageNet mean, std
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        # CXR8 mean, std
        #transforms.Normalize([0.518, 0.518, 0.518], [0.231, 0.231, 0.231])
        # ImageNet mean, std
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = '/home/ubuntu/Desktop/torch-hemorrhage/images_curated'
data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative_train_val'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


## Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

## Make a grid from batch
#out = torchvision.utils.make_grid(inputs)
#imshow(out, title=[class_names[x] for x in classes])

# Setup logging
logger = Logger('./logs')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#def train_model(model, criterion, optimizer, num_epochs=25):
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

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
            # Update optimizer leraning rate
            if phase == 'val':
                scheduler.step(epoch_loss, epoch)
                
            # TODO: Save checkpoints
            
                
            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            if phase == 'val':
                info = {
                    'loss': epoch_loss,
                    'accuracy': epoch_acc
                    }
                
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch)
                    
                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model_ft.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
                        
                ## (3) Log the images
                #info = {
                    #'images': to_np(images.view(-1, 28, 28)[:10])
                    #}
                
                #for tag, images in info.items():
                    #logger.image_summary(tag, images, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            if use_gpu:
                ax.set_title('predicted: {}'.format(class_names[Variable(preds.cuda()).cpu().data.numpy()[j][0]]))
            else:
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


#model_ft = models.resnet18(pretrained=True)
model_ft = models.densenet121(pretrained=True)
#num_ftrs = model_ft.fc.in_features	# for resnet
#model_ft.fc = nn.Linear(num_ftrs, 2)	# for resnet
num_ftrs = model_ft.classifier.in_features	# for densenet
model_ft.classifier = nn.Linear(num_ftrs, 2) # for densenet


if use_gpu:
    model_ft = model_ft.cuda()
    #model_ft = torch.nn.DataParallel(model_ft).cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters())

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', factor=0.1,
                                                  patience=7, min_lr=0.5e-6)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

# Save final model
torch.save(model_ft, 'model_ft.pt')

# Load saved model
#model_ft = torch.load('model_ft_10_epochs.pt')

#model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=20)

# Create class activation map (CAM) for test image
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# DenseNet final conv name is 'features'
model_ft.features.register_forward_hook(hook_feature)

# get the softmax weight
params = list(model_ft.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (1024, 1024)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

#img_pil = Image.open(os.path.join(data_dir,'val/images_infection/00000091_004.jpg'))
img_pil = Image.open('/home/ubuntu/Desktop/torch-cxr8/images_100/val/images_infection/00000091_004.jpg')
img_pil.save('00000091_004.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = model_ft(img_variable.cuda())

h_x = F.sigmoid(logit).data.squeeze()
probs, idx = h_x.sort(0, True)


# output the prediction
for i in range(0, 2):
    print('{:.3f} -> {}'.format(probs[i], class_names[idx[i]]))
    
#if use_gpu:
                #ax.set_title('predicted: {}'.format(class_names[Variable(preds.cuda()).cpu().data.numpy()[j][0]]))
            #else:
                #ax.set_title('predicted: {}'.format(class_names[preds[j]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
#print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('00000091_004.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)


#visualize_model(model_ft)

#plt.ioff()
#plt.show()


# PREDICTION
data_infection_dir = '/home/ubuntu/Desktop/torch-cxr8/images_1000/val/images_infection/'
data_not_infection_dir = '/home/ubuntu/Desktop/torch-cxr8/images_1000/val/images_not_infection/'
image_files = []
diagnosis = []
for file in os.listdir(data_infection_dir):
    if file.endswith(".jpg"):
        image_files.append(os.path.join(data_infection_dir, file))
        diagnosis.append("images_infection")
        
for file in os.listdir(data_not_infection_dir):
    if file.endswith(".jpg"):
        image_files.append(os.path.join(data_not_infection_dir, file))
        diagnosis.append("images_not_infection")

true_dx = np.array([])
true_dx_prob = np.array([])
infection_dx_prob = np.array([])
for i in range(0,len(image_files)):
    img_pil = Image.open(image_files[i])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = model_ft(img_variable.cuda())

    h_x = F.sigmoid(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)
    
    pred = class_names[idx[0]]
    print(pred)
    
    if(diagnosis[i] == "images_infection"):
        true_dx = np.append(true_dx,[0])
        #true_dx_prob = np.append(true_dx_prob,[h_x[0]])
    if(diagnosis[i] == "images_not_infection"):
        true_dx = np.append(true_dx,[1])
        #true_dx_prob = np.append(true_dx_prob,[h_x[1]])
        
    infection_dx_prob = np.append(infection_dx_prob,[h_x[0]])
        
# Compute and plot ROC curves
fpr, tpr, _ = roc_curve(true_dx, infection_dx_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

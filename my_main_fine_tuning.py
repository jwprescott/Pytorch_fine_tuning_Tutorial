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
from shutil import rmtree, copyfile

from pytorch_logger import Logger

import io
import requests
from PIL import Image
from torch.nn import functional as F
import cv2

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix


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

#datafile = open(os.path.join(DATA_DIR,'Data_Entry_2017.csv'),'rt')

#data_dir = '/home/ubuntu/Desktop/torch-hemorrhage/images_curated'
data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative'
#data_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative_small_set_unbalanced'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# set number of GPU device used {0,1,2,3}; that way can run program with different parameters
# on different GPUs at the same time
gpu_num = 1

# Create output dir and output files based on GPU number
output_dir = 'output_{}'.format(gpu_num)

if os.path.exists(output_dir):
    rmtree(output_dir)
os.makedirs(output_dir)

train_stats_file = os.path.join(output_dir,'train_stats.txt')
val_stats_file = os.path.join(output_dir,'val_stats.txt')
datafile = open(train_stats_file,'w')
datafile.close()
datafile = open(val_stats_file,'w')
datafile.close()

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
logger = Logger(os.path.join(output_dir,'logs'))


class_weights = torch.FloatTensor([0.015,0.985])
#class_weights = torch.FloatTensor([0.1,0.9])

#if use_gpu:
    #class_weights = class_weights.cuda(gpu_num)
    

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(output_dir,filename))
    if is_best:
        copyfile(filename, os.path.join(output_dir,'model_best.pth.tar'))
        
## To load checkpoint
#checkpoint = torch.load('checkpoint.pth.tar')
#start_epoch = checkpoint['epoch']
#best_acc = checkpoint['best_acc']
#model_ft = models.densenet121(pretrained=True)
#num_ftrs = model_ft.classifier.in_features	# for densenet
#model_ft.classifier = nn.Linear(num_ftrs, 1) # for densenet
#model_ft = model_ft.cuda(gpu_num)
#model_ft.load_state_dict(checkpoint['state_dict'])

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
            running_labels = np.array([])
            running_preds = np.array([])
            running_outputs_sig = np.array([])

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
                    inputs = Variable(inputs.cuda(gpu_num))
                    labels = Variable(labels.cuda(gpu_num))
                    weights = weights.cuda(gpu_num)
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
                running_labels = np.concatenate([running_labels,labels.long().data.cpu().numpy()])
                running_preds = np.concatenate([running_preds,preds.data.cpu().numpy()])
                running_outputs_sig = np.concatenate([running_outputs_sig,outputs_sig.data.cpu().numpy()])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            fpr, tpr, _ = roc_curve(running_labels, running_outputs_sig)
            roc_auc = auc(fpr, tpr)
            
            tn, fp, fn, tp = confusion_matrix(running_labels,running_preds).ravel()
            
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            fpr = fp / (fp + tn)
            fnr = fn / (tp + fn)
            fdr = fp / (tp + fp)
            
            f1_binary = f1_score(running_labels, running_preds, pos_label = 1,
                          average = 'binary')
            f1_weighted = f1_score(running_labels, running_preds, pos_label = 1,
                          average = 'weighted')

            print('{}, Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}, Sens: {:.4f}, Spec: {:.4f}, PPV: {:.4f}, NPV: {:.4f}, FPR: {:.4f}, FNR: {:.4f}, FDR: {:.4f}, F1 binary: {:.4f}, F1 weighted: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, roc_auc, sens, spec, ppv, npv, fpr, fnr, fdr, f1_binary, f1_weighted))

            # Save checkpoints
            #if phase == 'val' and epoch_acc > best_acc:
                #best_acc = epoch_acc
                #best_model_wts = model.state_dict()
            if phase == 'val':
                is_best = epoch_acc > best_acc
                best_acc = max(epoch_acc, best_acc)
                if is_best:
                    best_model_wts = model.state_dict()
                    
                save_checkpoint({
                    'epoch': epoch + 1,
                    #'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    }, is_best, output_dir)
                
            # Update optimizer leraning rate
            if phase == 'val':
                scheduler.step(epoch_loss, epoch)
                
            #============ Logging ============#
            # (1) Log the scalar values
            if phase == 'train':
                info = {
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_acc
                    }
                
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch)
                    
                with open(train_stats_file,'a') as datafile:
                    datafile.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, epoch_loss, epoch_acc, roc_auc, sens, spec, ppv, npv, fpr, fnr, fdr, f1_binary, f1_weighted))
                                        
            if phase == 'val':
                info = {
                    'val_loss': epoch_loss,
                    'val_accuracy': epoch_acc
                    }
                
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch)
                    
                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model_ft.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
                    
                # (3) Write loss and accuracy to file
                with open(val_stats_file,'a') as datafile:
                    datafile.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch, epoch_loss, epoch_acc, roc_auc, sens, spec, ppv, npv, fpr, fnr, fdr, f1_binary, f1_weighted))
                    
                # (4) Output ROC curve               
                fpr, tpr, _ = roc_curve(running_labels, running_outputs_sig)
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
                #plt.show()
                plt.savefig(os.path.join(output_dir,'ROC_{}.png'.format(epoch)),bbox_inches='tight')
                plt.close()
                        
                ## (5) Log the images
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
            inputs, labels = Variable(inputs.cuda(gpu_num)), Variable(labels.cuda(gpu_num))
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            if use_gpu:
                ax.set_title('predicted: {}'.format(class_names[Variable(preds.cuda(gpu_num)).cpu().data.numpy()[j][0]]))
            else:
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

# TODO: change outputs to 1, use BCELoss and create weights for all batches

model_ft = models.densenet121(pretrained=True)
num_ftrs = model_ft.classifier.in_features	# for densenet
model_ft.classifier = nn.Linear(num_ftrs, 1) # for densenet

if use_gpu:
    model_ft = model_ft.cuda(gpu_num)
    #model_ft = torch.nn.DataParallel(model_ft).cuda(gpu_num)

# Use WEIGHTED BINARY CROSS ENTROPY, with SIGMOID NONLINEARITY applied to fully
# connected output layer

#For example, this is quite easy to do with neural networks as you just modify your
#backpropogation algorithm to feed back the gradient of your new loss function. If you use
#the nn package in torch [torch/nn] then you simply add an nn.LogSigmoid module onto the
#end of the network and change the error criterion to nn.BCECriterion (binary cross
#entropy).


#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters())

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1,
                                                  patience=10, verbose=True)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=300)
#model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=50)



# Save final model
torch.save(model_ft, os.path.join(output_dir,'model_ft.pt'))

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
#weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
weight_softmax = params[-2].data.cpu().numpy()

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (1024, 1024)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    #for idx in class_idx:
        #cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        #cam = cam.reshape(h, w)
        #cam = cam - np.min(cam)
        #cam_img = cam / np.max(cam)
        #cam_img = np.uint8(255 * cam_img)
        #output_cam.append(cv2.resize(cam_img, size_upsample))
    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



data_test_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative_small_set_unbalanced/test'
data_test_outdir = os.path.join(output_dir,'test_out')

## DEBUG
#test_img_1 = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative_small_set_unbalanced/test/0_images_not_infection/00026825_005.jpg'
#test_img_2 = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative_small_set_unbalanced/test/0_images_not_infection/00027048_000.jpg'

#img_file = test_img_2

#img = Image.open(img_file)
#img.save('test_img.jpg')
#img_tensor = preprocess(img)
#img_variable = Variable(img_tensor.unsqueeze(0))
#logit = model_ft(img_variable.cuda(gpu_num))

for dx in class_names:
    if os.path.exists(os.path.join(data_test_outdir,dx)):
        rmtree(os.path.join(data_test_outdir,dx))
    os.makedirs(os.path.join(data_test_outdir,dx))
    for f in os.listdir(os.path.join(data_test_dir,dx)):
        features_blobs = []
        img = Image.open(os.path.join(data_test_dir,dx,f))
        img.save(os.path.join(data_test_outdir,dx,f))
        img_tensor = preprocess(img)
        img_variable = Variable(img_tensor.unsqueeze(0))
        logit = model_ft(img_variable.cuda(gpu_num))

        h_x = F.sigmoid(logit).data.squeeze()
        class_idx = torch.round(h_x).int().cpu().numpy()[0]
        
        CAMs = returnCAM(features_blobs[0], weight_softmax, [class_idx])

        # render the CAM and output
        #print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
        img = cv2.imread(os.path.join(data_test_outdir,dx,f))
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(data_test_outdir,dx,f), result)

#visualize_model(model_ft)

#plt.ioff()
#plt.show()


# PREDICTION
data_infection_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative/test/images_infection/'
data_not_infection_dir = '/home/ubuntu/Desktop/torch-cxr8/images_pneumonia_vs_negative/test/images_not_infection/'
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
    logit = model_ft(img_variable.cuda(gpu_num))

    h_x = F.sigmoid(logit).data.squeeze()
    class_idx = torch.round(h_x).int().cpu().numpy()[0]
    
    pred = class_names[class_idx]
    print('{:.3f} -> {}'.format(h_x.cpu().numpy()[0], pred))
    
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

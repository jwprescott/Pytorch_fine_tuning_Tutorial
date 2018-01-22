# From http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
# Super Resolution model definition in PyTorch

import io
import numpy as np

from torch import nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, models, transforms
import torch.onnx

# Super Resolution model definition in PyTorch
import torch.nn.init as init

from PIL import Image
import cv2

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)


## Create the super-resolution model by using the above model definition.
#torch_model = SuperResolutionNet(upscale_factor=3)

## Load pretrained model weights
#model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'

## Initialize model with the pretrained weights
#map_location = lambda storage, loc: storage
#if torch.cuda.is_available():
    #map_location = None
#torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

## set the train mode to false since we will only run the forward pass.
#torch_model.train(False)

## Input to the model
#batch_size = 1    # just a random number
#x = Variable(torch.randn(batch_size, 1, 224, 224), requires_grad=True)

## Export the model
#torch_out = torch.onnx._export(torch_model,             # model being run
                               #x,                       # model input (or a tuple for multiple inputs)
                               #"super_resolution.onnx", # where to save the model (can be a file or file-like object)
                               #export_params=True)      # store the trained parameter weights inside the model file

# Try densenet model
#torch_model = models.densenet121(pretrained=True)
torch_model = torch.load('/home/prescott/Desktop/output_20180110_162914/model_best_acc_cpu.pth.tar')

torch_model.train(False)

# Input to the model
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#img_pil = Image.open('/home/prescott/Projects/tf-hemorrhage/images_curated/subarachnoid_hemorrhage/34TY8_0_0012.jpg')
img_pil = Image.open('/home/prescott/Desktop/output_20180110_162914/test_out_test/1_images_infection/00023068_049.jpg')

img_tensor = preprocess(img_pil)
x = Variable(img_tensor.unsqueeze(0))
    
#batch_size = 1    # just a random number
#x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)

torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "chexnet.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

import onnx
import onnx_caffe2.backend

# Load the ONNX ModelProto object. model is a standard Python protobuf object
#model = onnx.load("super_resolution.onnx")
model = onnx.load("chexnet.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
prepared_backend = onnx_caffe2.backend.prepare(model)

# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

print("Exported model has been executed on Caffe2 backend, and the result looks good!")

# extract the workspace and the model proto from the internal representation
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net

# Now import the caffe2 mobile exporter
from caffe2.python import core
from caffe2.python.predictor import mobile_exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
c2_net = core.Net(c2_model)
init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_net, c2_model.external_input)

##Another possible way of doing the previous steps without using a workspace (from facebook's AICamera example ipython notebook, https://caffe2.ai/docs/AI-Camera-demo-android.html)
#from onnx_caffe2.backend import Caffe2Backend as c2
#init_net, predict_net = c2.onnx_graph_to_caffe2_net(model.graph, True)

# Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
with open('init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

## Verify it runs with predictor (from facebook's AICamera example)
#with open("squeeze_init_net.pb") as f:
    #init_net = f.read()
#with open("squeeze_predict_net.pb") as f:
    #predict_net = f.read()
#from caffe2.python import workspace
#p = workspace.Predictor(init_net, predict_net)
## The following code should run:
## img = np.random.rand(1, 3, 224, 224).astype(np.float32)
## p.run([img])

    
# Some standard imports
from caffe2.proto import caffe2_pb2
from caffe2.python import net_drawer, net_printer, visualize, workspace, utils

import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform
from IPython import display

### load the cat image and convert it to Ybr format
#img = Image.open("/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/cat_224x224.jpg")
#img_ycbcr = img.convert('YCbCr')
#img_y, img_cb, img_cr = img_ycbcr.split()

# load the image
#img = Image.open('/home/prescott/Projects/tf-hemorrhage/images_curated/subarachnoid_hemorrhage/34TY8_0_0012.jpg')
img = Image.open('/home/prescott/Desktop/output_20180110_162914/test_out_test/1_images_infection/00023068_049.jpg')
img_tensor = preprocess(img)
#img_y = img_tensor.cpu()
img_y = img_tensor.unsqueeze(0).cpu()


# Let's run the mobile nets that we generated above so that caffe2 workspace is properly initialized
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)

# Caffe2 has a nice net_printer to be able to inspect what the net looks like and identify
# what our input and output blob names are.
print(net_printer.to_string(core.Net(predict_net)))

#graph = net_drawer.GetPydotGraph(predict_net, rankdir="LR")
#display.Image(graph.create_png(), width=800)

# Now, let's also pass in the resized cat image for processing by the model.
#workspace.FeedBlob("1", np.array(img_y)[np.newaxis, np.newaxis, :, :].astype(np.float32))
workspace.FeedBlob("1", np.array(img_y).astype(np.float32))

# run the predict_net to get the model output
workspace.RunNetOnce(predict_net)

# Now let's get the model output classifier vector
img_class = workspace.FetchBlob("1277")

# Average pool layer for DenseNet 121
img_avg_pool = workspace.FetchBlob("1272")

# Classifier weight layer for DenseNet121
weights_classifier= workspace.FetchBlob("1274")

# Inner product
#cam = weights_classifier.dot(img_avg_pool.reshape(1024,49))
output_cam = []
cam = cam.reshape(7,7)
cam = cam - np.min(cam)
cam_img = cam / np.max(cam)
cam_img = np.uint8(255 * cam_img)
output_cam.append(cv2.resize(cam_img, (1024,1024)))

# Write result
img_orig = cv2.imread('/home/prescott/Desktop/output_20180110_162914/test_out_test/1_images_infection/00023068_049.jpg')
height, width, _ = img_orig.shape
heatmap = cv2.applyColorMap(cv2.resize(output_cam[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img_orig * 0.5
cv2.imwrite("output.jpg", result)


#img_out_y = Image.fromarray(np.uint8((img_out[0, 0]).clip(0, 255)), mode='L')

## get the output image follow post-processing step from PyTorch implementation
#final_img = Image.merge(
    #"YCbCr", [
        #img_out_y,
        #img_cb.resize(img_out_y.size, Image.BICUBIC),
        #img_cr.resize(img_out_y.size, Image.BICUBIC),
    #]).convert("RGB")

#final_img.save("/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/cat_superres_mobile.jpg")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

def conv_relu_maxp(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]

def conv_bn_relu(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    # flatten 24*32=768 to 120
    self.fc1 = nn.Linear(768, 120)
    self.fc2 = nn.Linear(120, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64,2)
    #self.fc4 = nn.Linear(64,10)
    #defining the 20% dropout
    self.dropout = nn.Dropout(0.2)

  def forward(self,x):
    x = x.view(x.shape[0],-1)
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.dropout(F.relu(self.fc2(x)))
    x = self.dropout(F.relu(self.fc3(x)))
    #x = self.dropout(F.relu(self.fc3(x)))
    #not using dropout on output layer
    x = F.log_softmax(self.fc4(x), dim=1)
    return x


# https://github.com/jeremyfix/deeplearning-lectures/tree/master/LabsSolutions/00-pytorch-FashionMNIST
class ThermalCNN(nn.Module):

    def __init__(self, num_classes):
        super(ThermalCNN, self).__init__()

        # By default, Linear layers and Conv layers use Kaiming He initialization

        self.features = nn.Sequential(
            *conv_relu_maxp(1, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5)
        )
        # You must compute the number of features manualy to instantiate the
        # next FC layer
        # self.num_features = 64*3*3

        # Or you create a dummy tensor for probing the size of the feature maps
        # our thermal data is converted into 24x32 grayscale image
        probe_tensor = torch.zeros((1,1,24,32))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
            *dropout_linear_relu(out_features.shape[0], 128, 0.5),
            *dropout_linear_relu(128, 256, 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  #  OR  x = x.view(-1, self.num_features)
        y = self.classifier(x)
        return y

class ModelCheckpoint:
    def __init__(self, filepath, model):
        self.min_loss = np.Inf
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if loss < self.min_loss:
            print(' Validation loss decreased({:.6f} -->{:.6f}). Saving Model ...'.format(self.min_loss, loss))
            torch.save(self.model.state_dict(), self.filepath)
            self.min_loss = loss

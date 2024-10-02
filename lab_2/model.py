import torch
import torch.nn.functional as F
import torch.nn as nn

class SnoutNet(nn.module):
    def __init__(self):
        super(SnoutNet, self).__init__()

        #input is three channels RBG, output is 64 channels, 3x3 kernel, need a stride of 4 to get 227x227 to 57x57, so 2 stride for convolution and 2 for max pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        #stride of 4 to get from 57x57 to 15x15, but need padding of 1 to make formula work, apply stride of 2 fro max pool after
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        #stride of 4 to get from 15x15 to 4x4, apply max pool stride of 2 after
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        #input size if 4096 ( 4 x 4 x 256 )
        self.fc1 = nn.Linear(4096,1024)

        # same size full connected layer
        self.fc2 = nn.Linear(1024,1024)

        #send to 1x2 output layer
        self.fc3 = nn.Linear(1024,2)

        def forward(self, X):
            X = self.conv1(X)
            X = self.maxpool(X)
            X = F.relu(X)
            X = self.conv2(X)
            X = self.maxpool(X)
            X = F.relu(X)
            X = self.conv3(X)
            X = self.maxpool(X)
            X = F.relu(X)
            X = self.fc1(X)
            X = F.relu(X)
            X = self.fc2(X)
            X = F.relu(X)
            X = self.fc3(X)

            return X
        
        

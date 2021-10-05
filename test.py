from __future__ import print_function, division
import time
from torch.nn.modules.module import T
from tqdm import tqdm
from datetime import datetime
import copy
import os
import argparse
# from termcolor import colored
import numpy as np
import pandas as pd
import cv2
import random
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import *
from plotgraph import plotgraph
from customdataset import CustomDataset, CSVDataset


parser = argparse.ArgumentParser(description="parameters")
parser.add_argument("--model", required=True, help="type in which model file you want to test")
parser.add_argument("--resnet_type", required=True, help="type of resnet")
args = parser.parse_args()

### parameters
model_name = "ResNet"
path = os.path.dirname(os.getcwd()) # "D:/projects"
datapath = "F:/data/test"
modelpath = path + "/" + model_name + "/models/" + args.model
batch_size = 128


### 사용 가능한 gpu 확인 및 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)



### test set for Cifar10
test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.229, 0.224, 0.225)),
                transforms.Resize(224)
])
test_set = datasets.CIFAR10(root=datapath, train=False,
                                       download=True, transform=test_transformer)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)





### model
if args.resnet_type == "resnet18":
    model = ResNet(BasicBlock, [2,2,2,2])

if args.resnet_type == "resnet34":
    model = ResNet(BasicBlock, [3, 4, 6, 3])

if args.resnet_type == "resnet50":
    model = ResNet(BottleNeck, [3,4,6,3])
    
if args.resnet_type == "resnet101":
    model = ResNet(BottleNeck, [3, 4, 23, 3])
     
if args.resnet_type == "resnet152":
    model =  ResNet(BottleNeck, [3, 8, 36, 3])
model.to(device)





## Test Function
def test():
    print("++++++++Testing in Progress+++++++++")
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    test_acc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(test_loader, leave=True):
            x, y = x.to(device), y.to(device)

            # prediction
            prediction = model(x)
            test_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len(test_data)

    print("Acc: [{:.2f}%]".format(
        test_acc
    ))
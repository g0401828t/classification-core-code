from __future__ import print_function, division
import time
from torch.nn.modules.module import T
from tqdm import tqdm
import datetime
import copy
import os
import argparse
# from termcolor import colored
import numpy as np
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
import plotgraph

np.random.seed(0)

parser = argparse.ArgumentParser(description="resnet_teacher")
parser.add_argument("--model_name", default="vgg16", type=str, help='setting model')
parser.add_argument("--mode", default="train", type=str, help="setting mode")
# parser.add_argument("--data_name", default="car", type=str)
parser.add_argument("--hp_lr", type=float, default=1e-2, help="setting learning rate")
parser.add_argument("--hp_wd", type=float, default=5e-4, help="setting weight decay")
parser.add_argument("--hp_bs", type=int, default=128, help="setting batch size")
parser.add_argument("--hp_ep", type=int, default=200, help="setting epochs")
parser.add_argument("--hp_opt", type=str, default="sgd", help="setting optimizer")
parser.add_argument("--hp_sch", type=str, default="cos", help="setting scheduler")
parser.add_argument("--num_worker", type=int, default=0)
parser.add_argument("--earlystop", type=int, default=7, help="how many iterations for early stop")

parser.add_argument("--description", type=str, default="", help="for saving loss, acc graph, and description")

args = parser.parse_args()

path = os.getcwd() # os.getcwd = "D:/projects/aifarm", os.path.dirname = "D:/projects" directory name of GoogLeNet
result_path = path + "/results"
modelpath = path + "/models"
if not os.path.exists(result_path):  # make path to save results (loss graph, acc graph)
      os.mkdir(result_path)
if not os.path.exists(modelpath):  # make path to save model.h (best model during training)
      os.mkdir(modelpath)
modelpath = modelpath + "/" + args.model_name + args.description + ".h"  # [modelpath].h <= model name to save. (ex. vggnet16_augmented.h)

"""
# read private_arguments
f = open("../p_command_multi/private_arguments.txt", "r")
lines = f.readlines()
for line in lines:
    line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
    line_split = line.split(" ")
    locals()[line_split[0][2:]] = int(line_split[1])
f.close()
"""


## dataset & dataloader

transform_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform = transforms.Compose(
    [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset1 = ImageFolder("../dataset/tomato/Tomato_all")
dataset = ImageFolder("../dataset/tomato/Tomato_all", transform=transform)
pdb.set_trace()

# Shuffle the indices
len_dataset = len(dataset)
len_train = int(len_dataset * 0.7)
len_val = int(len_dataset * 0.15)
len_test = int(len_dataset * 0.15)

print(len_dataset)

indices = np.arange(0, len_dataset)
np.random.shuffle(indices)  # shuffle the indicies

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=args.num_worker,
    sampler=torch.utils.data.SubsetRandomSampler(indices[:len_train]),
)
val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=args.num_worker,
    sampler=torch.utils.data.SubsetRandomSampler(indices[len_train:len_train+len_val]),
)
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=args.num_worker,
    sampler=torch.utils.data.SubsetRandomSampler(indices[len_train+len_val:len_train+len_val+len_train]),
)

## show dataset
"""
import matplotlib.pyplot as plt
def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
dataiter = iter(train_loader)
images, labels = dataiter.next()

classes = ('Tomato_D01', 'Tomato_D02', 'Tomato_D03', 'Tomato_D04', 'Tomato_D05', 'Tomato_D06', 'Tomato_D07', 'Tomato_D08', 'Tomato_D09', 'Tomato_H', 'Tomato_P01', 'Tomato_P02', 'Tomato_P03', 'Tomato_P04', 'Tomato_P05', 'Tomato_R01')
imshow(torchvision.utils.make_grid(images))
print("".join("%Ss" % classes[labels[j]] for j in range(4)))

"""



## Training
def train(model, criterion, optimizer, scheduler, num_epochs):
    print("++++++++Training in Progress+++++++++")
    model.train()
    loss_list, valloss_list, valacc_list = [], [], []
    best_acc = 0
    best_loss = float("inf")
    for epoch in range(num_epochs):
        avg_loss, val_loss, val_acc = 0, 0, 0

        for param_group in optimizer.param_groups:  # to see the learning rate per epoch
            current_lr =  param_group['lr']

        for x, y in tqdm(train_loader, leave=True):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()

            # forward propagation
            hypothesis = model(x)
            loss = criterion(hypothesis, y)

            # back propagation
            loss.backward()
            optimizer.step()

            avg_loss += loss/len_train  # calculate average loss per epoch
        
        ### validation
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=True):
                x, y = x.to(device), y.to(device)

                prediction = model(x)

                # calculate validation Loss
                val_loss += criterion(prediction, y) / len_val

                # calculate validation Accuracy
                val_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len_val

        print(datetime.now().time().replace(microsecond=0), "EPOCHS: [{}], current_lr: [{}], avg_loss: [{:.4f}], val_loss: [{:.4f}], val_acc: [{:.2f}%]".format(
                epoch+1, current_lr, avg_loss.item(), val_loss.item(), val_acc))

        # append list and plot graph
        loss_list.append(avg_loss.item())
        valloss_list.append(val_loss.item())
        valacc_list.append(val_acc)
        plotgraph(loss_list=loss_list, valloss_list=valloss_list, valacc_list=valacc_list, path = result_path, description=args.model_name + args.description)

        # # Early Stop based on val_acc
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     es = earlystop
        #     torch.save(model.state_dict(), modelpath)
        # else: 
        #     es -= 1
        # if es == 0: 
        #     # model.load_state_dict(torch.load(modelpath))
        #     print("Early Stopped and saved model")
        #     break

        # Early Stop based on val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            es = args.earlystop
            print("Best model saved")
            torch.save(model.state_dict(), modelpath)
        else: 
            print("Not Best, earlystopping in:", es, "step(s)")
            es -= 1
        if es == 0: 
            # model.load_state_dict(torch.load(model_save_name))
            print("Early Stopped and saved model")
            break

        # learning rate scheduler per epoch
        scheduler.step(val_loss)

    print("finished training")

    
### Test Function
def test():
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
            test_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len_test

    print("Acc: [{:.2f}%]".format(
        test_acc
    ))


if __name__ == "__main__":
    

    ## Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_my_model(model_name=args.model_name, num_classes = len(dataset.classes))
    model.to(device)

    ## Loss function
    criterion = nn.CrossEntropyLoss()

    ## Optimizer
    if args.hp_opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.hp_lr, momentum=0.9, weight_decay=args.hp_wd)
    if args.hp_opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.hp_lr, weight_decay=args.hp_wd)

    ## Scheduler
    if args.hp_sch == "msl":
        hp_lr_decay_ratio = 0.2

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                args.hp_ep * 0.3,
                args.hp_ep * 0.6,
                args.hp_ep * 0.8,
            ],
            gamma=hp_lr_decay_ratio,
        )
    if args.hp_sch == "cos":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.hp_ep)
    if args.hp_sch == "plateu":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    
    
    
    if args.mode == "train":
        ### train
        train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=args.hp_ep)
        ### test
        test()
    elif args.mode =="test":
        test()
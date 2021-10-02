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

## dataset & dataloader
class MyLazyDataset():
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):

        if self.transform:
            self.x = self.transform(self.dataset[index][0])
        else:
            self.x = self.dataset[index][0]
        self.y = self.dataset[index][1]
        return self.x, self.y
    
    def __len__(self):
        return len(self.dataset)

## Training
def train(model, criterion, optimizer, scheduler, num_epochs):
    print("++++++++Training in Progress+++++++++")
    model.train()
    loss_list, valloss_list, valacc_list = [], [], []
    best_acc = 0
    best_loss = float("inf")
    for epoch in range(num_epochs):
        print("EPOCH: [", epoch, "/", num_epochs, "]")
        avg_loss, val_loss, val_acc = 0, 0, 0

        for param_group in optimizer.param_groups:  # to see the learning rate per epoch
            current_lr =  param_group['lr']

        for x, y in tqdm(train_loader, leave=True):
            iter_time = time.time()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()

            # forward propagation
            hypothesis = model(x)
            loss = criterion(hypothesis, y)

            # back propagation
            loss.backward()
            optimizer.step()

            avg_loss += loss/len(train_data)  # calculate average loss per epoch
            print("Ieration Time:", time.time() - iter_time)
            break
        break
        
        ### validation
        model.eval()
        print("++++++++Validation in Progress+++++++++")
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=True):
                x, y = x.to(device), y.to(device)

                prediction = model(x)

                # calculate validation Loss
                val_loss += criterion(prediction, y) / len(val_data)

                # calculate validation Accuracy
                val_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len(val_data)

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


if __name__ == "__main__":
    
    np.random.seed(0)

    parser = argparse.ArgumentParser(description="resnet_teacher")
    parser.add_argument("--model_name", default="vgg16", type=str, help='setting model')
    parser.add_argument("--num_classes", default=16, help="num of classes for classification")
    parser.add_argument("--mode", default="train", type=str, help="setting mode")
    # parser.add_argument("--data_name", default="car", type=str)
    parser.add_argument("--hp_lr", type=float, default=1e-2, help="setting learning rate")
    parser.add_argument("--hp_wd", type=float, default=5e-4, help="setting weight decay")
    parser.add_argument("--hp_bs", type=int, default=128, help="setting batch size")
    parser.add_argument("--hp_ep", type=int, default=200, help="setting epochs")

    parser.add_argument("--hp_loss", type=str, default="ce", help="setting scheduler")
    parser.add_argument("--hp_opt", type=str, default="sgd", help="setting optimizer")
    parser.add_argument("--hp_sch", type=str, default="cos", help="setting scheduler")

    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--earlystop", type=int, default=7, help="how many iterations for early stop")

    parser.add_argument("--description", type=str, default="", help="for saving loss, acc graph, and description")

    args = parser.parse_args()

    # Inintialize Directory and paths
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
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transform = {
        'train': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomCrop((448,448)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]),
        
        'val': transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]),
    }

    data_dir = "../dataset/tomato/Tomato_all"
    dataset = ImageFolder(data_dir)
    ratio = 0.8
    lengths = [int(len(dataset)*ratio), int(len(dataset)*(1-ratio)/2), len(dataset)-int(len(dataset)*(ratio))-int(len(dataset)*(1-ratio)/2)]
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths)
    train_data = MyLazyDataset(train_set, transform['train'])
    val_data = MyLazyDataset(val_set, transform['val'])
    test_data = MyLazyDataset(test_set, transform['val'])
    
    ## for Custom Dataset => same speed as ImageFolder
    # datapath = "../dataset/tomato/Tomato_all"
    # train_data = CustomDataset(datapath,transform["train"])
    # val_data = CustomDataset(datapath,transform["val"]) #test transforms are applied
    # test_data = CustomDataset(datapath,transform["val"])

    ## for CSVDataset => No improvements..
    # df = pd.read_csv('../dataset/tomato/tomato_all.csv')
    # X = df.image_path.values
    # y = df.target.values
    # from sklearn.model_selection import train_test_split
    # (xtrain, xtest, ytrain, ytest) = (train_test_split(X, y, test_size=0.25, random_state=42))
    # train_data = CSVDataset(xtrain, ytrain, tfms=1)
    # test_data = CSVDataset(xtest, ytest, tfms=0)
    # val_data = test_data



    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.hp_bs,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.hp_bs,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.hp_bs,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
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
    

    ## Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_my_model(model_name=args.model_name, num_classes = args.num_classes)
    model.to(device)

    ## Loss function
    if args.hp_loss == "ce":
        criterion = nn.CrossEntropyLoss()
    if args.hp_loss == "my_loss":
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
    
    

    ## MainProcess (training & testing)
    if args.mode == "train":
        ### train
        train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=args.hp_ep)
        ### test
        # test()
    elif args.mode =="test":
        test()
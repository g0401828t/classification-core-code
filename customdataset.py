import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
from PIL import Image
import glob
import numpy
import random
import matplotlib.pyplot as plt


# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, datapath, transform=False):
        
        self.data_path = datapath
        self.transform = transform

        # save classes and image paths
        self.image_paths = []
        self.classes = []
        for c_path in glob.glob(self.data_path + "/*"):
            c_path = c_path.replace("\\", "/")  # for windows
            self.classes.append(c_path.split("/")[-1])
            for d_path in glob.glob(c_path + "/*"):
                d_path = d_path.replace("\\","/")
                self.image_paths.append(d_path)
        # print("len of imagese", len(self.image_paths))
        # print("image_path examples: \n", self.image_paths[0])
        # print("class example:", self.classes[0])

        # split paths to train:val:test = 8:1:1
        ratio = 0.8
        num_train = int(ratio*len(self.image_paths))
        num_val = int((1-ratio)/2*len(self.image_paths))
        train_image_paths, valid_image_paths, test_image_paths \
            = self.image_paths[:num_train], self.image_paths[num_train:num_train+num_val], self.image_paths[num_train+num_val:] 
        # print("train:val:test", len(train_image_paths), len(valid_image_paths), len(test_image_paths))

        # create index to classes
        idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in idx_to_class.items()}

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        # np_image = cv2.imread(image_filepath)
        # np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        # print(np_image[idx])
        # pil_image = Image.fromarray(np_image.astype("uint8"), "RGB")
        # print(pil_image[idx])

        pil_image = Image.open(image_filepath)
        pil_image = pil_image.convert("RGB")  # 3 channels
        # plt.figure(figsize=(5, 20))
        # plt.axis("off")
        # plt.imshow(pil_image, cmap=plt.cm.gray)
        # plt.show()

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(pil_image)
        
        return image, label
        
    def classes(self):
        return len(self.classes)






import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations
# image dataset module
class CSVDataset(Dataset):
    def __init__(self, path, labels, tfms=None):
        self.X = path
        self.y = labels
        # apply augmentations
        if tfms == 0: # if validating
            self.aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225], always_apply=True)
            ])
        else: # if training
            self.aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.3,
                    scale_limit=0.3,
                    rotate_limit=30,
                    p=1.0
                ),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225], always_apply=True)
            ])
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = image.convert("RGB")  # 3 channels
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
import os
import pandas as pd
import numpy as np
import joblib
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm


# get all the image paths
image_paths = list(paths.list_images("../dataset/tomato/Tomato_all"))
# create an empty DataFrame
data = pd.DataFrame()
labels = []
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    label = image_path.split(os.path.sep)[-2]
    data.loc[i, 'image_path'] = image_path
    labels.append(label)

labels = np.array(labels)
print(labels)
# one hot encode
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping an one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(labels)}")
for i in range(len(labels)):
    index = np.argmax(labels[i])
    data.loc[i, 'target'] = int(index)
# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)
# save as csv file
data.to_csv('../dataset/tomato/tomato_all.csv', index=False)
# # pickle the label binarizer
# joblib.dump(lb, '../outputs/lb.pkl')
# print('Save the one-hot encoded binarized labels as a pickled file.')
print(data.head())







import torch
from torch.utils.data import Dataset
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
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)



df = pd.read_csv('../dataset/tomato/tomato_all.csv')
X = df.image_path.values
y = df.target.values

from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = (train_test_split(X, y, test_size=0.25, random_state=42))

train_data = CSVDataset(xtrain, ytrain, tfms=1)
test_data = CSVDataset(xtest, ytest, tfms=0)
 
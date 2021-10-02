from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
import h5py

import os
import glob
import cv2
from tqdm import tqdm

IMG_WIDTH = 224
IMG_HEIGHT = 224



data_dir = "../dataset/tomato/Tomato_all"
nfiles = len(glob.glob(data_dir + './*'))
print(f'count of image files nfiles={nfiles}')

h5file = "../dataset/tomato/tomato_all1.h5"


# if not os.path.exists(h5file):  # make path to save results (loss graph, acc graph)
#     with h5py.File(h5file,'w') as  h5f:
#         for idx, class_path in enumerate(glob.iglob(data_dir + './*')):
#             class_path = class_path.replace("\\", "/")  # for windows
#             classes = class_path.split("/")[-1]
#             class_group = h5f.create_group(classes)

#             nfiles = len(glob.glob(class_path + './*'))
#             img_ds = class_group.create_dataset('images',shape=(nfiles, IMG_WIDTH, IMG_HEIGHT,3), dtype=int)
#             for cnt, data_path in tqdm(enumerate(glob.glob(class_path + "/*"))):
#                 data_path = data_path.replace("\\","/")
#                 img = cv2.imread(data_path, cv2.IMREAD_COLOR)
#                 # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
#                 img_resize = cv2.resize( img, (IMG_WIDTH, IMG_HEIGHT) )
#                 img_ds[cnt:cnt+1:,:,:] = img_resize
            
            

if not os.path.exists(h5file):  # make path to save results (loss graph, acc graph)
    with h5py.File(h5file,'w') as  h5f:
        nfiles = 0


        image_group = []
        classes_group = []
        for idx, class_path in enumerate(glob.iglob(data_dir + './*')):
            class_path = class_path.replace("\\", "/")  # for windows
            classes = class_path.split("/")[-1]
            # class_group = h5f.create_group(classes)

            # nfiles = len(glob.glob(class_path + './*'))
            for cnt, data_path in tqdm(enumerate(glob.glob(class_path + "/*"))):
                if cnt == 2:
                    break
                nfiles += 1
                data_path = data_path.replace("\\","/")
                img = cv2.imread(data_path, cv2.IMREAD_COLOR)
                # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
                # img_resize = cv2.resize( img, (IMG_WIDTH, IMG_HEIGHT) )
                # image_group[cnt:cnt+1:,:,:] = img
                # class_group[cnt:cnt+1:,:,:] = classes
                image_group.append(img)
                classes_group.append(classes)
        
        
    
        print("n_files", nfiles)
        img_dt = h5f.create_dataset('images', shape=[nfiles,  IMG_WIDTH, IMG_HEIGHT,3], dtype=int)
        class_dt = h5f.create_dataset('classes', shape=[nfiles,1], dtype=int)
        img_dt = image_group
        class_dt = classes_group



file = h5py.File(h5file, 'r')

print(file.keys())
import pdb
pdb.set_trace()

class DeephomographyDataset(Dataset):
    def __init__(self,hdf5file,imgs_key='images',labels_key='labels',
            transform=None):

        self.hdf5file=hdf5file


        self.imgs_key=imgs_key
        self.labels_key=labels_key
        self.transform=transform
    def __len__(self):

    # return len(self.db[self.labels_key])
        with h5py.File(self.hdf5file, 'r') as db:
            lens=len(db[self.labels_key])
        return lens
    def __getitem__(self, idx):
        with h5py.File(self.hdf5file,'r') as db:
            image=db[self.imgs_key][idx]
            label=db[self.labels_key][idx]
        sample={'images':image,'labels':label}
        if self.transform:
            sample=self.transform(sample)   
        return sample

# dataset = DeephomographyDataset(h5file)
# print(dataset[0])
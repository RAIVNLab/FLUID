import os.path as osp
import os
from PIL import Image
import numpy as np
import pandas as pd

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './Imagenet/'
        

class ContinuousDataset(Dataset):

    def __init__(self, root, transform, distribution = None):
        #Storing data and metadata
        self.dataset = []
        self.labels = []
        self.seen_classes = set()
        self.label_map = dict()
        self.imgs_per_class = []
        self.data_order = [] 
        class_folders = os.listdir(root)
        self.transform = transform
        if distribution is None:
            distribution = np.ones(len(class_folders))
        current_label = 0
        for class_folder in class_folders:
            folder_path = osp.join(root, class_folder)
            img_files = os.listdir(folder_path)
            num_imgs = int(np.floor(len(img_files)*distribution[current_label]))
            np.random.shuffle(img_files)
            for i in range(num_imgs):
                self.dataset.append(osp.join(folder_path, img_files[i]))
                self.labels.append(current_label)
            self.label_map[current_label] = class_folder
            self.imgs_per_class.append(num_imgs)
            current_label += 1
        #Initializing more metadata
        self.observed_samples = np.zeros(len(self), dtype = bool)
        self.dataset = np.array(self.dataset)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        path, label = self.dataset[i], self.labels[i]
        seen = label in self.seen_classes
        self.data_order.append(i)
        if not seen:
            self.seen_classes.add(label)
        self.observed_samples[i] = True
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, seen

    
    def get_observed_samples(self):
        return self.dataset[self.observed_samples], self.labels[self.observed_samples]
        

class OfflineDataset(Dataset):

    def __init__(self, online_dataset, transform):
        self.dataset, self.labels = online_dataset.get_observed_samples()
        self.reference_dataset = online_dataset
        self.transform = transform
        
    def __len__(self):
        length = len(self.dataset)
        if length == 0:
            return 1
        else: 
            return length

    def update(self):
        self.dataset, self.labels = self.reference_dataset.get_observed_samples()
    
    def __getitem__(self, i):
        path, label = self.dataset[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
        

        

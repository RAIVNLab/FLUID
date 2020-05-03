import os.path as osp
import os
from PIL import Image
import numpy as np
import pandas as pd

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import file_to_class, create_imagenet_map, create_novel_class_map
        

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
        

        
class ContinuousDatasetRF(Dataset):

    def __init__(self, root, transform, sequence_num):
        #Storing data and metadata
        self.seen_classes = set()
        self.transform = transform
        tmp_path = 'S' + str(sequence_num) + '/sequence' + str(sequence_num) + '.npy'
        self.sequence = np.load(os.path.join(root, tmp_path))

        tmp_path = 'S' + str(sequence_num) + '/class_map' + str(sequence_num) + '.npy'
        novel_classes = create_novel_class_map(root, sequence_num)
        self.seen_classes.update(range(1000))
        self.seen_classes.remove(novel_classes)
        class_map_base = novel_classes
        imagenet_classes = create_imagenet_map(root)
        self.class_map = imagenet_classes
        self.class_map.update(class_map_base)
        tmp_path = 'S' + str(sequence_num) + '/class_count' + str(sequence_num) + '.npy'
        self.imgs_per_class = np.load(os.path.join(root, tmp_path))
        self.counter = -1
        self.root = root

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self,i):
        path = self.sequence[self.counter]
        img_path = os.path.join(self.root, path)
        label = file_to_class(img_path, self.class_map)
        seen = label in self.seen_classes
        if not seen:
            self.seen_classes.add(label)
        image = self.transform(Image.open(img_path).convert('RGB'))
        self.counter += 1
        return image, label, seen

    #def imgs_per_class 
    def get_samples_seen(self):
        return self.counter

    def set_counter(self, counter):
        self.counter = counter

class OfflineDatasetRF(Dataset):

    def __init__(self, root, transform, sequence_num):
        self.transform = transform
        path = 'S' + str(sequence_num) + '/sequence' + str(sequence_num) + '.npy'
        self.sequence = np.load(os.path.join(root, path))
        path = 'S' + str(sequence_num) + '/class_map' + str(sequence_num) + '.npy'
        class_map_base = np.load(os.path.join(root, path), allow_pickle = True).item()
        self.class_map = create_imagenet_map(root)
        self.class_map.update(class_map_base)
        self.counter = 0
        self.root = root

    def __len__(self):
        #Trick pytorch to initialize empty dataset
        if self.counter == 0:
            return 1
        else: 
            return self.counter

    def update(self, count):
        self.counter = count
    
    def __getitem__(self, i):
        path = self.sequence[i]
        img_path = os.path.join(self.root, path)
        label = file_to_class(img_path, self.class_map)
        image = self.transform(Image.open(img_path).convert('RGB'))
        return image, label


    
class CategoriesSampler():

    def __init__(self, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

    def __len__(self):
        return self.n_batch
    
    def update(self, new_label):
        self.m_ind = []
        for i in range(max(new_label) + 1):
            ind = np.argwhere(new_label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            lens = np.array([len(x) for x in self.m_ind])
            classes = np.array([x[0] for x in np.argwhere(lens > self.n_per)])
            classes = classes[np.random.randint(len(classes), size = self.n_cls)]
            #classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append (l[pos])


            batch = torch.stack(batch).t().reshape(-1)
            yield batch
    
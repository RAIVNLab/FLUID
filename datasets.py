import os.path as osp
import os
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import file_to_class, create_imagenet_map, create_novel_class_map, create_train_transform
        
class ContinuousDatasetRF(Dataset):

    def __init__(self, root, transform, sequence_num):
        #Storing data and metadata
        self.seen_classes = set()
        self.transform = transform
        tmp_path = 'S' + str(sequence_num) + '/sequence' + str(sequence_num) + '.npy'
        self.sequence = np.load(os.path.join(root, tmp_path))

        novel_classes_map = create_novel_class_map(root, sequence_num)
        self.seen_classes.update(range(1000))
        self.seen_classes -= set(novel_classes_map.values())
        class_map_base = novel_classes_map

        imagenet_class_map = create_imagenet_map(root)
        self.class_map = imagenet_class_map
        self.class_map.update(class_map_base)
        tmp_path = 'S' + str(sequence_num) + '/imgs_per_class' + str(sequence_num) + '.npy'
        self.imgs_per_class = np.load(os.path.join(root, tmp_path))
        self.counter = -1
        self.root = root

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self,i):
        path = 'data/' + self.sequence[self.counter]
        img_path = os.path.join(self.root, path)
        label = file_to_class(img_path, self.class_map)
        seen = label in self.seen_classes
        if not seen:
            self.seen_classes.add(label)
        image = self.transform(Image.open(img_path).convert('RGB'))
        self.counter += 1
        return image, label, seen

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
        #Trick for pytorch to initialize empty dataset
        if self.counter == 0:
            return 1
        else: 
            return self.counter

    def update(self, count):
        self.counter = count
    
    def __getitem__(self, i):
        path = 'data/' + self.sequence[i]
        img_path = os.path.join(self.root, path)
        label = file_to_class(img_path, self.class_map)
        image = self.transform(Image.open(img_path).convert('RGB'))
        return image, label


    
class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            lens = np.array([len(x) for x in self.m_ind])
            classes = np.array([x[0] for x in np.argwhere(lens > self.n_per)])
            classes = classes[np.random.randint(len(classes), size = self.n_cls)]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append (l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class MetaImageNet(Dataset):

    def __init__(self, root):
        data = []
        label = []
        lb = 0

        self.wnids = []
        folders = os.listdir(root)
        for folder in folders:
            folder_path = os.path.join(root, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                data.append(file_path)
                label.append(lb)
            lb += 1


        self.data = data
        self.label = label

        self.tf = create_train_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.tf(Image.open(path).convert('RGB'))
        return image, label
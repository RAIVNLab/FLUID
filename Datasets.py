import os.path as osp
from PIL import Image
import numpy as np
import pandas as pd

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './Imagenet/'


class ContinuousDataset(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if label not in
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def known_classes(self):
        "Return all classes seen so far"
        return self.known_classes

    def num_seen_classes(self):
        self.num_seen_classes

    def observed_samples(self):
        "Return all seen samples so far"
        return self.observed_samples
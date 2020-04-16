import os.path as osp
import datasets
import numpy as np
import os
import torch
from datasets import ContinuousDataset, OfflineDataset
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def __init__(self, args, model, device):
        pass
    @abstractmethod
    def update(self, dataset, loader):
        pass

class BatchTrainer(Trainer):
    pass

class FineTuner(Trainer):
    pass

class InstanceInitialization(Trainer):
    pass

class NearestNeighbor(Trainer):
    def __init__(self, model, device,  args, loader):
        self.model = model
        self.device = device
        self.args = args
        self.loader = loader
    def update(self):
        eps = 1e-8
        proto = torch.zeros([self.args.num_classes, 512]).to(self.device)
        labels = torch.zeros(self.args.num_classes).to(self.device)
        for j, (data, label) in enumerate(self.loader):
            data = data.to(self.device)
            onehot = torch.zeros(label.size(0), self.args.num_classes)
            filled_onehot = onehot.scatter_(1, label.unsqueeze(dim=1), 1).to(self.device).detach()
            embeddings = self.model.features(data).detach()
            print(filled_onehot.permute((1, 0)).shape, embeddings.shape)
            new_prototypes = torch.mm(filled_onehot.permute((1, 0)), embeddings)
            proto += new_prototypes
            labels += filled_onehot.sum(dim = 0)
            del new_prototypes
            del filled_onehot
        final_proto = proto/(labels.unsqueeze(1)+eps)
        self.model.centroids = final_proto


def train(model, training_dataset, training_loader, optimizer, device, args):
    model = model.train()
    for i in range(args.epochs):
        for j, (data, label) in enumerate(training_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = F.cross_entropy(pred, label)/args.batch_factor
            loss.backward()
            if (j+1) % args.batch_factor == 0:
                optimizer.step()
                model.zero_grad()
    model = model.eval()
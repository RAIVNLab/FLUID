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
from utils import extract_layers
from models import extract_backbone
import sys

class Trainer(ABC):
    @abstractmethod
    def __init__(self, model, device, update_opts, offline_dataset):
        self.model = model.eval()
        self.device = device
        self.update_opts = update_opts
        self.offline_dataset = offline_dataset
        self.offline_loader = torch.utils.data.DataLoader(offline_dataset, batch_size=int(update_opts.offline_batch_size/update_opts.batch_factor),
                                                    shuffle=True, num_workers=8, pin_memory=True)

    @abstractmethod
    def update_model(self):
        pass
    def update_dataset(self, counter):
        self.offline_dataset.update(counter)

class InstanceInitialization(Trainer):
    pass

class CentroidTrainer(Trainer):
    def __init__(self, model, device,  update_opts, offline_dataset):
        super().__init__(model, device, update_opts, offline_dataset)
        x, _ = next(iter(self.offline_loader))
        x = x.to(device)
        self.feature_dim = model.features(x).shape[-1]
        model.backbone = model.backbone.eval()
        self.sample_counter = 0
        self.running_labels = torch.zeros(1000).to(self.device)
        self.running_proto = torch.zeros([1000, self.feature_dim]).to(self.device)

    def update_model(self):
        total_samples = self.offline_dataset.counter
        num_samples = total_samples - self.sample_counter
        eps = 1e-8
        for i in range(self.sample_counter, total_samples):
            data, label = self.offline_dataset.__getitem__(i)
            data = data.to(self.device).unsqueeze(0)
            label = torch.tensor([label])
            onehot = torch.zeros(label.size(0), 1000)
            filled_onehot = onehot.scatter_(1, label.unsqueeze(dim=1), 1).to(self.device).detach()
            embeddings = self.model.features(data).detach().unsqueeze(0)
            new_prototypes = torch.mm(filled_onehot.permute((1, 0)), embeddings) 
            self.running_proto += new_prototypes
            self.running_labels += filled_onehot.sum(dim = 0)
            del new_prototypes
            del filled_onehot
        proto = self.running_proto/(self.running_labels.unsqueeze(1)+eps)
        self.model.centroids = proto
        self.sample_counter = total_samples

class HybridTrainer(Trainer):
    def __init__(self, model, device,  update_opts, offline_dataset, class_map):
        super().__init__(model, device, update_opts, offline_dataset)
        x, _ = next(iter(self.offline_loader))
        x = x.to(device)
        self.feature_dim = model.features(x).shape[-1]
        model.backbone = model.backbone.eval()
        self.sample_counter = 0
        self.idcs = [x for x in np.arange(0,1000) if x not in class_map.values()]
        self.initialized_classes = set(self.idcs)
        #self.initialized_classes = set()
        self.params = []
        #centroids = torch.zeros([1000, self.feature_dim])
        #centroids[self.idcs] = self.model.classifier.weight[self.idcs]
        #self.model.base = torch.nn.Parameter(centroids.to(device))

        self.num_layers = update_opts.num_layers
        extract_layers(self.model, self.num_layers, self.params)
        # self.optimizer = torch.optim.SGD([self.model.centroids]+self.params, self.update_opts.lr,
        #                             momentum=self.update_opts.m,
        #                             weight_decay=1e-4)
        self.running_labels = torch.zeros(1000).to(self.device)
        self.running_proto = torch.zeros([1000, self.feature_dim]).to(self.device)
        self.counter = 0

    def update_model(self):
        if self.offline_dataset.counter+1 <= self.update_opts.transition_num:
            self.initialize_centroids()
        if self.offline_dataset.counter+1 == self.update_opts.transition_num:
            print('reinitializing')
            del self.running_labels
            del self.running_proto
            torch.cuda.empty_cache()
            self.optimizer = torch.optim.SGD([self.model.centroids]+self.params, self.update_opts.lr,
                                    momentum=self.update_opts.m,
                                    weight_decay=1e-4)  
            #self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #print(self.offline_dataset.counter > (n+z) and (self.offline_dataset.counter+1) % 5000 == 0)
        if self.offline_dataset.counter >= self.update_opts.transition_num and (self.offline_dataset.counter+1) % self.update_opts.ft_interval == 0:
            print('training')
            self.train()

    def initialize_centroids(self):
        self.model.eval()
        total_samples = self.offline_dataset.counter
        eps = 1e-8
        # self.running_labels = torch.zeros(1000).to(self.device)
        # self.running_proto = torch.zeros([1000, self.feature_dim]).to(self.device)
        for i in range(self.sample_counter, total_samples):
            data, label = self.offline_dataset.__getitem__(i)
            #if label not in self.initialized_classes:
            #self.initialized_classes.add(label)
            data = data.to(self.device).unsqueeze(0)
            label = torch.tensor([label])
            onehot = torch.zeros(label.size(0), 1000)
            filled_onehot = onehot.scatter_(1, label.unsqueeze(dim=1), 1).to(self.device).detach()
            embeddings = self.model.features(data).detach().squeeze().unsqueeze(0)
            #embeddings = (embeddings/embeddings.norm()).unsqueeze(0)
            new_prototypes = torch.mm(filled_onehot.permute((1, 0)), embeddings)
            # if label == 0:
            #     print(embeddings)
            self.running_proto += new_prototypes
            self.running_labels += filled_onehot.sum(dim = 0)
            del new_prototypes
            del filled_onehot
        proto = self.running_proto/(self.running_labels.unsqueeze(1)+eps)
        #proto = proto/20
        proto = self.running_proto/(self.running_proto.norm(dim=1).unsqueeze(1) + eps)
        #proto = torch.zeros([1000, self.feature_dim]).to(self.device)
        #print(proto[torch.tensor(self.idcs)].sum())
        #self.model.centroids = torch.nn.Parameter(self.model.centroids.data + proto)
        #self.model.centroids = torch.nn.Parameter(self.model.base + proto)
        self.model.centroids = torch.nn.Parameter(proto)
        self.sample_counter = total_samples

    def train(self):
        self.model.train()
        for i in range(self.update_opts.epochs):
            for j, (data, label) in enumerate(self.offline_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                pred = self.model(data)
                loss = F.cross_entropy(pred, label)/self.update_opts.batch_factor
                loss.backward()
                if (j+1) % self.update_opts.batch_factor == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
        self.model = self.model.eval()

class BatchTrainer(Trainer):
    def __init__(self, model, device, update_opts, offline_dataset):
        super().__init__(model, device, update_opts, offline_dataset)
        self.optimizer = torch.optim.SGD(model.parameters(), update_opts.lr,
                                    momentum=update_opts.m,
                                    weight_decay=1e-4)
    def update_model(self):
        self.model.train()
        for i in range(self.update_opts.epochs):
            for j, (data, label) in enumerate(self.offline_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                pred = self.model(data)
                loss = F.cross_entropy(pred, label)/self.update_opts.batch_factor
                loss.backward()
                if (j+1) % self.update_opts.batch_factor == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
        self.model = self.model.eval()

class FineTune(Trainer):
    def __init__(self, model, device, update_opts, offline_dataset):
        super().__init__(model, device, update_opts, offline_dataset)
        self.params = []
        extract_layers(model, update_opts.num_layers, self.params)
        self.optimizer = torch.optim.SGD(self.params, self.update_opts.lr,
                                    momentum=self.update_opts.m,
                                    weight_decay=1e-4)

    def update_model(self):
        self.model.train()
        for i in range(self.update_opts.epochs):
            for j, (data, label) in enumerate(self.offline_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                pred = self.model(data)
                loss = F.cross_entropy(pred, label)/self.update_opts.batch_factor
                loss.backward()
                if (j+1) % self.update_opts.batch_factor == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
        torch.save(self.params, 'weight_examine')
        self.model = self.model.eval()

class SplitTrainer(Trainer):
    def __init__(self, model, device, update_opts, offline_dataset):
        super().__init__(model, device, update_opts, offline_dataset)
        self.params = model.novel_classifier.parameters()
        self.optimizer = torch.optim.SGD(self.params, self.update_opts.lr,
                                    momentum=self.update_opts.m,
                                    weight_decay=1e-4)

    def update_model(self):
        self.model.train()
        for i in range(self.update_opts.epochs):
            for j, (data, label) in enumerate(self.offline_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                pred = self.model(data)
                loss = F.cross_entropy(pred, label)/self.update_opts.batch_factor
                loss.backward()
                if (j+1) % self.update_opts.batch_factor == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
        self.model = self.model.eval()

class NoTrain(Trainer):
    def __init__(self, model, device, update_opts, offline_dataset):
        super().__init__(model, device, update_opts, offline_dataset)

    def update_model(self):
        pass


class OLTRTrainer(Trainer):
    def __init__(self, model, device, update_opts, offline_dataset):
        super().__init__(model, device, update_opts, offline_dataset)
        self.optimizer = torch.optim.SGD(model.parameters(), update_opts.lr,
                                    momentum=update_opts.m,
                                    weight_decay=1e-4)

    def update_model(self):
        for i in range(self.update_opts.epochs):
            for model in self.model.networks.values():
                model.train()

            torch.cuda.empty_cache()
            for j, (data, label) in enumerate(self.offline_loader):
                data = data.to(self.device)
                labels = label.to(self.device)
                with torch.set_grad_enabled(True):
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.model.batch_forward(data, labels,
                                       centroids=self.model.memory['centroids'],
                                       phase='train')
                    self.model.batch_loss(labels)
                    self.model.batch_backward()

                # pred = self.model(data)
                # loss = F.cross_entropy(pred, label)/self.update_opts.batch_factor
                # loss.backward()
                # if (j+1) % self.update_opts.batch_factor == 0:
                #     self.optimizer.step()
                #     self.model.zero_grad()

        torch.cuda.empty_cache()
        # reset to eval mode
        for model in self.model.networks.values():
            model.eval()

def create_trainer(model, device, offline_dataset, update_opts, class_map):
    if update_opts.trainer == 'batch':
        trainer = BatchTrainer(model, device, update_opts, offline_dataset)
    elif update_opts.trainer == 'finetune':
        trainer = FineTune(model, device, update_opts, offline_dataset)
    elif update_opts.trainer == 'knn':
        trainer = CentroidTrainer(model, device, update_opts, offline_dataset)
    elif update_opts.trainer == 'split':
        trainer = SplitTrainer(model, device, update_opts, offline_dataset)
    elif update_opts.trainer == 'none':
        trainer = NoTrain(model, device, update_opts, offline_dataset)
    elif update_opts.trainer == 'hybrid':
        trainer = HybridTrainer(model, device, update_opts, offline_dataset, class_map)
    else: 
        sys.exit("Given Trainer not currently specified. Check your --trainer argument.")
    return trainer



    
    
    
    
    

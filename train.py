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



def train(model, training_dataset, training_loader, optimizer, args):
    model = model.train()
    training_dataset.update()
    for i in range(args.epochs):
        for j, (data, label) in enumerate(training_loader):
            data = data.cuda()
            label = label.cuda()
            pred = model(data)
            loss = F.cross_entropy(pred, label)/args.batch_factor
            print(loss.item())
            loss.backward()
            if (j+1) % args.batch_factor == 0:
                optimizer.step()
                model.zero_grad()
    model = model.eval()
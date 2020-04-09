import argparse
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


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
def ImagenetEval(model):
    parser = Options()
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu[0])
    model.to(device)

    #TODO: debug multi-gpu training
    # if len(args.gpu) > 1:
    #     model = nn.DataParallel(model)

    test_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    training_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    data_distribution = construct_distribution()
    dataset = ContinuousDataset(root = args.root, transform = test_tf, distribution = data_distribution)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers=0)
    training_dataset = OfflineDataset(dataset, training_tf)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=int(256/args.batch_factor),
                                                  shuffle=True, num_workers=8, pin_memory=True)

    tracker = OnlineMetricTracker(args.experiment_name, len(dataset), dataset.imgs_per_class, args.num_classes, args.root)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.m,
                                    weight_decay=1e-4,)
    correct_log = np.zeros(len(dataset), dtype = bool)
    for i, (batch, label, seen) in enumerate(loader):
        batch = batch.cuda()
        prediction = model(batch)
        tracker.track(prediction, label, seen)

        if i % args.training_interval == 0 and i > 0:
            print('Training after sample: {}'.format(i))
            print('Current accuracy: {}'.format(np.sum(correct_log[i-1000:i])/1000))
            train(model, training_dataset, training_loader, optimizer, args)
    tracker.write_metrics()


def train(model, training_dataset, training_loader, optimizer, args):
    model = model.train()
    training_dataset.update()
    for i in range(args.epochs):
        for j, (data, label) in enumerate(training_loader):
            data = data.cuda()
            label = label.cuda()
            pred = model(data)
            loss = F.cross_entropy(pred, label)/args.batch_factor
            loss.backward()
            if (j+1) % args.batch_factor == 0:
                optimizer.step()
                model.zero_grad()
    model = model.eval()
    return model



def construct_distribution():
    n = 50
    x = np.arange(1 + n, 1001 + n)
    distribution = (n / x)
    np.random.shuffle(distribution)
    return distribution

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--training_interval', type=int, default=5000)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--m', type=float, default=0.1)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--epochs', type=int, default=2)
        parser.add_argument('--num_classes', type=int, default=1000)
        parser.add_argument('--batch_factor', type=int, default=2)
        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()
        args.gpu = [int(x) for x in args.gpu.split(' ')]
        return args

class OnlineMetricTracker():
    def __init__(self, experiment_name, data_length, imgs_per_class, num_classes = 1000, root = ''):
        self.ood_correct = 0 #out of distribution accuracy
        self.accuracy_log = correct_log = np.zeros(data_length, dtype = bool)
        self.per_class_acc = np.zeros(num_classes)
        self.imgs_per_class = imgs_per_class
        self.experiment_name = experiment_name
        self.write_path = os.path.join(root, experiment_name)
        self.counter

    def write_metrics(self):
        write_path = os.path.join(self.write_path, self.experiment_name)
        np.save(os.path.join(write_path, 'ood_acc'), self.ood_correct/self.num_classes)
        np.save(os.path.join(write_path, 'accuracy_log'), self.accuracy_log)
        np.save(os.path.join(write_path, 'per_class_acc'), self.per_class_acc)
        np.save(os.path.join(write_path, 'imgs_per_class'), self.imgs_per_class)

    def track(self, pred, label, seen):
        correct = (torch.argmax(pred).item() == label).item()
        self.accuracy_log[self.counter] = correct
        self.per_class_acc[label] += correct
        if not seen:
            self.ood_correct += correct
        self.counter += 1



if __name__ == "__main__":
    model = models.resnet18()
    ImagenetEval(model)
import argparse
import os.path as osp
import datasets
import numpy as np
import os
import torch
from datasets import ContinuousDatasetRF, OfflineDatasetRF
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
from train import train
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
def sequential_eval(model, update):
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
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = ContinuousDatasetRF(args.data_root, test_tf, args.sequence_num)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers=0)
    training_dataset = OfflineDatasetRF(args.data_root, train_tf, args.sequence_num)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=int(args.offline_batch_size/args.batch_factor),
                                                  shuffle=True, num_workers=8, pin_memory=True)
    tracker = OnlineMetricTracker(args.experiment_name, dataset.imgs_per_class, args.num_classes, args.result_path)
    optimizer = torch.optim.SGD(model.fc.parameters(), args.lr,
                                    momentum=args.m,
                                    weight_decay=1e-4,)
    for i, (batch, label, seen) in enumerate(loader):
        batch = batch.to(device)
        prediction = model(batch)
        tracker.track(prediction, label, seen)

        if (i+1) % args.training_interval == 0 :
            print('Training after sample: {}'.format(i))
            print('Current accuracy: {}'.format(np.sum(tracker.accuracy_log)/i))
            #training_dataset.update(dataset.get_samples_seen())
            #update(model, training_dataset, training_loader, optimizer, args)
    tracker.write_metrics()

# def construct_distribution():
#     n = 50
#     x = np.arange(1 + n, 1001 + n)
#     distribution = (n / x)
#     np.random.shuffle(distribution)
#     return distribution

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--training_interval', type=int, default=5000)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--m', type=float, default=0.1)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--data_root', type=str, default='')
        parser.add_argument('--result_path', type=str, default='results')
        parser.add_argument('--epochs', type=int, default=2)
        parser.add_argument('--num_classes', type=int, default=1000)
        parser.add_argument('--batch_factor', type=int, default=2)
        parser.add_argument('--sequence_num', type=int, default=2)
        parser.add_argument('--online_batch_size', type=int, default=1)
        parser.add_argument('--offline_batch_size', type=int, default=256)
        parser.add_argument('--experiment_name', type=str, default='Test')
        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()
        args.gpu = [int(x) for x in args.gpu.split(' ')]
        return args

class OnlineMetricTracker():
    def __init__(self, experiment_name, imgs_per_class, num_classes = 1000, root = ''):
        self.ood_correct = 0 #out of distribution accuracy
        self.accuracy_log = []
        self.per_class_acc = np.zeros(num_classes)
        self.imgs_per_class = imgs_per_class
        self.experiment_name = experiment_name
        self.write_path = os.path.join(root, experiment_name)
        self.num_classes = num_classes
        self.counter = 0
        if not os.path.isdir(self.write_path):
            os.mkdir(self.write_path)

    def write_metrics(self):
        np.save(os.path.join(self.write_path, 'ood_acc'), self.ood_correct/self.num_classes)
        np.save(os.path.join(self.write_path, 'accuracy_log'), self.accuracy_log)
        np.save(os.path.join(self.write_path, 'per_class_acc'), self.per_class_acc/self.imgs_per_class)
        np.save(os.path.join(self.write_path, 'imgs_per_class'), self.imgs_per_class)

    def track(self, pred, label, seen):
        correct = (torch.argmax(pred).item() == label)
        self.accuracy_log.append(correct)
        self.per_class_acc[label] += correct
        if not seen:
            self.ood_correct += correct
        self.counter += 1


if __name__ == "__main__":
    model = models.resnet18(pretrained = True)
    model.eval()
    sequential_eval(model, train)
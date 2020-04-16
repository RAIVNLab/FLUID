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
from train import train, NearestNeighbor
from models import KNN
import warnings
from options import Options
from metrics import OnlineMetricTracker

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
def sequential_eval(model, trainer):
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

    dataset = ContinuousDatasetRF(args.root, test_tf, args.sequence_num)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers=0)
    training_dataset = OfflineDatasetRF(args.root, train_tf, args.sequence_num)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=int(args.offline_batch_size/args.batch_factor),
                                                  shuffle=True, num_workers=8, pin_memory=True)
    tracker = OnlineMetricTracker(args.experiment_name, dataset.imgs_per_class, args.num_classes, args.result_path)
    if args.finetune:
        params = model.fc.parameters()
    else:
        params = model.parameters()
    optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.m,
                                    weight_decay=1e-4,)
    #model = KNN(model, device, args)
    #trainer = trainer(model, device, args, training_loader)
    for i, (batch, label, seen) in enumerate(loader):
        batch = batch.to(device)
        prediction = model(batch)
        tracker.track(prediction, label, seen)

        if (i+1) % args.training_interval == 0 :
            print('Training after sample: {}'.format(i))
            print('Current accuracy: {}'.format(np.sum(tracker.accuracy_log[i-1000:i])/1000))
            training_dataset.update(dataset.get_samples_seen())
            #trainer.update()
            #update
            trainer(model, training_dataset, training_loader, optimizer, device, args)
    tracker.write_metrics()
    print(len(tracker.accuracy_log))
    print(tracker.counter)


if __name__ == "__main__":
    model = models.resnet18(pretrained = True)
    model.eval()
    #trainer = NearestNeighbor
    #sequential_eval(model, NearestNeighbor)
    sequential_eval(model, train)
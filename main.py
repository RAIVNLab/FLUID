import datasets
import numpy as np
import os
import torch
from datasets import ContinuousDatasetRF, OfflineDatasetRF
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import create_train_transform, create_test_transform, log_settings, create_novel_class_map
from models import create_model
from trainer import create_trainer
import warnings
from options import Options
from metrics import OnlineMetricTracker
from utils import extract_layers
import sys

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
sys.path.insert(0, os.path.abspath("OLTR1/"))
from run_networks import model as oltr_model


def sequential_eval(model, trainer, online_dataset, tracker, args):
    print("Sequential eval just started")
    device = torch.device('cuda', args.sys_opts.gpu[0])
    online_loader = torch.utils.data.DataLoader(online_dataset, batch_size = 1, shuffle = True, num_workers=0)
    #Note: Shuffle is set to True, but data order is fixed. See ContinuousDatasetRF.__getitem__(). 
    for i, (batch, label, seen) in enumerate(online_loader):
        batch = batch.to(device)
        model.batch_forward(batch, label,
                           centroids=model.memory['centroids'],
                           phase='test')
        prediction = model.logits
        tracker.track(prediction, label, seen)
        if (i+1) % args.online_opts.training_interval == 0:
            trainer.update_dataset(online_dataset.get_samples_seen())
            trainer.update_model()
        if (i+1) % args.sys_opts.log_interval == 0:
            j = online_dataset.get_samples_seen()
            acc = tracker.current_accuracy(args.sys_opts.log_interval, j)
            print('Training after sample: {}'.format(j))
            print('Current accuracy: {}'.format(acc))
            tracker.write_metrics()
    tracker.write_metrics()


if __name__ == "__main__":
    args = Options()
    args.parse_args()
    device = torch.device('cuda', args.sys_opts.gpu[0])
    print("---After device")
    model = create_model(args.model_opts, args.sys_opts, device)
    print("---After model creation")
    class_map_novel = create_novel_class_map(args.sys_opts.root, args.sys_opts.sequence_num)

    train_tf = create_train_transform()
    test_tf = create_test_transform()
    online_dataset = ContinuousDatasetRF(args.sys_opts.root, test_tf, args.sys_opts.sequence_num)
    offline_dataset = OfflineDatasetRF(args.sys_opts.root, train_tf, args.sys_opts.sequence_num)
    print("After creating datasets")
    trainer = create_trainer(model, device, offline_dataset, args.update_opts, class_map_novel)
    imgs_per_class = np.load(os.path.join(args.sys_opts.root, 'S' + str(args.sys_opts.sequence_num)
                                          + '/' + 'imgs_per_class' + str(args.sys_opts.sequence_num) + '.npy'))
    tracker = OnlineMetricTracker(args.sys_opts.experiment_name, imgs_per_class, args.model_opts.num_classes,
                                  args.sys_opts.result_path, args.ood_opts.report_ood)
    tracker.create_experiment_folder()
    args.log_settings()
    sequential_eval(model, trainer, online_dataset, tracker, args)

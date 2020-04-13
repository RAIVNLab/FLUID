import numpy as np
import os
import torch

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
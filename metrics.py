import numpy as np
import os
import torch
import torch.nn.functional as F


class OnlineMetricTracker:
    def __init__(self, experiment_name, imgs_per_class, num_classes = 1000, result_path = '',
                 report_ood=False):
        self.ood_correct = 0  # out of distribution accuracy
        self.report_ood = report_ood
        print("Report OOD = ", self.report_ood)
        self.total_ood = 0
        self.accuracy_log = []
        self.ood_softmax_threshold_log = []
        self.ind_softmax_threshold_log = []
        self.ood_logits_threshold_log = []
        self.ind_logits_threshold_log = []
        self.per_class_acc = np.zeros(num_classes)
        self.per_class_samples = np.zeros(num_classes)
        self.experiment_name = experiment_name
        self.write_path = os.path.join(result_path, experiment_name)
        self.num_classes = num_classes
        self.counter = 0
        self.imgs_per_class = imgs_per_class
            
    def write_metrics(self):
        print(self.ood_correct, " ", self.total_ood)
        if self.report_ood:
            np.save(os.path.join(self.write_path, 'ood_acc'), self.ood_correct/self.total_ood)
        np.save(os.path.join(self.write_path, 'accuracy_log'), self.accuracy_log)
        np.save(os.path.join(self.write_path, 'ood_softmax_threshold_log'), self.ood_softmax_threshold_log)
        np.save(os.path.join(self.write_path, 'ind_softmax_threshold_log'), self.ind_softmax_threshold_log)
        np.save(os.path.join(self.write_path, 'ood_logits_threshold_log'), self.ood_logits_threshold_log)
        np.save(os.path.join(self.write_path, 'ind_logits_threshold_log'), self.ind_logits_threshold_log)

        np.save(os.path.join(self.write_path, 'per_class_acc'), self.per_class_acc/self.imgs_per_class)

    def track(self, pred, label, seen):
        correct = (torch.argmax(pred).item() == int(label))
        self.accuracy_log.append(correct)
        self.per_class_acc[label] += correct
        # print(pred.size())
        # print(correct, " ", torch.argmax(pred).item(), " ", label, " ", prob[torch.argmax(pred).item()],
        #       torch.sum(prob * prob), " seen  = ", seen)

        if self.report_ood:
            prob = F.softmax(pred[0], dim=0)
            threshold_softmax = float(prob[torch.argmax(pred).item()].detach())
            threshold_logits = float(torch.max(pred).detach())
            if not seen:
                self.ood_correct += correct
                self.total_ood += 1
                self.ood_softmax_threshold_log.append(threshold_softmax)
                self.ood_logits_threshold_log.append(threshold_logits)
            else:
                self.ind_softmax_threshold_log.append(threshold_softmax)
                self.ind_logits_threshold_log.append(threshold_logits)

        self.counter += 1
    
    def current_accuracy(self, interval_length, current_sample):
        start = int(np.clip(current_sample-interval_length, 0, np.inf))
        true_length = current_sample - start
        return np.sum(self.accuracy_log[start:])/true_length

    def create_experiment_folder(self):
        if not os.path.isdir(self.write_path):
            os.mkdir(self.write_path)

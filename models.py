import torchvision.models as models
import torch
import torch.nn as nn
from utils import euclidean_metric

class KNN(nn.Module):
    def __init__(self, model, device, args):
        super().__init__()
        modules=list(model.children())[:-1]
        backbone=nn.Sequential(*modules)
        self.backbone = backbone
        self.centroids = torch.zeros([args.num_classes, 512]).to(device) #Change this to automatically learn the feature size

    def forward(self, x):
        features = self.backbone(x).squeeze().unsqueeze(0)
        logits = euclidean_metric(features, self.centroids)
        return logits

    def features(self, x):
        return self.backbone(x).squeeze()


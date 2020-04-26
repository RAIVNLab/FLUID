import torchvision.models as models
import torch
import torch.nn as nn
import sys
from utils import euclidean_metric, cosine_sim
import numpy as np
import os
from utils import extract_layers

class KNN(nn.Module):
    def __init__(self, model, sim_measure):
        super().__init__()
        self.sim_measure = sim_measure
        test_device = next(model.parameters()).device
        test_val = torch.zeros(1, 3,224,224).to(test_device)
        _, feature_dim = model(test_val).shape
        self.backbone = model
        self.centroids = torch.zeros([1000, feature_dim]) 

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        #features = self.backbone(x).squeeze().unsqueeze(0)
        features = self.backbone(x).view(batch_size,-1)
        logits = self.sim_measure(features, self.centroids)
        return logits

    def features(self, x):
        return self.backbone(x).squeeze()

    def to(self, device):
        self.centroids = self.centroids.to(device)
        self.backbone = self.backbone.to(device)
    
    def initialize_centroids(self, pretrain_classes):
        pass

class SplitModel(nn.Module):
    #TODO generalize to multi-layer
    def __init__(self, model, split_layers, sequence_num, root, num_classes):
        self.num_classes = num_classes
        self.model = extract_backbone(model)
        test_device = next(self.model.parameters()).device
        test_val = torch.zeros(1, 3,224,224).to(test_device)
        _, feature_dim = self.model(test_val).shape
        path = 'S' + str(sequence_num) + '/class_map' + str(sequence_num) + '.npy'
        class_map_base = np.load(os.path.join(root, path), allow_pickle = True).item()
        self.base_idx = torch.tensor([x for x in np.arange(0,1000) if x not in class_map_base.values()])
        self.novel_idx = torch.tensor(list(class_map_base.values()))
        self.params = []
        extract_layers(model, split_layers, self.params)
        final_layer = self.params[-1]
        self.model = extract_backbone(model)
        self.base_classifier = torch.nn.Parameter(final_layer[self.base_idx]) 
        self.base_classifier.requires_grad = False
        self.novel_classifier = torch.nn.Linear(feature_dim, 750)
    

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        features =  self.model(x).view(batch_size,-1)
        output = torch.zeros(batch_size, self.num_classes)
        output[self.novel_idx] = features @ self.base_classifier.T
        output[self.base_idx] = self.novel_classifier(features)
        return output

def create_model(model_opts, sys_opts):
    if model_opts.backbone == 'resnet-18':
        backbone = models.resnet18(pretrained = model_opts.pretrained)
    elif model_opts.backbone == 'resnet-34':
        backbone = models.resnet34(pretrained = model_opts.pretrained)
    elif model_opts.backbone == 'resnet-50':
        backbone = models.resnet50(pretrained = model_opts.pretrained)
    elif model_opts.backbone == 'mobilenetv2':
        backbone = models.mobilenet_v2(pretrained = model_opts.pretrained)
    elif model_opts.backbone == 'densenet-161':
        backbone = models.densenet161(pretrained = model_opts.pretrained)
    else:
        sys.exit("Given model not in predefined set of models")
    if model_opts.classifier == 'knn':
        backbone = extract_backbone(backbone)
        if model_opts.similarity_measure == 'euclidean':
            measure =  euclidean_metric
        elif model_opts.similarity_measure == 'cosine':
            measure = cosine_sim
        model = KNN(backbone, measure)
    elif model_opts.classifier == 'linear':
        model = backbone
    elif model_opts.classifier == 'split':
        model = SplitModel(model, model_opts.split_layers, sys_opts.sequence_num, sys_opts.root, model_opts.num_classes)
    else:
        sys.exit("Given classifier not in predefined set of classifiers")
    return model

def extract_backbone(model):
    """Given a pretrained model from pytorch, return the feature extractor. Some models have 
    additional functional layers in the forward implementation which are not included in children 
    and so those are added back per specification."""
    model_name = str(type(model)).split('.')[-2]
    if model_name == 'resnet':
        modules=list(model.children())[:-1]
        modules.append(nn.Flatten())
        backbone=nn.Sequential(*modules)
    elif model_name == 'mobilenet':
        modules=list(model.children())[:-1]
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        backbone=nn.Sequential(*modules)
    elif model_name == 'densenet':
        modules=list(model.children())[:-1]
        modules.append(nn.ReLU())
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        backbone=nn.Sequential(*modules)
    else: 
        sys.exit("Model not currently implemented for nearest neighbors. See extract_backbone to implement.")
    return backbone
    

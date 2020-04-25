import torchvision.models as models
import torch
import torch.nn as nn
import sys
from utils import euclidean_metric


class KNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        # pretrain_classes, feature_dim = list(model.children())[-1].weight.shape
        # modules=list(model.children())[:-1]
        # backbone=nn.Sequential(*modules)
        test_device = next(model.parameters()).device
        test_val = torch.zeros(1, 3, 224, 224).to(test_device)
        _, feature_dim = model(test_val).shape
        self.backbone = model
        self.centroids = torch.zeros([1000, feature_dim])

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # features = self.backbone(x).squeeze().unsqueeze(0)
        features = self.backbone(x).view(batch_size, -1)
        logits = euclidean_metric(features, self.centroids)
        return logits

    def features(self, x):
        return self.backbone(x).squeeze()

    def to(self, device):
        self.centroids = self.centroids.to(device)
        self.backbone = self.backbone.to(device)

    def initialize_centroids(self, pretrain_classes):
        pass


def create_model(pretrained, architecture, classifier, new_classes_indices):
    if architecture == 'resnet-18':
        backbone = models.resnet18(pretrained=pretrained)
    elif architecture == 'resnet-34':
        backbone = models.resnet34(pretrained=pretrained)
    elif architecture == 'resnet-50':
        backbone = models.resnet50(pretrained=pretrained)
    elif architecture == 'mobilenetv2':
        backbone = models.mobilenet_v2(pretrained=pretrained)
    elif architecture == 'densenet-161':
        backbone = models.densenet161(pretrained=pretrained)
    else:
        sys.exit("Given model not in predefined set of models")
    if classifier == 'knn':
        backbone = extract_backbone(backbone)
        model = KNN(backbone)
    elif classifier == 'linear':
        model = backbone
    elif classifier == 'linear_random_init_all_classes':
        pass
        # backbone = extract_backbone(backbone)
        # model = backbone # TODO
    elif classifier == 'linear_random_init_new_classes':
        model = backbone
        model = randomize_classifier(model, randomize_new=True, new_classes_indices=new_classes_indices)
    else:
        sys.exit("Given classifier not in predefined set of classifiers")
    return model


def randomize_classifier(model, randomize_old=False, randomize_new=False, old_classes_indices=None,
                         new_classes_indices=None):
    model_classifier = list(model.children)[-1]
    random_classifier = nn.Linear(model_classifier.in_features, model_classifier.out_features,
                                  model_classifier.bias is not None)

    if randomize_old:
        # old_classes_indices = [0, 1, 2, 5] #TODO
        model_classifier.weights[old_classes_indices, :] = torch.Tensor(random_classifier[old_classes_indices, :])

    if randomize_new:
        # new_classes_indices = [3, 4, 5] #TODO
        print(len(new_classes_indices), " total new classes")
        model_classifier.weights[new_classes_indices, :] = torch.Tensor(random_classifier[new_classes_indices, :])
    return model


def extract_backbone(model):
    """Given a pretrained model from pytorch, return the feature extractor. Some models have 
    additional functional layers in the forward implementation which are not included in children 
    and so those are added back per specification."""
    model_name = str(type(model)).split('.')[-2]
    if model_name == 'resnet':
        modules = list(model.children())[:-1]
        modules.append(nn.Flatten())
        backbone = nn.Sequential(*modules)
    elif model_name == 'mobilenet':
        modules = list(model.children())[:-1]
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        backbone = nn.Sequential(*modules)
    elif model_name == 'densenet':
        modules = list(model.children())[:-1]
        modules.append(nn.ReLU())
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        backbone = nn.Sequential(*modules)
    else:
        sys.exit("Model not currently implemented for nearest neighbors. See extract_backbone to implement.")
    return backbone

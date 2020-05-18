import numpy as np
import os
import torchvision.transforms as transforms
import torch


def create_sequence(parameter_list):
    pass


def file_to_class(file_name, imagenet_map):
    img_name = file_name.split('/')[-1]
    class_name = img_name.split('_')[0]
    return imagenet_map[class_name]


def create_imagenet_map(root):
    read_path = os.path.join(root, 'imagenet_classes.txt')
    with open(read_path, 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]
    key = dict(zip(class_id_to_key, np.arange(1000)))
    return key


def create_novel_class_map(root, sequence_num):
    tmp_path = 'S' + str(sequence_num) + '/class_map' + str(sequence_num) + '.npy'
    class_map_base = np.load(os.path.join(root, tmp_path), allow_pickle=True).item()

    return class_map_base


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def cosine_sim(a, b):
    eps = 1e-10
    a_norm = a / (a.norm(dim=1)[:, None] + eps)
    b_norm = b / (b.norm(dim=1)[:, None] + eps)
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return torch.mm(a, b.transpose(0, 1)), res


def extract_layers(model, num_layers, params):
    """Extract the paramaters for the last num_layers layers from a pytorch model. Parameters
    for each layer are stored as a list in params"""
    trainable_layers = ["<class 'torch.nn.modules.conv.Conv2d'>",
                        "<class 'torch.nn.modules.linear.Linear'>"]
    children = list(model.children())
    i = 1
    while len(params) < num_layers and i <= len(children):
        child = children[-i]
        if str(type(child)) in trainable_layers:
            params += list(child.parameters())
        if len(list(child.children())) > 0:
            extract_layers(child, num_layers, params)
        i += 1


def remove_classifier(model):
    pass


def create_test_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return test_tf


def create_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf


def log_settings(args, experiment_name, result_path):
    write_path = os.path.join(result_path, experiment_name)
    f = open(os.path.join(write_path, "Settings.txt"), "w")
    f.write(str(args))
    f.close()

import numpy as np
import os

def create_sequence(parameter_list):
    pass

def file_to_class(file_name, imagenet_map):
    img_name = file_name.split('/')[-1]
    class_name = img_name.split('_')[0]
    return imagenet_map[class_name]

def create_imagenet_map(root):
    read_path = os.path.join(root,'imagenet_classes.txt')
    with open(read_path, 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]
    key = dict(zip(class_id_to_key,np.arange(1000)))
    return key
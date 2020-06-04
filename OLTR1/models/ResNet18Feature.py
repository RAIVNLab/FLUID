from OLTR1.models.ResNetFeature import *
from OLTR1.utils import *

def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, test=False, *args):
    
    print('Loading Scratch ResNet 18 Feature Model.')
    resnet18 = PreTrainResnet(BasicBlock, 18, use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)

    return resnet18

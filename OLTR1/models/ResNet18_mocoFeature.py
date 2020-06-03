from OLTR1.models.ResNetFeature import *
from OLTR1.utils import *

def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, test=False, *args):
    
    print('Loading Scratch ResNet 18 Feature Model.')
    model = PreTrainMoco(18, use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            model = init_weights(model=model,
                                    weights_path='OLTR1/logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return model

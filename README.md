# I***N*** TH***E*** WIL***D*** - ***NED***

The PyTorch implementation for the NED learning and evaluation framework. The NED framework aims to more closely simulate real-world learning conditions while naturally conglomerating the objectives of previous learning frameworks such as few-shot, continual, and self-supervised learning. NED is designed to enable research towards general ML systems that incorporate the diverse set of ML techniques developed across subfields which we show is non-trivial. 

To learn more about the framework and the unexpected findings it produces check out the  [paper](https://arxiv.org/abs/2007.02519). 


## Getting Started




Running the code:
main.py evaluates a given model, classifier, and update strategy on one of the sequences of data. 
An example command which runs fine-tuning with a resnet-18 architecture on sequence 2: 

python main.py --classifier linear --backbone resnet-18 --trainer fine-tune --sequence_num 2

NCM with a resnet-50 architecture on sequence 3:

python main.py --classifier NCM --backbone resnet-50 --trainer KNN --sequence_num 2
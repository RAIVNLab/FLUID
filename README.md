# I***N*** TH***E*** WIL***D*** - ***NED***

The PyTorch implementation for the NED learning and evaluation framework. The NED framework aims to more closely simulate real-world learning conditions while naturally conglomerating the objectives of previous learning frameworks such as few-shot, continual, and self-supervised learning. NED is designed to enable research towards general ML systems that incorporate the speciailized techniques and insights made across the diverse set of ML subfields. 

To learn more about the framework and the unexpected findings it produces check out the  [paper](https://arxiv.org/abs/2007.02519). To submit results to the NED Leaderboard visit the [website](https://raivn.cs.washington.edu/projects/InTheWild/).


## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/RAIVNLab/InTheWild.git
cd InTheWild
```

- Install [PyTorch](http://pytorch.org) and other dependencies
  - For pip users, type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
- Download data 
    - Download the compressed data file from [google drive](https://drive.google.com/uc?export=download&id=).
    - The data is organized into 6 folders. The folder ***1k_res*** contains the images from the base classes of ImageNet-1k that are not included in standard ImageNet-1k. ***SequenceN*** contains the images and ***sequenceN.npy*** contains the image order for the nth sequence of data. Sequences 1-2 are for validation and sequences 3-5 are for testing.  
    |-- data <br>
       |--1k_res
       |--S1
        |--sequence1
        |--sequence1.npy
       |--S2
        |--sequence2
        |--sequence2.npy
        **.**
        **.**
       |--S5
        |--sequence5
        |--sequence5.npy



### Running The Code 
##### Code Structure
Main.py deploys an ML system which is updated and evaluated under the incoming stream of data. The ML system consists of a model (in model.py) and an update strategies (in trainer.py). To see the various options or add your own while running main.py see options.py. For more thorough documentation on the structure of the code see overview.md. 

##### Example Commands
Deploy a pretrained ResNet-18 updated with fine-tuning on sequence 2:
```bash
python main.py --classifier linear --backbone resnet-18 --trainer fine-tune --sequence_num 2 --pretrain
```
Nearest Class Mean with ResNet-50 on sequence 4:
```bash
python main.py --classifier NCM --backbone resnet-50 --trainer KNN --sequence_num 4 --pretrain
```
<br />
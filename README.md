# FLUID

The PyTorch implementation for the FLUID learning and evaluation framework. The FLUID framework aims to more closely simulate real-world learning conditions while naturally conglomerating the objectives of previous learning frameworks such as few-shot, continual, and self-supervised learning. FLUID is designed to enable research towards general ML systems that incorporate the specialized techniques and insights made across the diverse set of ML subfields. 

To learn more about the framework and the revealing insights on generalization and current methods [paper](https://arxiv.org/abs/2007.02519). To submit results to the FLUID Leaderboard visit the [website](https://raivn.cs.washington.edu/projects/InTheWild/). For more documentation of the code see [overview.md](overview.md). 


## Getting Started
### Installation
- Clone this repo:
```bash
git clone git@github.com:RAIVNLab/FLUID.git
cd git@github.com:RAIVNLab/FLUID.git
```

- Install [PyTorch](http://pytorch.org) and other dependencies
  - For pip users, type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
- Download data 
    - Download the compressed data file from [google drive](https://drive.google.com/file/d/1IL9NidHS2kBW2rzFNnzL1TIA_zzSdRaW/edit), unzip it, and place it in the InTheWild folder.
    - The data is organized into 6 folders. The folder ***1k_res*** contains the images from the base classes of ImageNet-1k that are not included in standard ImageNet-1k. ***SequenceN*** contains the images and ***sequenceN.npy*** contains the image order for the nth sequence of data. Sequences 1-2 are for validation and sequences 3-5 are for testing.  
    |-- data <br />
       |--1k_res <br />
       |--S1 <br />
        |--sequence1 <br />
        |--sequence1.npy <br />
       |--S2 <br />
        |--sequence2 <br />
        |--sequence2.npy <br />
        **.** <br />
        **.** <br />
       |--S5 <br />
        |--sequence5 <br />
        |--sequence5.npy <br />


### Running The Code 
##### Code Structure
Main.py deploys an ML system which is updated and evaluated under the incoming stream of data. The ML system consists of a model (in model.py) and an update strategies (in trainer.py). For more documentation of the code see [overview.md](overview.md).

##### Example Commands
Deploy a pretrained ResNet-18 updated with fine-tuning on sequence 5:
```bash
python main.py --classifier linear --backbone resnet-18 --trainer finetune --sequence_num 5 --pretrain
```
Nearest Class Mean with pretrained ResNet-50 on sequence 4:
```bash
python main.py --classifier knn --backbone resnet-50 --trainer knn --sequence_num 4 --pretrain 
```
<br />

For questions about the code base, data, or paper email mcw244 at cs.washington.edu

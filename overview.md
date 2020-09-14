# Code Overview
To help users extend the code to their own methods and models we briefly explain the functionality and relevant options for each module. The various options can be found in options.py.

[datasets.py](datasets.py) - Handles the dataloading during the streaming phase and offline training phase. Meta-training for ImageNet-1k pretraining is handled here while pretrained PyTorch models are used for supervised Imagenet-1k pretraining.
* `ContinuousDatasetRF` -  Streams data one at a time from a given sequence.
* `OfflineDatasetRF` Used for the offline training phase. Tracks which data has been sampled by `ContinuousDatasetRF` and only iterates over previously seen data.
* `MetaImageNet` - A meta-training dataloader for ImageNet-1k used to train MAML and Prototypical Networks on ImageNet-1k.
* **Relevant Options**
  * `--sequence_num` - Determines which sequence (1-5) is run. Sequence 1-2 are validation and 3-5 are testing. 
  
[Models.py](Models.py) - Contains the models used and the functions for assembling the network architecture, method, and similarity metric if applicable. 
* `KNN` - Implementation for the Nearest Class Mean method. To be paired with `CentroidTrainer` in Trainer.py
* `Hybrid` - Implementation for the Prototype Tuning method. To be paired with the `HybridTrainer`.
* `create_model` - Assembles the various available options for backbone architecture, similarity metric, and learning method. Add your own model to the list of options here to train them in main.py.
* **Relevant Options**
  * `--backbone` - The architecture used to extract features. Resnet-10, Resnet-18, etc.
  * `--classifier` - The classifier used on the features from the backbone. See `create_model` in models.py for all of the classifiers that can be used. 
  * `--pretrained` - Use the flag `--pretrained` to initialize the model before starting the streaming process. Use `--path_to_model` to specify the path to the pretrained weights. If `--path_to_model` is not given then supervised pretrained models from PyTorch will be loaded. 
  * `--similarity_measure` - The similarity metric used for Nearest Class Mean, Prototypical Networks, and Prototype Tuning. 

[Trainer.py](Trainer.py) - Handles the training routines for offline and online training of models from models.py. 
*   `Centroid Trainer` - Calculates the mean feature vector for each class on the fly order to perform NCM. It can be paired with the `KNN` model and the similarity measure of your choosing. 
*  `HybridTrainer` - Calculates the mean feature vector for NCM, then performs fine-tuning every `--ft_interval` number of samples starting after the first `--transition_num` number of samples. 
* `BatchTrainer` - Trains all layers of a given model on all previously seen streaming data in an offline fashion. Training is performed every `--interval` number of samples seen for `--epochs` number of epochs. Used for standard training in the original paper. 
* `FineTune` - Finetunes `--num_layers` of a given model starting from the final linear layer. Training is performed every `--interval` number of samples seen for `--epochs` number of epochs. Used for finetuning in the original paper.
* `create_trainer` - Creates and returns the trainer given the `update_opts` specified in the command line. 
* **Relevant Options**
  * `--lr` - The learning rate used by the optimizer in trainer. Used for Finetuning, standard training, prototype tuning.  
  * `--m` - The momentum used by the optimizer. Used for Finetuning, standard training, prototype tuning.
  * `--num_layers` - The number of layers trained during finetuning, standard training, and prototype tuning starting from the final linear layer. 
  * `--epochs` - The number of epochs trained at each offline training interval for all methods that use offline training. 
  * `--offline_batch_size` - The number samples per batch while offline training for methods that use offline training.
  * `--batch_factor` - The number of times the gradients are accumulated before taking a gradient step. The true batch size is equal to `--batch_factor` times `--offline_batch_size`. Useful for training with larger batch sizes on GPUs that can't fit large batches on GPU ram. 
  * `--trainer` - The type of trainer that will be used. The options can be found in `create_trainer`.
  * `--transition_num` - The number of samples encountered before switching to finetuning for prototype tuning. 
  * `--ft_interval` - The frequency for training the model for prototype tuning which is used by HybridTrainer. The model is trained every `--ft_interval` number of samples.

[Metrics.py](Metrics.py) - The modules used to calculate and track the metrics reported for the NED evaluation. 
* `OnlineMetricTracker` - Tracks the accuracy, the accuracy for each of the 1000 classes, the logits for the out of distribution samples (used to calculate AUROC and F1 score). 
* **Relevant Options**
    * `--report_ood` - Whether to report the out of distribution metrics.
    
[options.py](options.py) - The parser for all settings used by the various NED modules. 
### Sequence Meta Data 
The sequence, the number of images for each class in the sequence, and the mapping from class name to sequence index are stored in the S*N* folder for the *N*th sequence

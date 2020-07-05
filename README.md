## Data and further instructions for running InTheWild coming soon.

Running the code:
main.py evaluates a given model, classifier, and update strategy on one of the sequences of data. 
An example command which runs fine-tuning with a resnet-18 architecture on sequence 2: 

python main.py --classifier linear --backbone resnet-18 --trainer fine-tune --sequence_num 2

NCM with a resnet-50 architecture on sequence 3:

python main.py --classifier NCM --backbone resnet-50 --trainer KNN --sequence_num 2







Data Folder Structure: <br />
The data and meta data for each sequence should be in the respective folder labeled S(sequence number) <br />
SequenceN: folder containing all samples from 22k needed for sequence N <br />
SequenceN.npy: The file containing the paths to each img in the order of the sequence <br />
<br />

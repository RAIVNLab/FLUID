
import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--training_interval', type=int, default=5000)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--m', type=float, default=0.1)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--root', type=str, default='')
        parser.add_argument('--result_path', type=str, default='results')
        parser.add_argument('--epochs', type=int, default=2)
        parser.add_argument('--num_classes', type=int, default=1000)
        parser.add_argument('--batch_factor', type=int, default=2)
        parser.add_argument('--sequence_num', type=int, default=2)
        parser.add_argument('--online_batch_size', type=int, default=1)
        parser.add_argument('--offline_batch_size', type=int, default=256)
        parser.add_argument('--experiment_name', type=str, default='Test')
        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()
        args.gpu = [int(x) for x in args.gpu.split(' ')]
        return args
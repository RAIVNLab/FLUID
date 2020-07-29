import argparse
import os


class Options():
    def __init__(self):
        #  Offline Options
        self.update_opts = argparse.ArgumentParser()
        offline = self.update_opts.add_argument_group('update options')
        offline.add_argument('--lr', type=float, default=0.1)
        offline.add_argument('--m', type=float, default=0.1)
        offline.add_argument('--num_layers', type=int, default=1,
                             help='Number of layers to fine-tune')
        offline.add_argument('--epochs', type=int, default=2)
        offline.add_argument('--offline_batch_size', type=int, default=256)
        offline.add_argument('--batch_factor', type=int, default=2)
        offline.add_argument('--trainer', type=str, default='batch')
        offline.add_argument('--transition_num', type=int, default=10000)
        offline.add_argument('--ft_interval', type=int, default=5000) #Hybrid training option for how often to fine tune

        #  Online Options
        self.online_opts = argparse.ArgumentParser()
        online = self.online_opts.add_argument_group('online options')
        online.add_argument('--training_interval', type=int, default=5000)
        online.add_argument('--online_batch_size', type=int, default=1)

        #  Modeling Options
        self.model_opts = argparse.ArgumentParser()
        model = self.model_opts.add_argument_group('modeling options')
        model.add_argument('--backbone', type=str, default='resnet-18',
                           help=('The architecture backbone to deploy (Resnet-18), See models.py for options'))
        model.add_argument('--classifier', type=str, default='linear')
        model.add_argument('--num_classes', type=int, default=1000)
        model.add_argument('--pretrained', action='store_true',
                           help='Initialize model with pretraining')
        model.add_argument('--path_to_model', type=str, default=None, help='Allows init with different trained models')
        model.add_argument('--similarity_measure', type=str, default='euclidean')
        model.add_argument('--split_layers', type=int, default=1)

        # system Options
        self.sys_opts = argparse.ArgumentParser()
        sys = self.sys_opts.add_argument_group('system options')
        sys.add_argument('--gpu', type=str, default='0')
        sys.add_argument('--root', type=str, default="")
        sys.add_argument('--result_path', type=str, default='results')
        sys.add_argument('--sequence_num', type=int, default=2)
        sys.add_argument('--experiment_name', type=str, default='Test')
        sys.add_argument('--log_interval', type=int, default=5000)
        sys.add_argument('--load_path', type = str, default = 'models/2020-05-17_214804/model.th')
        
        #Boiler Plate Code
        # online.add_argument('-f', type=str, help = "for debugging in jupyter")
        # model.add_argument('-f', type=str, help = "for debugging in jupyter")
        # offline.add_argument('-f', type=str, help = "for debugging in jupyter")
        # sys.add_argument('-f', type=str, help = "for debugging in jupyter")

        # Out of distribution detection
        self.ood_opts = argparse.ArgumentParser()
        ood_opts = self.ood_opts.add_argument_group('out-of-distribution options')
        ood_opts.add_argument('--report_ood', action='store_true',
                           help='Report ood metrics')

    def parse_args(self):
        self.update_opts = self.update_opts.parse_known_args()[0]
        self.online_opts = self.online_opts.parse_known_args()[0]
        self.model_opts = self.model_opts.parse_known_args()[0]
        self.sys_opts = self.sys_opts.parse_known_args()[0]
        self.sys_opts.gpu = [int(x) for x in self.sys_opts.gpu.split(' ')]
        self.ood_opts = self.ood_opts.parse_known_args()[0]

    def log_settings(self):
        write_path = os.path.join(self.sys_opts.result_path, self.sys_opts.experiment_name)
        f = open(os.path.join(write_path, "Settings.txt"), "w")
        settings = str(self.sys_opts) + str(self.model_opts) + str(self.online_opts) + str(self.update_opts) \
                   + str(self.ood_opts)
        strings = ['Namespace', '(', ')']
        replacements = ['', '', ', ']
        for string, replacement in zip(strings, replacements):
            settings = settings.replace(string, replacement)
        f.write(settings)
        f.close()

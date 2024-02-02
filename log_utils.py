import pickle
import numpy as np


class DepthLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metric_dict = {'train': [], 'eval': []}
        self.np_metric_dict = {}
        self.data_count = {'train': -1, 'eval': -1}

    def set_data_count(self, key, val):
        self.data_count[key] = val

    def add_metric(self, mode, epoch, **kwargs):
        # t_error, r_error is assumed to be float
        if len(self.metric_dict[mode]) == epoch:
            self.metric_dict[mode].append([])
        
        batch_metric = {}
        for k in kwargs.keys():
            batch_metric[k] = kwargs[k]
        
        self.metric_dict[mode][epoch].append(batch_metric)

    def print_last(self, mode, metrics=[]):
        # Print the last entry of designated metrics. Useful for extracting running average values
        last_epoch = len(self.metric_dict[mode]) - 1
        
        if len(metrics) == 0 and len(self.metric_dict[mode]) != 0:
            metrics = self.metric_dict[mode][last_epoch][-1].keys()
        for m in metrics:
            last_val = round(self.metric_dict[mode][last_epoch][-1][m], 4)
            print(f"Mode {mode} {m}: {last_val}")
    
    def convert_np(self):
        valid_mode = [m for m in self.metric_dict.keys() if len(self.metric_dict[m]) != 0]
        for mode in valid_mode:
            num_epoch = len(self.metric_dict[mode])
            num_batch = len(self.metric_dict[mode][0])
            num_metrics = len(self.metric_dict[mode][0][0].keys())
            metric_names = self.metric_dict[mode][0][0].keys()
            self.np_metric_dict[mode] = np.zeros([num_epoch, num_batch, num_metrics])

            for e in range(num_epoch):
                for b in range(num_batch):
                    for m, name in enumerate(metric_names):
                        self.np_metric_dict[mode][e, b, m] = self.metric_dict[mode][e][b][name]


def save_logger(pickle_name, logger: DepthLogger):
    pkl_file = open(pickle_name, 'wb')
    pickle.dump(logger, pkl_file)
    pkl_file.close()


def load_logger(pickle_name):
    pkl_file = open(pickle_name, 'rb')
    logger = pickle.load(pkl_file)
    pkl_file.close()
    return logger

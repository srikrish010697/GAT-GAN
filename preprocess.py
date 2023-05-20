import os

import pandas as pd
import numpy as np
import ast
import re
import argparse
from utils import load_multiple,normalize_data
import torch



def import_data(args):
    if (args.file_type == 'preprocessed_data') | (args.file_type == 'csv') :
        data = load_multiple(args.dir,args.file_type)
        if data is None:
            raise ValueError('Directory empty.')
    else:
        raise ValueError('Invalid preprocessed_data type.')
    return data

def preprocess(args):
    data = import_data(args)
    if args.dataset == 'motor':
        data = data.values
        seq_len = 833  #sampling frequency / frequency of signal = 50000/60
        total_samples = data.shape[0]
        num_samples = int(total_samples / seq_len)
        total_samples = num_samples * seq_len
        temp_data = data[:total_samples].reshape((num_samples, seq_len, data.shape[1]))

        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 52), dtype=int))
        motor_16 = temp_data[:, indices, :]
        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 13), dtype=int))
        motor_64 = temp_data[:, indices, :]
        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 6.5), dtype=int))
        motor_128 = temp_data[:, indices, :]
        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 3.25), dtype=int))
        motor_256 = temp_data[:, indices, :]

        motor_16 = motor_16.reshape(motor_16.shape[0] * motor_16.shape[1], motor_16.shape[2])
        motor_64 = motor_64.reshape(motor_64.shape[0] * motor_64.shape[1], motor_64.shape[2])
        motor_128 = motor_128.reshape(motor_128.shape[0] * motor_128.shape[1], motor_128.shape[2])
        motor_256 = motor_256.reshape(motor_256.shape[0] * motor_256.shape[1], motor_256.shape[2])

        os.makedirs('./preprocessed_data')
        pd.DataFrame(normalize_data(motor_16)).to_csv('./preprocessed_data/motor_16.csv', index=False)
        pd.DataFrame(normalize_data(motor_64)).to_csv('./preprocessed_data/motor_64.csv', index=False)
        pd.DataFrame(normalize_data(motor_128)).to_csv('./preprocessed_data/motor_128.csv', index=False)
        pd.DataFrame(normalize_data(motor_256)).to_csv('./preprocessed_data/motor_256.csv', index=False)

    elif args.dataset == 'traffic' :
        data[0] = data[0].apply(lambda x: list(map(float, re.split(r'[;\s]+', x.strip('[]')))))
        traffic_data = data.values
        traffic_data_new = []
        for i in range(0, len(traffic_data)):
            traffic_data_new.append(traffic_data[i][0])
        traffic_data_new = np.array(traffic_data_new)
        traffic_data_new = traffic_data_new.reshape(440, 144, 963)

        traffic_data_new_256 = np.zeros((traffic_data_new.shape[0], 256, 15))
        for i in range(traffic_data_new.shape[0]):
            for j in range(15):
                traffic_data_new_256[i, :, j] = np.linspace(traffic_data_new[i, :, j].min(),
                                                            traffic_data_new[i, :, j].max(), 256)
        traffic_data_new_16 = traffic_data_new[:, :16, :15]
        traffic_data_new_64 = traffic_data_new[:, :64, :15]
        traffic_data_new_128 = traffic_data_new[:, :128, :15]

        traffic_data_new_16 = traffic_data_new_16.reshape(440 * 16, 15)
        traffic_data_new_64 = traffic_data_new_64.reshape(440 * 64, 15)
        traffic_data_new_128 = traffic_data_new_128.reshape(440 * 128, 15)
        traffic_data_new_256 = traffic_data_new_256.reshape(440 * 256, 15)

        pd.DataFrame(normalize_data(traffic_data_new_16)).to_csv(
            './preprocessed_data/traffic_16.csv', index=False)
        pd.DataFrame(normalize_data(traffic_data_new_64)).to_csv(
            './preprocessed_data/traffic_64.csv', index=False)
        pd.DataFrame(normalize_data(traffic_data_new_128)).to_csv(
            './preprocessed_data/traffic_128.csv', index=False)
        pd.DataFrame(normalize_data(traffic_data_new_256)).to_csv(
            './preprocessed_data/traffic_256.csv', index=False)

    elif args.dataset == 'ecg':
        data = np.concatenate(data, axis=0)
        data = np.log(5 * (data - data.min() + 1))
        data_raw = data.from_numpy(data).float()
        pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
        data_pre = pipeline.transform(data_raw)
        data = data_pre.cpu().detach().numpy()
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        seq_len = 360
        total_samples = data.shape[0]
        num_samples = int(total_samples / seq_len)
        total_samples = num_samples * seq_len
        temp_data = data[:total_samples].reshape((num_samples, seq_len, data.shape[1]))

        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 22.5), dtype=int))
        ecg_16 = temp_data[:, indices, :]

        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 5.625), dtype=int))
        ecg_64 = temp_data[:, indices, :]

        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 2.8125), dtype=int))
        ecg_128 = temp_data[:, indices, :]

        indices = list(np.linspace(0, np.shape(temp_data)[1] - 1, int(np.shape(temp_data)[1] / 1.40625), dtype=int))
        ecg_256 = temp_data[:, indices, :]

        ecg_16 = ecg_16.reshape(ecg_16.shape[0] * ecg_16.shape[1], ecg_16.shape[2])
        ecg_64 = ecg_64.reshape(ecg_64.shape[0] * ecg_64.shape[1], ecg_64.shape[2])
        ecg_128 = ecg_128.reshape(ecg_128.shape[0] * ecg_128.shape[1], ecg_128.shape[2])
        ecg_256 = ecg_256.reshape(ecg_256.shape[0] * ecg_256.shape[1], ecg_256.shape[2])

        pd.DataFrame(ecg_16).to_csv('./preprocessed_data/ecg_16.csv', index=False)
        pd.DataFrame(ecg_64).to_csv('./preprocessed_data/ecg_64.csv', index=False)
        pd.DataFrame(ecg_128).to_csv('./preprocessed_data/ecg_128.csv', index=False)
        pd.DataFrame(ecg_256).to_csv('./preprocessed_data/ecg_256.csv', index=False)

class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x

class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="motor")
    parser.add_argument("--dir", type=str)
    parser.add_argument("--load_multiple", type=bool, default=True)
    parser.add_argument("--file_type", type=str, default="csv")
    args = parser.parse_args()
    preprocess(args)
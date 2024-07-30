import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor
import torch.nn.functional as F

np.random.seed(0)


def replace_values(x):
    # Replace -1 with 1 in the last column
    last_column = x[:, -1]
    last_column = torch.where(last_column == -1, torch.tensor(1), last_column)
    x[:, -1] = last_column

    # Replace -1 with 0
    x = torch.where(x == -1, torch.tensor(0), x)

    return x


def preprocess(data):
    t = pad_sequence([F.pad(tensor(d), (0, 1, 0, 0), mode='constant', value=0) for d in data], batch_first=True,
                     padding_value=-1)
    return torch.stack([replace_values(ti) for ti in t])


class SaladsDataset(Dataset):
    def __init__(self, data):
        self.__data = preprocess(data)
        self.sequence_length = self.__data.shape[1]

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]

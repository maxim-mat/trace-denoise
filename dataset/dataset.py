import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset.imgaug import GetTransforms
from dataset.utils import transform
from torch.nn.utils.rnn import pad_sequence
from torch import tensor

np.random.seed(0)


class SaladsDataset(Dataset):
    def __init__(self, data):
        self.__data = pad_sequence([tensor(d) for d in data], batch_first=True, padding_value=-1)
        self.sequence_length = self.__data.shape[1]

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]

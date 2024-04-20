import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset.imgaug import GetTransforms
from dataset.utils import transform
from torch import tensor

np.random.seed(0)


class SaladsDataset(Dataset):
    def __init__(self, data):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return tensor([[ts, te, *act.toarray()[0]] for ts, te, act in self.__data[idx]])

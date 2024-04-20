import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from dataset.dataset import ImageDataset
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_subset_sampler(dataset, sample_percent):
    dataset_size = len(dataset)
    num_samples = int(dataset_size * sample_percent)
    indices = np.random.permutation(dataset_size)[:num_samples]
    return torch.utils.data.SubsetRandomSampler(indices)


def get_data(args, cfg, pk=None, sample_percent=1):
    if pk is not None:
        with open(pk, 'rb') as f:
            dataloader = pkl.load(f)
            return dataloader
    dataset = ImageDataset(cfg.train_csv, cfg, mode='train')
    sampler = get_subset_sampler(dataset, sample_percent)
    dataloader_train = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, sampler=sampler)
    return dataloader_train


def setup_logging(run_name):
    os.makedirs(os.path.join(r"..\dev\diffusion\models", run_name), exist_ok=True)
    os.makedirs(os.path.join(r"..\dev\diffusion\runs", run_name), exist_ok=True)
    os.makedirs(os.path.join(r"..\dev\diffusion\images", run_name), exist_ok=True)

import argparse
import json
import os
import pickle as pkl
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
import torch.nn as nn
import torch
from tqdm import tqdm

from dataset.dataset import SaladsDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter
from ddpm.ddmp_multinomial import Diffusion
import plotly.express as px
import logging
from denoisers.SimpleDenoiser import SimpleDenoiser
from denoisers.UnetDenoiser import UnetDenoiser
from denoisers.ConvolutionDenoiser import ConvolutionDenoiser
from utils import calculate_metrics
from scipy.stats import wasserstein_distance


@dataclass
class Config:
    data_path: str = None
    summary_path: str = None
    device: str = None
    num_epochs: int = None
    learning_rate: float = None
    num_timesteps: int = None
    num_workers: int = None
    test_every: int = None
    denoiser_hidden: int = None
    denoiser_layers: int = None
    batch_size: int = None
    denoiser: str = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default=r"config.json", help="configuration path")
    parser.add_argument('--save_path', default=None, help="path for checkpoints and results")
    parser.add_argument('--resume', default=0, type=int, help="resume previous run")
    args = parser.parse_args()
    return args


def initialize():
    args = parse_args()
    with open(args.cfg_path, "r") as f:
        cfg_json = json.load(f)
        cfg = Config(**cfg_json)
    with open(cfg.data_path, "rb") as f:
        dataset = pkl.load(f)
    if args.save_path is None:
        args.save_path = cfg.summary_path
    for path in (args.save_path, cfg.summary_path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(cfg.summary_path, "cfg.json"), "w") as f:
        json.dump(cfg_json, f)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return args, cfg, dataset, logger


def save_ckpt(model, opt, epoch, cfg, train_loss, test_loss, best=False):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    torch.save(ckpt, os.path.join(cfg.summary_path, 'last.ckpt'))
    if best:
        torch.save(ckpt, os.path.join(cfg.summary_path, 'best.ckpt'))


def evaluate(diffuser, denoiser, criterion, test_loader, cfg, summary, epoch):
    denoiser.eval()
    total_loss = 0.0
    accs, recalls, precisions, f1s, aucs, dists = [], [], [], [], [], []
    l = len(test_loader)
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.permute(0, 2, 1).to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)
            x_hat = torch.cat(
                [diffuser.denoise(denoiser, xi.unsqueeze(0), ti.unsqueeze(0)) for xi, ti in
                 zip(x_t, t)], dim=0
            )
            x_hat_softmax = torch.softmax(x_hat, dim=1)
            wasserstein_dist = np.mean(
                [wasserstein_distance(xi, xhi) for xi, xhi in
                 zip(torch.argmax(x, dim=1).to('cpu'), torch.argmax(x_hat_softmax, dim=1).to('cpu'))])
            auc = np.mean([roc_auc_score(xi, xhi.transpose(0, 1), average='macro', multi_class='ovr') for xi, xhi in
                           zip(torch.argmax(x, dim=1).to('cpu'), x_hat_softmax.to('cpu'))])
            acc, recall, precision, f1 = calculate_metrics(torch.argmax(x, dim=1).to('cpu'),
                                                           torch.argmax(x_hat_softmax, dim=1).to('cpu'))
            accs.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            aucs.append(auc)
            dists.append(wasserstein_dist)
            summary.add_scalar("dist_test", wasserstein_dist, global_step=epoch * l + i)
            summary.add_scalar("accuracy_test", acc, global_step=epoch * l + i)
            summary.add_scalar("recall_test", recall, global_step=epoch * l + i)
            summary.add_scalar("precision_test", precision, global_step=epoch * l + i)
            summary.add_scalar("f1_test", f1, global_step=epoch * l + i)
            summary.add_scalar("auc_test", auc, global_step=epoch * l + i)
            output = denoiser(x_t, t)
            loss = criterion(output, eps)
            total_loss += loss.item()
            summary.add_scalar("MSE_test", loss.item(), global_step=epoch * l + i)
    average_loss = total_loss / l
    average_acc = np.mean(accs)
    average_recall = np.mean(recalls)
    average_precision = np.mean(precisions)
    average_f1 = np.mean(f1s)
    average_auc = np.mean(aucs)
    average_dist = np.mean(dists)
    denoiser.train()
    return average_loss, average_acc, average_recall, average_precision, average_f1, average_auc, average_dist


def train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg, summary, logger):
    train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, test_f1, test_auc = \
        [], [], [], [], [], [], [], []
    l = len(train_loader)
    test_epoch = 0
    best_loss = float('inf')
    denoiser.train()
    for epoch in tqdm(range(cfg.num_epochs)):
        epoch_loss = 0.0
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.permute(0, 2, 1).to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)  # each item in batch gets different level of noise based on timestep
            output = denoiser(x_t, t)
            loss = criterion(output, eps)
            loss.backward()
            optimizer.step()

            summary.add_scalar("MSE_train", loss.item(), global_step=epoch * l + i)
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / l)

        if epoch % cfg.test_every == 0:
            logger.info("testing epoch")
            test_epoch_loss, test_epoch_acc, test_epoch_recall, test_epoch_precision, test_epoch_f1, test_epoch_auc, \
                test_epoch_dist = evaluate(diffuser, denoiser, criterion, test_loader, cfg, summary, test_epoch)
            test_dist.append(test_epoch_dist)
            test_losses.append(test_epoch_loss)
            test_epoch += 1
            logger.info("saving model")
            save_ckpt(denoiser, optimizer, epoch, cfg, train_losses[-1], test_losses[-1], test_epoch_loss < best_loss)
            best_loss = test_epoch_loss if test_epoch_loss < best_loss else best_loss

    return train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, test_f1, test_auc


def main():
    args, cfg, dataset, logger = initialize()
    salads_dataset = SaladsDataset(dataset)
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=0.8, shuffle=True, random_state=17)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
    )

    diffuser = Diffusion(noise_steps=cfg.num_timesteps)

    if cfg.denoiser == "unet":
        denoiser = UnetDenoiser(in_ch=19, out_ch=19, max_input_dim=salads_dataset.sequence_length).to(
            cfg.device).float()
    elif cfg.denoiser == "conv":
        denoiser = ConvolutionDenoiser(input_dim=19, output_dim=19, num_layers=10).to(cfg.device).float()
    else:
        denoiser = SimpleDenoiser(input_dim=19, hidden_dim=cfg.denoiser_hidden, output_dim=19,
                                  num_layers=cfg.denoiser_layers, time_dim=128, device=cfg.device).to(
            cfg.device).float()
    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    criterion = nn.MSELoss()
    summary = SummaryWriter(cfg.summary_path)

    train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, test_f1, test_auc = \
        train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg, summary, logger)
    px.line(train_losses).write_html(os.path.join(cfg.summary_path, "train_loss.html"))
    px.line(test_losses).write_html(os.path.join(cfg.summary_path, "test_losses.html"))
    px.line(test_dist).write_html(os.path.join(cfg.summary_path, "test_dist.html"))
    px.line(test_acc).write_html(os.path.join(cfg.summary_path, "test_acc.html"))
    px.line(test_precision).write_html(os.path.join(cfg.summary_path, "test_precision.html"))
    px.line(tests_recall).write_html(os.path.join(cfg.summary_path, "tests_recall.html"))
    px.line(test_f1).write_html(os.path.join(cfg.summary_path, "test_f1.html"))
    px.line(test_auc).write_html(os.path.join(cfg.summary_path, "test_auc.html"))


if __name__ == "__main__":
    main()

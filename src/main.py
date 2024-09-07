import argparse
import json
import os
import pickle as pkl
from dataclasses import dataclass
import random

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
import torch.nn as nn
import torch
from tqdm import tqdm

from dataset.dataset import SaladsDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from ddpm.ddpm_multinomial import Diffusion
import plotly.express as px
import logging
from denoisers.SimpleDenoiser import SimpleDenoiser
from denoisers.UnetDenoiser import UnetDenoiser
from denoisers.ConvolutionDenoiser import ConvolutionDenoiser
from denoisers.ConditionalUnetDenoiser import ConditionalUnetDenoiser
from utils import calculate_metrics
from scipy.stats import wasserstein_distance
import warnings

warnings.filterwarnings("ignore")


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
    num_classes: int = None
    batch_size: int = None
    denoiser: str = None
    eval_train: bool = None
    predict_on: str = None
    conditional_dropout: float = None
    mode: str = None
    train_percent: float = None
    seed: int = None

    def __post_init__(self):
        if self.mode == "uncond":
            self.conditional_dropout = 1.0


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
        for i, (x, y) in enumerate(test_loader):
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)
            x_hat = diffuser.sample(denoiser, y.shape[0], cfg.num_classes, denoiser.max_input_dim, y,
                                    cfg.predict_on)
            if i == l:
                with open(os.path.join(cfg.summary_path, f"epoch_{epoch}_batch_{i}_test.pkl"), "wb") as f:
                    pkl.dump({"original": x, "denoised": x_hat}, f)
            x_hat_softmax = torch.softmax(x_hat, dim=1)
            wasserstein_dist = np.mean(
                [wasserstein_distance(xi, xhi) for xi, xhi in
                 zip(torch.argmax(x, dim=1).to('cpu'), torch.argmax(x_hat_softmax, dim=1).to('cpu'))])
            try:
                auc = roc_auc_score(torch.cat([x for x in torch.argmax(x, dim=1).to('cpu')], dim=0),
                                    torch.cat([x for x in x_hat_softmax.to('cpu')], dim=1).transpose(0, 1),
                                    average='macro', multi_class='ovr')
            except Exception as e:
                auc = -1
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
            output = denoiser(x_t, t, y)
            loss = criterion(output, eps) if cfg.predict_on == 'noise' else criterion(output, x)
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
    train_losses, test_losses, test_dist, test_acc, test_precision, test_recall, test_f1, test_auc = \
        [], [], [], [], [], [], [], []
    train_dist, train_acc, train_precision, train_recall, train_f1, train_auc = [], [], [], [], [], []
    l = len(train_loader)
    best_loss = float('inf')
    denoiser.train()
    for epoch in tqdm(range(cfg.num_epochs)):
        epoch_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)  # each item in batch gets different level of noise based on timestep
            if np.random.random() < cfg.conditional_dropout:
                y = None
            output = denoiser(x_t, t, y)
            loss = criterion(output, eps) if cfg.predict_on == 'noise' else criterion(output, x)
            loss.backward()
            optimizer.step()

            summary.add_scalar("MSE_train", loss.item(), global_step=epoch * l + i)
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / l)

        if epoch % cfg.test_every == 0:
            logger.info("testing epoch")
            if cfg.eval_train:
                denoiser.eval()
                with torch.no_grad():
                    sample_index = random.choice(range(len(train_loader)))
                    for i, batch in enumerate(train_loader):
                        if i == sample_index:
                            x, y = batch
                            break
                    x = x.permute(0, 2, 1).to(cfg.device).float()
                    y = y.permute(0, 2, 1).to(cfg.device).float()
                    x_hat = diffuser.sample(denoiser, y.shape[0], cfg.num_classes, denoiser.max_input_dim, y,
                                            cfg.predict_on)
                    if epoch % 100 == 0:
                        with open(os.path.join(cfg.summary_path, f"epoch_{epoch}_train.pkl"), "wb") as f:
                            pkl.dump({"original": x, "denoised": x_hat}, f)
                    x_hat_softmax = torch.softmax(x_hat, dim=1)
                    wasserstein_dist = np.mean(
                        [wasserstein_distance(xi, xhi) for xi, xhi in
                         zip(torch.argmax(x, dim=1).to('cpu'), torch.argmax(x_hat_softmax, dim=1).to('cpu'))]
                    )
                    try:
                        auc = roc_auc_score(torch.cat([x for x in torch.argmax(x, dim=1).to('cpu')], dim=0),
                                            torch.cat([x for x in x_hat_softmax.to('cpu')], dim=1).transpose(0, 1),
                                            average='macro', multi_class='ovr')
                    except Exception as e:
                        auc = -1
                    acc, recall, precision, f1 = calculate_metrics(torch.argmax(x, dim=1).to('cpu'),
                                                                   torch.argmax(x_hat_softmax, dim=1).to('cpu'))
                    train_acc.append(acc)
                    train_recall.append(recall)
                    train_precision.append(precision)
                    train_f1.append(f1)
                    train_auc.append(auc)
                    train_dist.append(wasserstein_dist)
                    summary.add_scalar("dist_train", wasserstein_dist, global_step=epoch * l)
                    summary.add_scalar("accuracy_train", acc, global_step=epoch * l)
                    summary.add_scalar("recall_train", recall, global_step=epoch * l)
                    summary.add_scalar("precision_train", precision, global_step=epoch * l)
                    summary.add_scalar("f1_train", f1, global_step=epoch * l)
                    summary.add_scalar("auc_train", auc, global_step=epoch * l)
                denoiser.train()

            test_epoch_loss, test_epoch_acc, test_epoch_recall, test_epoch_precision, test_epoch_f1, test_epoch_auc, \
                test_epoch_dist = evaluate(diffuser, denoiser, criterion, test_loader, cfg, summary, epoch)
            test_dist.append(test_epoch_dist)
            test_losses.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)
            test_recall.append(test_epoch_recall)
            test_precision.append(test_epoch_precision)
            test_f1.append(test_epoch_f1)
            test_auc.append(test_epoch_auc)
            logger.info("saving model")
            save_ckpt(denoiser, optimizer, epoch, cfg, train_losses[-1], test_losses[-1],
                      test_epoch_loss < best_loss)
            best_loss = test_epoch_loss if test_epoch_loss < best_loss else best_loss

    return (train_losses, test_losses, test_dist, test_acc, test_precision, test_recall, test_f1, test_auc,
            train_acc, train_recall, train_precision, train_f1, train_auc, train_dist)


def main():
    args, cfg, dataset, logger = initialize()
    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    diffuser = Diffusion(noise_steps=cfg.num_timesteps)

    if cfg.denoiser == "unet":
        denoiser = UnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                max_input_dim=salads_dataset.sequence_length).to(cfg.device).float()
    elif cfg.denoiser == "unet_cond":
        denoiser = ConditionalUnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                           max_input_dim=salads_dataset.sequence_length).to(cfg.device).float()
    elif cfg.denoiser == "conv":
        denoiser = ConvolutionDenoiser(input_dim=cfg.num_classes, output_dim=cfg.num_classes, num_layers=10).to(
            cfg.device).float()
    else:
        denoiser = SimpleDenoiser(input_dim=cfg.num_classes, hidden_dim=cfg.denoiser_hidden, output_dim=cfg.num_classes,
                                  num_layers=cfg.denoiser_layers, time_dim=128, device=cfg.device).to(
            cfg.device).float()
    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    criterion = nn.MSELoss() if cfg.predict_on == 'noise' else nn.CrossEntropyLoss()
    summary = SummaryWriter(cfg.summary_path)

    (train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, test_f1, test_auc, train_acc,
     train_recall, train_precision, train_f1, train_auc, train_dist) = \
        train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg, summary, logger)
    px.line(train_losses).write_html(os.path.join(cfg.summary_path, "train_loss.html"))
    px.line(test_losses).write_html(os.path.join(cfg.summary_path, "test_losses.html"))
    px.line(test_dist).write_html(os.path.join(cfg.summary_path, "test_dist.html"))
    px.line(test_acc).write_html(os.path.join(cfg.summary_path, "test_acc.html"))
    px.line(test_precision).write_html(os.path.join(cfg.summary_path, "test_precision.html"))
    px.line(tests_recall).write_html(os.path.join(cfg.summary_path, "tests_recall.html"))
    px.line(test_f1).write_html(os.path.join(cfg.summary_path, "test_f1.html"))
    px.line(test_auc).write_html(os.path.join(cfg.summary_path, "test_auc.html"))
    px.line(train_acc).write_html(os.path.join(cfg.summary_path, "train_acc.html"))
    px.line(train_recall).write_html(os.path.join(cfg.summary_path, "train_recall.html"))
    px.line(train_precision).write_html(os.path.join(cfg.summary_path, "train_precision.html"))
    px.line(train_f1).write_html(os.path.join(cfg.summary_path, "train_f1.html"))
    px.line(train_auc).write_html(os.path.join(cfg.summary_path, "train_auc.html"))
    px.line(train_dist).write_html(os.path.join(cfg.summary_path, "train_dist.html"))

    final_results = {"train":
        {
            "loss": train_losses[-1],
            "acc": train_acc[-1],
            "precision": train_precision[-1],
            "recall": train_recall[-1],
            "f1": train_f1[-1],
            "auc": train_auc[-1],
            "dist": train_dist[-1]
        },
        "test":
            {
                "loss": test_losses[-1],
                "acc": test_acc[-1],
                "precision": test_precision[-1],
                "recall": tests_recall[-1],
                "f1": test_f1[-1],
                "auc": test_auc[-1],
                "dist": test_dist[-1]
            }
    }
    with open(os.path.join(cfg.summary_path, "final_results.json"), "w") as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    main()

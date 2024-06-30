import argparse
import json
import os
import pickle as pkl
from dataclasses import dataclass

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
    sampled_dist = torch.inf
    l = len(test_loader)
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)
            # if i == 0:
            #     x_hat = torch.cat(
            #         [denoise_single(diffuser, denoiser, xi.unsqueeze(0), ti.unsqueeze(0), cfg) for xi, ti in
            #          zip(x_t, t)], dim=0
            #     )
            #     sampled_dist = levenshtein_dist_batch(x_hat, x)
            #     summary.add_scalar("dist_test", sampled_dist, global_step=epoch)
            output = denoiser(x_t, t)
            loss = criterion(output, eps)
            total_loss += loss.item()
            summary.add_scalar("MSE_test", loss.item(), global_step=epoch * l + i)
    average_loss = total_loss / l
    denoiser.train()
    return average_loss, sampled_dist


def train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg, summary, logger):
    train_losses, test_losses, test_dist = [], [], []
    l = len(train_loader)
    test_epoch = 0
    best_loss = float('inf')
    denoiser.train()
    for epoch in tqdm(range(cfg.num_epochs)):
        epoch_loss = 0.0
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(cfg.device).float()
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
            test_epoch_loss, test_epoch_dist = evaluate(diffuser, denoiser, criterion, test_loader, cfg, summary,
                                                        test_epoch)
            test_dist.append(test_epoch_dist)
            test_losses.append(test_epoch_loss)
            test_epoch += 1
            logger.info("saving model")
            save_ckpt(denoiser, optimizer, epoch, cfg, train_losses[-1], test_losses[-1], test_epoch_loss < best_loss)
            best_loss = test_epoch_loss if test_epoch_loss < best_loss else best_loss

    return train_losses, test_losses, test_dist


def main():
    args, cfg, dataset, logger = initialize()
    train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, shuffle=True, random_state=17)
    train_salads = SaladsDataset(train_dataset)
    test_salads = SaladsDataset(test_dataset)

    train_loader = DataLoader(
        train_salads,
        batch_size=cfg.batch_size,
        collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=-1)
    )
    test_loader = DataLoader(
        test_salads,
        batch_size=cfg.batch_size,
        collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=-1)
    )

    diffuser = Diffusion(noise_steps=cfg.num_timesteps)
    denoiser = SimpleDenoiser(input_dim=19, hidden_dim=cfg.denoiser_hidden, output_dim=19,
                              num_layers=cfg.denoiser_layers, time_dim=128, device=cfg.device).to(cfg.device).float()
    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    criterion = nn.MSELoss()
    summary = SummaryWriter(cfg.summary_path)

    train_losses, test_losses, test_dist = train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader,
                                                 cfg, summary, logger)
    px.line(train_losses).write_html(os.path.join(cfg.summary_path, "train_loss.html"))
    px.line(test_losses).write_html(os.path.join(cfg.summary_path, "test_losses.html"))
    px.line(test_dist).write_html(os.path.join(cfg.summary_path, "test_dist.html"))


if __name__ == "__main__":
    main()

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
from ddpm.modules import UNet


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
    parser.add_argument('--save_path', help="path for checkpoints and results")
    parser.add_argument('--resume', default=0, type=int, help="resume previous run")
    args = parser.parse_args()
    return args


def initialize():
    args = parse_args()
    with open(args.cfg_path, "r") as f:
        cfg = Config(**json.load(f))
    with open(cfg.data_path, "rb") as f:
        dataset = pkl.load(f)
    return args, cfg, dataset


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


class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, time_dim, device):
        super(SimpleDenoiser, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).float()
        self.fc = nn.Linear(hidden_dim, output_dim).float()
        self.time_dim = time_dim
        self.device = device
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                input_dim
            ),
        ).float()

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        emb = self.embed_layer(t).unsqueeze(1).repeat(1, x.shape[1], 1)
        combined = (x + emb).float()
        out, _ = self.lstm(combined)
        out = self.fc(out)
        return out


def denoise_single(diffuser, denoiser, x_t, t, cfg):
    x_hat = x_t
    denoiser.eval()
    for i in reversed(range(1, t.item())):
        ti = (torch.ones(1) * i).long().to(cfg.device)
        eps_hat = denoiser(x_hat, ti)
        alpha = diffuser.alpha[t][:, None, None]
        alpha_hat = diffuser.alpha_hat[t][:, None, None]
        x_hat = 1 / torch.sqrt(alpha) * (x_hat - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * eps_hat)
    denoiser.train()
    return x_hat


def levenshtein_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def levenshtein_dist_batch(x, y):
    x, y = torch.argmax(x, dim=-1), torch.argmax(y, dim=-1)
    dist = 0
    for xi, yi in tqdm(zip(x, y)):
        dist += levenshtein_dist(xi.to('cpu'), yi.to('cpu'))
    return dist / x.shape[0]


def evaluate(diffuser, denoiser, criterion, test_loader, cfg, summary, epoch):
    denoiser.eval()
    total_loss = 0.0
    l = len(test_loader)
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)
            x_hat = torch.cat([denoise_single(diffuser, denoiser, xi.unsqueeze(0), ti.unsqueeze(0), cfg) for xi, ti in
                               zip(x_t, t)], dim=0)
            output = denoiser(x_t, t)
            loss = criterion(output, eps)
            total_loss += loss.item()
            summary.add_scalar("MSE_test", loss.item(), global_step=epoch * l + i)
    average_loss = total_loss / l
    denoiser.train()
    return average_loss


def train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg, summary):
    train_losses, test_losses = [], []
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
            test_epoch_loss = evaluate(diffuser, denoiser, criterion, test_loader, cfg, summary, test_epoch)
            test_losses.append(test_epoch_loss)
            test_epoch += 1

            save_ckpt(denoiser, optimizer, epoch, cfg, train_losses[-1], test_losses[-1], test_epoch_loss < best_loss)
            best_loss = test_epoch_loss if test_epoch_loss < best_loss else best_loss

    return train_losses, test_losses


def main():
    args, cfg, dataset = initialize()
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

    train_losses, test_losses = train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg, summary)


if __name__ == "__main__":
    main()

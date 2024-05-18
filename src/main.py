import argparse
import json
import pickle as pkl
from dataclasses import dataclass

from torch.optim import AdamW
import torch.nn as nn
import torch

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


class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, time_dim, device):
        super(SimpleDenoiser, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.time_dim = time_dim
        self.device = device
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                input_dim
            ),
        )

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
        emb = self.embed_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        out, _ = self.lstm(x + emb)
        out = self.fc(out)
        return out


def train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg):
    summary = SummaryWriter()
    for epoch in range(cfg.num_epochs):
        for x in train_loader:
            optimizer.zero_grad()
            x = x.to(cfg.device)
            t = diffuser.sample_timesteps(cfg.num_timesteps).to(cfg.device)
            x_t, eps = diffuser.noise_data(x, t)
            output = denoiser(x_t, t)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()


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

    diffuser = Diffusion()
    denoiser = SimpleDenoiser(input_dim=19, hidden_dim=cfg.denoiser_hidden, output_dim=19,
                              num_layers=cfg.denoiser_layers, time_dim=128, device=cfg.device).to(cfg.device)
    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    criterion = nn.MSELoss()

    train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg)


if __name__ == "__main__":
    main()

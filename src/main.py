import pickle as pkl

from torch.optim import AdamW
import torch.nn as nn

from dataset.dataset import SaladsDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter
from ddpm.ddmp_multinomial import Diffusion
from ddpm.modules import UNet

num_epochs = 1000
batch_size = 10
device = "cuda"


class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SimpleDenoiser, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


def train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg):
    summary = SummaryWriter()
    for epoch in range(cfg.num_epochs):
        for x in train_loader:
            x = x.to(device)
            t = diffuser.sample_timesteps(cfg.num_timesteps)
            x_t, eps = diffuser.noise_data(x, t)


if __name__ == "__main__":
    with open("../data/pickles/50_salads_one_hot.pkl", "rb") as f:
        dataset = pkl.load(f)

    cfg = {}

    train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, shuffle=True, random_state=17)
    train_salads = SaladsDataset(train_dataset)
    test_salads = SaladsDataset(test_dataset)

    train_loader = DataLoader(train_salads, batch_size=batch_size,
                              collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=-1))
    test_loader = DataLoader(test_salads, batch_size=batch_size,
                             collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=-1))

    diffuser = Diffusion()
    denoiser = SimpleDenoiser(input_dim=19, hidden_dim=512, output_dim=19, num_layers=8)
    optimizer = AdamW
    criterion = nn.MSELoss()

    train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, cfg)

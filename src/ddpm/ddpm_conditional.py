import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from Diffusion.utils import *
from Diffusion.modules import UNet_conditional, EMA
import logging
from tensorboardX import SummaryWriter
import json
from easydict import EasyDict as edict
import plotly.graph_objects as go

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels).to(self.device)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None).to(self.device)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    with open(args.cfg_path, 'r') as f:
        cfg = edict(json.load(f))
    with open(os.path.join(r"..\dev\diffusion\runs", args.run_name, f"config.json"), 'w') as f:
        json.dump(cfg, f, indent=4)
    device = args.device
    dataloader = get_data(args, cfg, sample_percent=0.1)
    model = UNet_conditional(input_dim=cfg.long_side, num_classes=len(cfg.num_classes), device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=cfg.long_side, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    images, _ = next(iter(dataloader))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    loss_epochs = []

    for epoch in range(cfg.epoch):
        logging.info(f"Starting epoch {epoch}:")
        running_loss = 0.0
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            logger.add_graph(model, [images, t, labels])
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            running_loss += loss.item()

        loss_epochs.append(running_loss / l)

        # generate images each epoch
        labels = torch.randint(0, 2, (2, len(cfg.num_classes))).long().to(device)
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        plot_images(sampled_images)
        save_images(sampled_images, os.path.join(r"..\dev\diffusion\images", args.run_name, f"{epoch}.jpg"))
        save_images(ema_sampled_images, os.path.join(r"..\dev\diffusion\images", args.run_name, f"{epoch}_ema.jpg"))
        torch.save(model.state_dict(), os.path.join(r"..\dev\diffusion\models", args.run_name, f"ckpt_{epoch}.pt"))
        torch.save(ema_model.state_dict(), os.path.join(r"..\dev\diffusion\models", args.run_name, f"ema_ckpt_{epoch}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(r"..\dev\diffusion\models", args.run_name, f"optim.pt"))

        # torch.save(model.state_dict(), os.path.join(r"..\dev\diffusion\runs", args.run_name, f"ckpt_{epoch}.pt"))
        # torch.save(model.state_dict(), os.path.join(r"..\dev\diffusion\runs", args.run_name, f"ema_ckpt_{epoch}.pt"))

    fig = go.Figure(
        data=go.Scatter(
            x=list(range(cfg.epoch)),
            y=loss_epochs,
            mode='lines+markers',
            name='Training Loss'
        ),
        layout=go.Layout(
            title='Training Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
        )
    )

    fig.show()
    fig.write_image(os.path.join(r"..\dev\diffusion\runs", args.run_name, f"train_loss.png"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "meme"
    args.cfg_path = r"../config/example_EXP.json"
    args.num_workers = 16
    args.device = "cuda:0"
    args.log_minibatch = 100
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)


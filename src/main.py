import argparse
import json
import logging
import os
import pickle as pkl
import random
import warnings
from dataclasses import dataclass
from typing import Tuple
import networkx as nx
import numpy as np
import plotly.express as px
import pm4py
import torch
import torch.nn as nn
import torch_geometric.data
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from dataset.dataset import SaladsDataset
from ddpm.ddpm_multinomial import Diffusion
from denoisers.ConditionalUnetDenoiser import ConditionalUnetDenoiser
from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from denoisers.ConditionalUnetGraphDenoiser import ConditionalUnetGraphDenoiser
from denoisers.ConditionalUnetNodeEmbeddingDenoiser import ConditionalUnetNodeEmbeddingDenoiser
from denoisers.ConvolutionDenoiser import ConvolutionDenoiser
from denoisers.SimpleDenoiser import SimpleDenoiser
from denoisers.UnetDenoiser import UnetDenoiser
from src.denoisers.ConditionalUnetAttentionGraphDenoiser import ConditionalUnetAttentionGraphDenoiser
from src.utils.graph_utils import prepare_process_model_for_hetero_gnn
from utils.initialization import initialize
from utils import calculate_metrics
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset, pad_to_multiple_of_n
from utils.graph_utils import prepare_process_model_for_gnn, get_process_model_reachability_graph_transition_matrix

warnings.filterwarnings("ignore")


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


def evaluate(diffuser, denoiser, criterion, test_loader, transition_matrix, cfg, summary, epoch, extra_criterion=None):
    denoiser.eval()
    total_loss = 0.0
    results_accumulator = {'x': [], 'y': [], 'x_hat': []}
    l = len(test_loader)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            x_hat, matrix_hat = diffuser.sample_with_matrix(denoiser, y.shape[0], cfg.num_classes,
                                                            denoiser.max_input_dim, transition_matrix.shape[-1],
                                                            y, cfg.predict_on)
            results_accumulator['x'].append(x)
            results_accumulator['y'].append(y)
            results_accumulator['x_hat'].append(x_hat.permute(0, 2, 1))
            if not cfg.enable_gnn:
                loss = criterion(x_hat, eps) if cfg.predict_on == 'noise' else criterion(x_hat, x)
            else:
                loss = 0.99 * criterion(x_hat, x) + \
                       0.01 * extra_criterion(
                    matrix_hat.view(x.shape[0], cfg.num_classes + 1, -1),
                    transition_matrix.view(transition_matrix.size(1), -1).repeat(matrix_hat.shape[0], 1, 1)
                )
            total_loss += loss.item()
            summary.add_scalar("MSE_test", loss.item(), global_step=epoch * l + i)

        x_argmax = torch.argmax(torch.cat(results_accumulator['x'], dim=0), dim=1).to('cpu')
        y_cat = torch.cat(results_accumulator['y'], dim=0)
        x_hat_logit = torch.cat(results_accumulator['x_hat'], dim=0)

        x_argmax_flat = x_argmax.reshape(-1).to('cpu')
        x_hat_flat = x_hat_logit.reshape(-1, cfg.num_classes).to('cpu')
        x_hat_prob_flat = torch.softmax(x_hat_flat, dim=1).to('cpu')
        x_hat_argmax_flat = torch.argmax(x_hat_prob_flat, dim=1).to('cpu')
        x_hat_prob = torch.softmax(x_hat_logit, dim=1).to('cpu')
        x_hat_argmax = torch.argmax(x_hat_prob, dim=1)

        auc = roc_auc_score(x_argmax_flat, x_hat_prob_flat, multi_class='ovr', average='macro')
        w2 = np.mean([wasserstein_distance(xi, xhi) for xi, xhi in zip(x_argmax, x_hat_argmax)])
        accuracy = accuracy_score(x_argmax_flat, x_hat_argmax_flat)
        precision = precision_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
        recall = recall_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
        f1 = f1_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
        average_loss = total_loss / l
        with open(os.path.join(cfg.summary_path, f"epoch_{epoch}_test.pkl"), "wb") as f:
            pkl.dump({"original": x, "denoised": x_hat}, f)

        summary.add_scalar("dist_test", w2, global_step=epoch * l)
        summary.add_scalar("accuracy_test", accuracy, global_step=epoch * l)
        summary.add_scalar("recall_test", recall, global_step=epoch * l)
        summary.add_scalar("precision_test", precision, global_step=epoch * l)
        summary.add_scalar("f1_test", f1, global_step=epoch * l)
        summary.add_scalar("auc_test", auc, global_step=epoch * l)
        denoiser.train()
    return average_loss, accuracy, recall, precision, f1, auc, w2


def train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, transition_matrix, cfg, summary, logger,
          extra_criterion=None):
    train_losses, test_losses, test_dist, test_acc, test_precision, test_recall, test_f1, test_auc = \
        [], [], [], [], [], [], [], []
    train_dist, train_acc, train_precision, train_recall, train_f1, train_auc = [], [], [], [], [], []
    l = len(train_loader)
    transition_matrix = transition_matrix.unsqueeze(0)
    # transition_matrix = torch.randn(1, 1, transition_matrix.shape[0], transition_matrix.shape[0])
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
            output, matrix_hat = denoiser(x_t, t, y)
            if not cfg.enable_gnn:
                loss = criterion(output, eps) if cfg.predict_on == 'noise' else criterion(output, x)
            else:
                loss = 0.99 * criterion(output, x) + \
                       0.01 * extra_criterion(
                    matrix_hat.view(x.shape[0], cfg.num_classes + 1, -1),
                    transition_matrix.view(transition_matrix.size(1), -1).repeat(matrix_hat.shape[0], 1, 1)
                )
            loss.backward()
            optimizer.step()

            summary.add_scalar("MSE_train", loss.item(), global_step=epoch * l + i)
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / l)

        if epoch % cfg.test_every == 0:
            logger.info("testing epoch")
            if cfg.eval_train:
                denoiser.eval()
                with (torch.no_grad()):
                    sample_index = random.choice(range(len(train_loader)))
                    for i, batch in enumerate(train_loader):
                        if i == sample_index:
                            x, y = batch
                            break
                    x = x.permute(0, 2, 1).to(cfg.device).float()
                    y = y.permute(0, 2, 1).to(cfg.device).float()
                    x_hat, _ = diffuser.sample_with_matrix(denoiser, y.shape[0], cfg.num_classes,
                                                                    denoiser.max_input_dim,
                                                                    transition_matrix.shape[-1], y, cfg.predict_on)
                    x_hat = x_hat.permute(0, 2, 1)
                    x_argmax = torch.argmax(x, dim=1).to('cpu')
                    x_argmax_flat = torch.argmax(x, dim=1).reshape(-1).to('cpu')
                    x_hat_flat = x_hat.reshape(-1, cfg.num_classes).to('cpu')
                    x_hat_prob_flat = torch.softmax(x_hat_flat, dim=1).to('cpu')
                    x_hat_argmax_flat = torch.argmax(x_hat_prob_flat, dim=1).to('cpu')
                    x_hat_prob = torch.softmax(x_hat, dim=1).to('cpu')
                    x_hat_argmax = torch.argmax(x_hat_prob, dim=1)
                    auc = roc_auc_score(x_argmax_flat, x_hat_prob_flat, multi_class='ovr', average='micro')
                    w2 = np.mean([wasserstein_distance(xi, xhi) for xi, xhi in zip(x_argmax, x_hat_argmax)])
                    accuracy = accuracy_score(x_argmax_flat, x_hat_argmax_flat)
                    precision = precision_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
                    recall = recall_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
                    f1 = f1_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
                    with open(os.path.join(cfg.summary_path, f"epoch_{epoch}_train.pkl"), "wb") as f:
                        pkl.dump({"original": x, "denoised": x_hat}, f)
                    train_acc.append(accuracy)
                    train_recall.append(recall)
                    train_precision.append(precision)
                    train_f1.append(f1)
                    train_auc.append(auc)
                    train_dist.append(w2)
                    summary.add_scalar("dist_train", w2, global_step=epoch * l)
                    summary.add_scalar("accuracy_train", accuracy, global_step=epoch * l)
                    summary.add_scalar("recall_train", recall, global_step=epoch * l)
                    summary.add_scalar("precision_train", precision, global_step=epoch * l)
                    summary.add_scalar("f1_train", f1, global_step=epoch * l)
                    summary.add_scalar("auc_train", auc, global_step=epoch * l)
                denoiser.train()

            test_epoch_loss, test_epoch_acc, test_epoch_recall, test_epoch_precision, test_epoch_f1, test_epoch_auc, \
                test_epoch_dist = evaluate(diffuser, denoiser, criterion, test_loader, transition_matrix, cfg, summary,
                                           epoch, extra_criterion)
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
    logger.info(f"train size: {len(train_dataset)} test size: {len(test_dataset)}")
    rg_transition_matrix = None
    metadata = None
    extra_criterion = None
    if cfg.enable_gnn:
        dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                                  preprocess=remove_duplicates_dataset)
        rg_nx, rg_transition_matrix = get_process_model_reachability_graph_transition_matrix(dk_process_model,
                                                                                             dk_init_marking)
        rg_transition_matrix = torch.tensor(rg_transition_matrix, device=cfg.device).float()
        rg_transition_matrix = pad_to_multiple_of_n(rg_transition_matrix)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    diffuser = Diffusion(noise_steps=cfg.num_timesteps)

    if cfg.denoiser == "unet":
        denoiser = UnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                max_input_dim=salads_dataset.sequence_length).to(cfg.device).float()
    elif cfg.denoiser == "unet_cond":
        if cfg.enable_gnn:
            denoiser = ConditionalUnetMatrixDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                     max_input_dim=salads_dataset.sequence_length,
                                                     transition_dim=rg_transition_matrix.shape[-1]).to(
                cfg.device).float()
        else:
            denoiser = ConditionalUnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                               max_input_dim=salads_dataset.sequence_length).to(cfg.device).float()
    elif cfg.denoiser == "conv":
        denoiser = ConvolutionDenoiser(input_dim=cfg.num_classes, output_dim=cfg.num_classes, num_layers=10).to(
            cfg.device).float()
    else:
        denoiser = SimpleDenoiser(input_dim=cfg.num_classes, hidden_dim=cfg.denoiser_hidden, output_dim=cfg.num_classes,
                                  num_layers=cfg.denoiser_layers, time_dim=128, device=cfg.device).to(
            cfg.device).float()
    if cfg.parallelize:
        denoiser = nn.DataParallel(denoiser, device_ids=[0, 1])

    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    criterion = nn.MSELoss() if cfg.predict_on == 'noise' else nn.CrossEntropyLoss()
    if cfg.enable_gnn:
        extra_criterion = nn.CrossEntropyLoss()
    summary = SummaryWriter(cfg.summary_path)

    (train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, test_f1, test_auc, train_acc,
     train_recall, train_precision, train_f1, train_auc, train_dist) = \
        train(diffuser, denoiser, optimizer, criterion, train_loader, test_loader, rg_transition_matrix,
              cfg, summary, logger, extra_criterion=extra_criterion)

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

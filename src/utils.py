import argparse
import json
import logging
import multiprocessing
import os
from typing import Callable
from functools import partial
import numpy as np
import pandas as pd
import pm4py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from numpy import mean
from tqdm import tqdm
import pickle as pkl

import sktr
from src.Config import Config
from src.dataset.dataset import SaladsDataset
from sktr.sktr import convert_dataset_to_df, sfmx_mat_to_sk_trace, from_discovered_model_to_PetriNet, \
    recover_single_trace


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
    for xi, yi in zip(x, y):
        dist += levenshtein_dist(xi.to('cpu'), yi.to('cpu'))
    return dist / x.shape[0]


def calculate_metrics(y_true, y_pred):
    accs, recalls, precisions, f1s = [], [], [], []
    for yti, ypi in zip(y_true, y_pred):
        accs.append(accuracy_score(yti, ypi))
        precisions.append(precision_score(yti, ypi, average='macro', zero_division=0))
        recalls.append(recall_score(yti, ypi, average='macro', zero_division=0))
        f1s.append(f1_score(yti, ypi, average='macro', zero_division=0))
        # aucs.append(roc_auc_score(yti, ypi, average='macro', multi_class='ovr'))
    return mean(accs), mean(recalls), mean(precisions), mean(f1s)


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])

    return df_copy


def convert_dataset_to_train_process_df(dataset: SaladsDataset, cfg: Config):
    dk_process_df, _ = convert_dataset_to_df(dataset, cfg.activity_names)
    return prepare_df_cols_for_discovery(dk_process_df)


def convert_dataset_to_stochastic_traces(dataset: SaladsDataset, cfg: Config):
    _, stochastic_list = convert_dataset_to_df(dataset, cfg.activity_names)
    return stochastic_list


def resolve_process_discovery_method(method_name: str) -> Callable:
    match method_name:
        case "inductive":
            return pm4py.discover_petri_net_inductive
        case _:
            raise AttributeError(f"Unsupported discovery method: {method_name}")


def subsample_time_series(trace_data: dict, num_indexes: int, axis: int = 0):
    """
    reduces trace lengths to num_indexes
    :param trace_data: original dk and sk traces
    :param num_indexes: length of new, under-sampled, sequences
    :param axis: axis along which to sample
    :return: reduced length df and sk traces and respective sampled indexes
    """
    dk_sample, sk_sample, sample_indexes = [], [], []
    for dk_trace, sk_trace in zip(trace_data['target'], trace_data['stochastic']):
        if dk_trace.shape[0] != sk_trace.shape[0]:
            raise ValueError(f"trace lengths: {dk_trace.shape[0]} (dk), {sk_trace.shape[0]} (sk) missmatch")
        random_indexes = sorted(np.random.choice(dk_trace.shape[axis], min(num_indexes, len(dk_trace)), replace=False))
        dk_sample.append(dk_trace[random_indexes])
        sk_sample.append(sk_trace[random_indexes])
        sample_indexes.append(random_indexes)
    return {'target': dk_sample, 'stochastic': sk_sample}, sample_indexes


def train_sktr(dataset: SaladsDataset, cfg: Config) -> sktr.sktr.PetriNet:
    """
    run process discovery on the train dataset
    :param dataset: train dataset
    :param cfg:
    :return: process model petri net
    """
    df_train = convert_dataset_to_train_process_df(dataset, cfg)
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(df_train)
    model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=1)
    return model


def discover_dk_process(dataset: SaladsDataset, cfg: Config) -> tuple[pm4py.PetriNet, pm4py.Marking, pm4py.Marking]:
    df_train = convert_dataset_to_train_process_df(dataset, cfg)
    process_discovery_method = resolve_process_discovery_method(cfg.process_discovery_method)
    return process_discovery_method(df_train)


def process_sk_trace(sk_trace: pd.DataFrame, activity_names, round_precision, model, non_sync_penalty):
    print("started processing trace")
    recovered_trace = recover_single_trace(
        sfmx_mat_to_sk_trace(sk_trace, 0, activity_names=activity_names, round_precision=round_precision),
        model, non_sync_penalty, activity_names
    )
    print("ended processing trace")
    return recovered_trace


def evaluate_sktr_on_dataset(dataset: SaladsDataset, model: sktr.sktr.PetriNet, cfg: Config):
    stochastic_traces_matrices = convert_dataset_to_stochastic_traces(dataset, cfg)
    args_list = [
        (sk_trace, cfg.activity_names, cfg.round_precision, model, 1)
        for sk_trace in stochastic_traces_matrices
    ]
    with multiprocessing.Pool(processes=min(1, os.cpu_count() - 4)) as pool:
        recovered_traces = list(
            pool.starmap(process_sk_trace, args_list)
        )

    return recovered_traces

from typing import Callable

import pandas as pd
import pm4py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from numpy import mean

from src.Config import Config
from src.dataset.dataset import SaladsDataset


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


def convert_dataset_to_df(dataset: SaladsDataset, activity_names: dict):
    deterministic, stochastic = torch.stack([x[0] for x in dataset], axis=0), torch.stack([x[1] for x in dataset],
                                                                                          axis=0)
    deterministic = torch.argmax(deterministic.permute(0, 2, 1), dim=1)
    stochastic = stochastic.permute(0, 2, 1)

    df_deterministic = pd.DataFrame(
        {
            'concept:name': [activity_names[i.item()] for trace in deterministic for i in trace],
            'case:concept:name': [str(i) for i, trace in enumerate(deterministic) for _ in range(len(trace))]
        }
    )

    stochastic_list = [x.unsqueeze(0) for x in stochastic]

    return df_deterministic, stochastic_list


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])

    return df_copy


def convert_dataset_to_train_process_df(dataset: SaladsDataset, cfg: Config):
    (dk_process_df, _), (_, _) = convert_dataset_to_df(dataset, cfg.activity_names)
    return prepare_df_cols_for_discovery(dk_process_df)


def resolve_process_discovery_method(method_name: str) -> Callable:
    match method_name:
        case "inductive":
            return pm4py.discover_petri_net_inductive
        case _:
            raise AttributeError(f"Unsupported discovery method: {method_name}")

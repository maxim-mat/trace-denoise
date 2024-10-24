import multiprocessing
import os
from typing import Callable

import numpy as np
import pandas as pd
import pm4py

from src.dataset.dataset import SaladsDataset
from src.sktr.sktr import convert_dataset_to_df, from_discovered_model_to_PetriNet, recover_single_trace, \
    sfmx_mat_to_sk_trace
import src.sktr.sktr
from src.utils import Config


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


def train_sktr(dataset: SaladsDataset, cfg: Config) -> src.sktr.sktr.PetriNet:
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


def evaluate_sktr_on_dataset(dataset: SaladsDataset, model: src.sktr.sktr.PetriNet, cfg: Config):
    stochastic_traces_matrices = convert_dataset_to_stochastic_traces(dataset, cfg)
    args_list = [
        (sk_trace, cfg.activity_names, cfg.round_precision, model, 1)
        for sk_trace in stochastic_traces_matrices
    ]
    with multiprocessing.Pool(processes=cfg.num_workers) as pool:
        recovered_traces = list(
            pool.starmap(process_sk_trace, args_list)
        )

    return recovered_traces
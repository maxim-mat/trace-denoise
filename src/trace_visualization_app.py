from collections import defaultdict

import streamlit as st
import pickle as pkl

import torch
from sklearn.model_selection import train_test_split

from denoisers.ConditionalUnetDenoiser import ConditionalUnetDenoiser
from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from utils.graph_utils import get_process_model_reachability_graph_transition_matrix
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset, pad_to_multiple_of_n
from utils.Config import Config
import plotly.express as px
import plotly.graph_objects as go
from dataset.dataset import SaladsDataset
from ddpm.ddpm_multinomial import Diffusion
from utils.initialization import initialize
import os
import argparse
import json
from streamlit.components.v1 import html
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py import convert_to_bpmn
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=r"D:\Projects\trace-denoise\pretty_results", help="base directory of results")
    args = parser.parse_args()
    return args


def load_experiment_config(target_dir):
    config_path = os.path.join(target_dir, "cfg.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return Config(**json.load(f))
    else:
        st.warning("Configuration file not found.")
        return None


def load_experiment_data_and_model(target_dir, cfg):
    with open(cfg.data_path, "rb") as f:
        base_dataset = pkl.load(f)
    dataset = SaladsDataset(base_dataset['target'], base_dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)
    dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                              preprocess=remove_duplicates_dataset)
    diffuser = Diffusion(noise_steps=cfg.num_timesteps, device=cfg.device)
    if cfg.enable_matrix:
        rg_nx, rg_transition_matrix = get_process_model_reachability_graph_transition_matrix(dk_process_model,
                                                                                             dk_init_marking)
        rg_transition_matrix = torch.tensor(rg_transition_matrix, device=cfg.device).float()
        rg_transition_matrix = pad_to_multiple_of_n(rg_transition_matrix)
        denoiser = ConditionalUnetMatrixDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                 max_input_dim=dataset.sequence_length,
                                                 transition_dim=rg_transition_matrix.shape[-1],
                                                 device=cfg.device).to(cfg.device).float()
    else:
        rg_transition_matrix = torch.randn((cfg.num_classes, 2, 2)).to(cfg.device)
        denoiser = ConditionalUnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                           max_input_dim=dataset.sequence_length,
                                           device=cfg.device).to(cfg.device).float()
    ckpt_path = os.path.join(target_dir, "best.ckpt")
    denoiser.load_state_dict(torch.load(ckpt_path, map_location=cfg.device)['model_state'])
    final_res_path = os.path.join(target_dir, "final_results.json")
    if os.path.exists(final_res_path):
        with open(final_res_path, "r") as f:
            final_res = json.load(f)
    else:
        st.warning("Final results not found.")

    return (train_dataset, test_dataset, dk_process_model, dk_init_marking, dk_final_marking, rg_transition_matrix,
            diffuser, denoiser, final_res)


def organize_html_files_by_metric(html_files):
    metrics_dict = defaultdict(lambda: {"train": None, "test": None})
    for file in html_files:
        parts = file.split("_", 1)
        if len(parts) == 2 and parts[1].endswith(".html"):
            dataset = "test" if parts[0] == "tests" else parts[0]  # train or test
            metric = "loss" if parts[1].replace(".html", "") == "losses" else parts[1].replace(".html", "")
            metrics_dict[metric][dataset] = file
    return metrics_dict


def get_dataset_trace_kinds(dataset, denoiser, diffuser, transition_matrix, target_dir, cfg):
    x = {i: torch.argmax(xi[0], dim=1) for i, xi in enumerate(dataset)}
    y = {i: torch.argmax(xi[1], dim=1) for i, xi in enumerate(dataset)}
    # cursed code for unpickling results with baked in device
    # with open(os.path.join(target_dir, "epoch_2800_train.pkl"), "rb") as f:
    #     raw_data = f.read()
    # fixed_data = raw_data.replace(b'cuda:1', b'cuda:0')
    # accumulator_dict = pkl.loads(fixed_data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    x_hat_accumulator = []
    for _, y in tqdm(loader):
        y = y.permute(0, 2, 1).to(cfg.device).float()
        diffusion_sample, _, _, _, _ = diffuser.sample_with_matrix(denoiser, len(y), cfg.num_classes,
                                                                   denoiser.max_input_dim,
                                                                   transition_matrix.shape[-1], transition_matrix, None,
                                                                   y.to(cfg.device),
                                                                   cfg.predict_on)
        x_hat_accumulator.append(diffusion_sample)
    x_hat_tensor = torch.cat(x_hat_accumulator, dim=0) if cfg.batch_size > 1 else torch.stack(x_hat_accumulator)
    x_hat = {i: xi for i, xi in enumerate(torch.argmax(torch.softmax(x_hat_tensor, dim=1), dim=1))}
    return x, y, x_hat


def visualize_trace(x, y, x_hat):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(x))), y=x, mode='lines', name='Original',
                             line=dict(color='blue', dash='solid')))
    fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode='lines', name='Argmax',
                             line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=list(range(len(x_hat))), y=x_hat, mode='lines', name='Reconstructed',
                             line=dict(color='green', dash='dot')))
    return fig


def main():
    args = parse_args()
    base_dir = args.base_dir

    st.set_page_config(layout="wide")  # Enable wide layout
    st.title("Experiment Results Visualizer")
    experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if experiments:
        selected_subdirectory = st.selectbox("Select an experiment:", experiments)

        # Load and display experiment configuration
        if selected_subdirectory:
            target_dir = os.path.join(base_dir, selected_subdirectory)
            cfg = load_experiment_config(target_dir)
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            (train_dataset, test_dataset, dk_process_model, dk_init_marking, dk_final_marking, transition_matrix,
             diffuser, denoiser, final_res) = load_experiment_data_and_model(target_dir, cfg)

            st.subheader("Final Results Summary")
            st.json(final_res)

            st.subheader("Visualization Plots")
            html_files = [f for f in os.listdir(target_dir) if f.endswith(".html")]
            if html_files:
                metrics_dict = organize_html_files_by_metric(html_files)
                for metric, datasets in metrics_dict.items():
                    st.markdown(f"### Metric: {metric}")
                    col1, col2 = st.columns(2)

                    if datasets["train"]:
                        with col1:
                            st.markdown("#### Train")
                            with open(os.path.join(target_dir, datasets["train"]), "r", encoding="utf-8") as f:
                                plot_html = f.read()
                                html(plot_html, height=600)

                    if datasets["test"]:
                        with col2:
                            st.markdown("#### Test")
                            with open(os.path.join(target_dir, datasets["test"]), "r", encoding="utf-8") as f:
                                plot_html = f.read()
                                html(plot_html, height=600)

            st.subheader("Process Model Visualization")
            try:
                gviz = pn_visualizer.apply(dk_process_model, dk_init_marking, dk_final_marking)
                st.graphviz_chart(gviz.source)
                bpmn = convert_to_bpmn(dk_process_model, dk_init_marking, dk_final_marking)
                bpmn_gviz = bpmn_visualizer.apply(bpmn)
                st.graphviz_chart(bpmn_gviz.source)
            except Exception as e:
                st.error(f"Error generating Petri net visualization: {e}")

            st.subheader("Recovery Explorer")
            selected_dataset = st.selectbox("Select a dataset:", ["train", "test"])
            if selected_dataset:
                # dataset = train_dataset if selected_dataset == "train" else test_dataset
                # x, y, x_hat = get_dataset_trace_kinds(dataset, denoiser, diffuser, transition_matrix, target_dir, cfg)
                with open(os.path.join(target_dir, f"trace_kinds_{selected_dataset}.pkl"), "rb") as f:
                    x, y, x_hat = pkl.load(f)
                selected_trace = st.selectbox("Select a trace:", list(x.keys()))
                if selected_trace:
                    x_trace = x[selected_trace].cpu()
                    y_trace = y[selected_trace].cpu()
                    x_hat_trace = x_hat[selected_trace].cpu()
                    html(visualize_trace(x_trace, y_trace, x_hat_trace).to_html(), height=600)


if __name__ == "__main__":
    main()

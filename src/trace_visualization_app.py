import streamlit as st
import pickle as pkl
from denoisers.ConditionalUnetDenoiser import ConditionalUnetDenoiser
from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from utils.Config import Config
import plotly.express as px
import plotly.graph_objects as go
from dataset.dataset import SaladsDataset
from ddpm.ddpm_multinomial import Diffusion
from utils.initialization import initialize
import os


def do_stuff():
    pass


def main():
    base_dir = "./runs"
    experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


if __name__ == "__main__":
    main()

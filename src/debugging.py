import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset import SaladsDataset
from utils.pm_utils import discover_dk_process
from utils.graph_utils import prepare_process_model_for_gnn
from utils.Config import Config
import json
import pickle as pkl
import pm4py
from pm4py.sim import play_out
from modules.GraphEncoder import GraphEncoder


if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg_json = json.load(f)
        cfg = Config(**cfg_json)

    with open(cfg.data_path, "rb") as f:
        dataset = pkl.load(f)
    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)

    model, initial, final = discover_dk_process(train_dataset, cfg)
    thingy = play_out(model, initial, final, parameters={"NO_TRACES": 5, "MAX_TRACE_LENGTH": 100})  # apply in basic_playout

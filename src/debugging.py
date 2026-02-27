import pickle as pkl
from dataset.dataset import SaladsDataset
import json
from utils.Config import Config
from sklearn.model_selection import train_test_split
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset
from utils.graph_utils import prepare_process_model_for_heterognn


if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg_json = json.load(f)
        cfg = Config(**cfg_json)
    with open("../data/pickles/breakfast_unified.pkl", "rb") as f:
        dataset = pkl.load(f)

    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=0.7, shuffle=True, random_state=42)
    dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                              preprocess=remove_duplicates_dataset)
    pm_nx_data = prepare_process_model_for_heterognn(dk_process_model, dk_init_marking, dk_final_marking, cfg)

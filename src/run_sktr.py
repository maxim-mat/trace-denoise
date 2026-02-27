import os

from sklearn.model_selection import train_test_split

from dataset.dataset import SaladsDataset
from utils.initialization import initialize
from utils.pm_utils import train_sktr, evaluate_sktr_on_dataset

import pickle as pkl


def main():
    args, cfg, dataset, logger = initialize()
    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)
    sktr_model = train_sktr(train_dataset, cfg)
    sktr_recovered = evaluate_sktr_on_dataset(test_dataset, sktr_model, cfg)
    with open(os.path.join(cfg.summary_path, "recovered_traces.pkl"), "wb") as f:
        pkl.dump(sktr_recovered, f)
    with open(os.path.join(cfg.summary_path, "test_dataset.pkl"), "wb") as f:
        pkl.dump(test_dataset, f)


if __name__ == "__main__":
    main()

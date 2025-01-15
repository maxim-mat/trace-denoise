import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from modules.CrossAttention import CrossAttention
from dataset.dataset import SaladsDataset
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset, remove_duplicates_trace, conformance_measure
from utils.initialization import initialize

if __name__ == "__main__":
    model = ConditionalUnetMatrixDenoiser(20, 20, 6000, 608).to("cuda")
    args, cfg, dataset, logger = initialize()
    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)
    dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                              preprocess=remove_duplicates_dataset)
    x = torch.randn(5, 20, 6000).to("cuda")
    y = torch.randn(5, 20, 6000).to("cuda")
    t = torch.randint(low=1, high=200, size=(5,)).to("cuda")
    M = torch.randn(21, 608, 608).unsqueeze(0).to("cuda")
    x_hat, m_hat, loss, l1, l2 = model(x, t, x, M, y)
    x_argmax = torch.argmax(x_hat, dim=1).to('cpu')
    x_processed = [remove_duplicates_trace(xi).tolist() for xi in x_argmax]
    conformance_measure(x_processed, dk_process_model, dk_init_marking, dk_final_marking, cfg.activity_names)
    print("done")

import json
from utils.Config import Config
import pickle as pkl
from dataset.dataset import SaladsDataset
from sklearn.model_selection import train_test_split
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset, pad_to_multiple_of_n
from utils.graph_utils import prepare_process_model_for_gnn, get_process_model_reachability_graph_transition_matrix
from modules.GraphEncoder import GraphEncoder
from torch.utils.data import DataLoader
from ddpm.ddpm_multinomial import Diffusion
from denoisers.ConditionalUnetGraphDenoiser import ConditionalUnetGraphDenoiser


if __name__ == "__main__":

    with open("src/config.json", "r") as f:
        cfg_json = json.load(f)
        cfg = Config(**cfg_json)

    with open("data/pickles/50_salads_unified.pkl", "rb") as f:
        dataset = pkl.load(f)

    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])


    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=0.7, shuffle=True, random_state=42)


    dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                              preprocess=remove_duplicates_dataset)
    pm_nx_data = prepare_process_model_for_gnn(dk_process_model, dk_init_marking, dk_final_marking, cfg)

    g_enc = GraphEncoder(pm_nx_data.num_nodes, 128, 128, 128, pooling=None).to(cfg.device).float()
    g_enc_add = GraphEncoder(pm_nx_data.num_nodes, 128, 128, 128).to(cfg.device).float()
    pm_nx_data = pm_nx_data.to(cfg.device)
    out_g = g_enc(pm_nx_data)
    out_g_add = g_enc_add(pm_nx_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8
    )

    dummy_x, dummy_y = (w.permute(0, 2, 1).to(cfg.device).float() for w in next(iter(train_loader)))

    diffuser = Diffusion(noise_steps=cfg.num_timesteps, device=cfg.device)

    graph_denoiser = ConditionalUnetGraphDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                  max_input_dim=salads_dataset.sequence_length, graph_data=pm_nx_data,
                                                  num_nodes=pm_nx_data.num_nodes, embedding_dim=128, hidden_dim=128,
                                                  pooling=None).to(cfg.device)
    t = diffuser.sample_timesteps(dummy_x.shape[0]).to(cfg.device)
    temp = graph_denoiser(dummy_x, t, None, None, y=dummy_x, drop_graph=True)[0]

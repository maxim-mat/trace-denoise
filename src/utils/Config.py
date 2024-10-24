from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = None
    summary_path: str = None
    device: str = None
    parallelize: bool = None
    num_epochs: int = None
    learning_rate: float = None
    num_timesteps: int = None
    num_workers: int = None
    test_every: int = None
    denoiser_hidden: int = None
    denoiser_layers: int = None
    num_classes: int = None
    batch_size: int = None
    denoiser: str = None
    eval_train: bool = None
    predict_on: str = None
    conditional_dropout: float = None
    mode: str = None
    train_percent: float = None
    seed: int = None
    max_seq_len: int = None
    enable_gnn: bool = None
    n2v_dimensions: int = None
    n2v_walk_length: int = None
    n2v_num_walks: int = None
    n2v_p: int = None
    n2v_q: int = None
    n2v_window: int = None
    n2v_min_count: int = None
    n2v_batch_words: int = None
    enable_mutual_information: bool = None
    enable_process_model_covariance: bool = None
    activity_names: dict[int, str] = None
    process_discovery_method: str = None
    round_precision: int = 2

    def __post_init__(self):
        if self.mode == "uncond":
            self.conditional_dropout = 1.0
        if self.activity_names is not None:
            self.activity_names = {int(k): v for k, v in self.activity_names.items()}

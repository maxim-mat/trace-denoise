from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = None
    summary_path: str = None
    device: str = None
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

    def __post_init__(self):
        if self.mode == "uncond":
            self.conditional_dropout = 1.0

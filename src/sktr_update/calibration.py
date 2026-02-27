"""
Temperature‑scaling and calibration for softmax probabilities.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sktr_update.config import logger
from sktr_update.constants import CASE_COLUMN, LABEL_COLUMN
from sktr_update.utils import inverse_softmax


class TemperatureScaling(nn.Module):
    """
    Module to learn a single temperature scalar.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def calibrate_probabilities_global_numpy(
    softmax_list: List[np.ndarray],
    df: pd.DataFrame,
    case_col: str,
    label_col: str,
    temp_bounds: Tuple[float, float],
    only_return_temperature: bool = False,
    global_temperature: Optional[float] = None
) -> Any:
    """
    Perform global temperature scaling across all cases.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_logits, all_labels = [], []
    num_classes = softmax_list[0].shape[0]
    unique_cases = df[case_col].unique()

    for case in unique_cases:
        probs = softmax_list[list(unique_cases).index(case)]
        labels = df[df[case_col] == case][label_col].astype(int).values
        logits = inverse_softmax(probs)
        L = min(logits.shape[1], len(labels))
        logits, labels = logits[:, :L], labels[:L]
        assert logits.shape[0] == num_classes
        all_logits.append(logits)
        all_labels.extend(labels)

    combined_logits = np.hstack(all_logits)
    combined_labels = np.array(all_labels)
    combined_logits = torch.tensor(combined_logits.T, dtype=torch.float32, device=device)
    combined_labels = torch.tensor(combined_labels, dtype=torch.long, device=device)

    if global_temperature is None:
        model = TemperatureScaling().to(device)
        optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=100)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(model(combined_logits), combined_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            model.temperature.clamp_(*temp_bounds)
        global_temperature = model.temperature.item()
        logger.info(f"Optimal temperature: {global_temperature}")

    if only_return_temperature:
        return global_temperature

    calibrated = []
    for logits in all_logits:
        lt = torch.tensor(logits, dtype=torch.float32, device=device)
        with torch.no_grad():
            scaled = lt / global_temperature
            calibrated.append(torch.softmax(scaled, dim=0).cpu().numpy())
    logger.info("Calibration complete.")
    return calibrated


def calibrate_softmax(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    softmax_train: List[np.ndarray],
    softmax_test: List[np.ndarray],
    temp_bounds: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Wrapper to first find the best temperature, then apply it to test softmax.
    """
    temp = calibrate_probabilities_global_numpy(
        softmax_list=softmax_train,
        df=train_df,
        case_col=CASE_COLUMN,
        label_col=LABEL_COLUMN,
        temp_bounds=temp_bounds,
        only_return_temperature=True
    )
    return calibrate_probabilities_global_numpy(
        softmax_list=softmax_test,
        df=test_df,
        case_col=CASE_COLUMN,
        label_col=LABEL_COLUMN,
        temp_bounds=temp_bounds,
        global_temperature=temp
    )

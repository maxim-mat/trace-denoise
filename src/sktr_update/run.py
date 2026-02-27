"""
Run several iterations of the SKTR vs Argmax experiment and average the results.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sktr_update.core import compare_stochastic_vs_argmax_random_indices
from sktr_update.utils import compute_average_accuracies


def run_multiple_iterations(
    df: pd.DataFrame,
    softmax_lst: List[Any],
    num_iterations: int,
    seeds: List[int],
    n_train_traces: int = 10,
    n_test_traces: int = 10,
    n_indices: int = 10,
    lambdas: List[float] = [0.1, 0.3, 0.6],
    alpha: float = 0.6,
    temp_bounds: Tuple[int, int] = (2, 2),
    cost_function: Union[str, Callable[[float], float]] = 'linear',
    train_cases: Optional[List[Any]] = None,
    test_cases: Optional[List[Any]] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    For each seed, run compare_stochastic_vs_argmax_random_indices,
    then compute the mean SKTR and Argmax accuracies over all runs.
    """
    sktr_accs, argmax_accs = [], []
    for i in range(num_iterations):
        seed = seeds[i]
        rec_df, _, _ = compare_stochastic_vs_argmax_random_indices(
            df=df,
            softmax_lst=softmax_lst,
            n_indices=n_indices,
            activity_prob_threshold=0.01,
            cost_function=cost_function,
            random_seed=seed,
            test_cases=test_cases,
            train_cases=train_cases,
            n_test_traces=n_test_traces,
            n_train_traces=n_train_traces,
            allow_intersection=True,
            include_duplicate_traces=True,
            sequential_sampling=True,
            round_precision=4,
            return_model=True,
            lambdas=lambdas,
            alpha=alpha,
            use_cond_probs=True,
            use_calibration=True,
            temp_bounds=temp_bounds
        )
        summary_rows = rec_df[rec_df['step'] == 'Separator']
        sk, ag = compute_average_accuracies(summary_rows, 'trace_summary')
        if sk is not None: sktr_accs.append(sk)
        if ag is not None: argmax_accs.append(ag)

    avg_sktr = np.mean(sktr_accs) if sktr_accs else None
    avg_argmax = np.mean(argmax_accs) if argmax_accs else None
    return avg_sktr, avg_argmax

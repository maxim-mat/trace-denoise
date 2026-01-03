"""
Data‑processing utilities for filtering, splitting, and converting softmax logs.
"""

from typing import List, Dict, Optional, Tuple, Any, Sequence, Union

import numpy as np
import pandas as pd
import random
import torch

from sktr_update.config import logger
from sktr_update.constants import CASE_COLUMN

# These are assumed to come from your existing RunningHorizon module:
from sktr_update.utils import (
    convert_tensors_to_numpy,
    train_test_log_split
)

from sktr_update.classes import Place, Transition, PetriNet, Marking


def prepare_softmax(
    softmax_list: Optional[List]
) -> Optional[np.ndarray]:
    """
    Convert a list of softmax arrays (or tensors) into a list of NumPy arrays.
    """
    logger.debug("Preparing softmax list.")
    if softmax_list is None:
        return None
    try:
        return convert_tensors_to_numpy(softmax_list)
    except Exception as e:
        logger.error(f"Failed to convert softmax list: {e}")
        raise


def filter_indices(
    df: pd.DataFrame,
    softmax_list: Optional[List[np.ndarray]],
    n_indices: int,
    sequential_sampling: bool,
    random_seed: int
) -> Tuple[pd.DataFrame, Optional[List[np.ndarray]]]:
    """
    Filter each trace in `df` down to `n_indices` event positions,
    and—if provided—filter the corresponding softmax matrices in lockstep.

    Parameters
    ----------
    df : pd.DataFrame
        Log dataframe with a 'case:concept:name' column.
    softmax_list : Optional[List[np.ndarray]]
        If given, a list of NumPy arrays (shape (n_classes, n_events))
        aligned one‑to‑one with each unique case in `df`.
    n_indices : int
        How many event positions to sample per trace.
    sequential_sampling : bool
        If True, sample within each run of identical activities;
        otherwise uniformly at random.
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[List[np.ndarray]]]
        - Filtered DataFrame containing only the sampled rows.
        - Parallel list of filtered softmax arrays, or None if no softmax_list.
    """
    logger.debug(
        "filter_indices(n_indices=%d, sequential=%s, seed=%d)",
        n_indices, sequential_sampling, random_seed
    )

    if softmax_list is not None:
        # 1. Collect unique case IDs in appearance order
        case_ids = df['case:concept:name'].drop_duplicates().tolist()
        if len(case_ids) != len(softmax_list):
            raise ValueError(
                f"Number of cases ({len(case_ids)}) does not match "
                f"number of softmax matrices ({len(softmax_list)})"
            )

        # 2. Pre-validate that each softmax matrix width = number of events in that trace
        #    Build a map: case -> event count
        counts = df['case:concept:name'].value_counts().to_dict()
        for case, matrix in zip(case_ids, softmax_list):
            expected = counts.get(case, 0)
            actual = matrix.shape[1]
            if actual != expected:
                raise ValueError(
                    f"Trace {case!r}: DataFrame has {expected} events "
                    f"but softmax matrix has {actual} columns"
                )

        # 3. Build mapping and delegate to helper that also does the same check again
        softmax_by_case: Dict[str, np.ndarray] = dict(zip(case_ids, softmax_list))

        df_filt, sm_filt = select_with_softmax_dict(
            df, softmax_by_case, n_indices, sequential_sampling, random_seed
        )
        
        logger.info("Filtered log length: %d", len(df_filt))
        return df_filt, sm_filt

    # No softmax provided: sample DataFrame only
    df_filt = _sample_df_only(df, n_indices, sequential_sampling, random_seed)
    logger.info("Sampled log length: %d", len(df_filt))
    return df_filt, None


def select_with_softmax_dict(
    df: pd.DataFrame,
    softmax_by_case: Dict[str, np.ndarray],
    n_indices: int,
    sequential_sampling: bool,
    random_seed: int
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Helper to sample both DataFrame and softmax matrices in sync,
    using a dict mapping from case ID to its softmax array.
    """
    rng = random.Random(random_seed)
    filtered_dfs: List[pd.DataFrame] = []
    filtered_sfts: List[np.ndarray] = []

    for case, group in df.groupby('case:concept:name', sort=False):
        group = group.reset_index(drop=True)

        if case not in softmax_by_case:
            raise KeyError(f"No softmax matrix found for case {case!r}")
        sftm = softmax_by_case[case]

        if sftm.shape[1] != len(group):
            raise ValueError(
                f"Trace {case!r}: softmax cols ({sftm.shape[1]}) != rows ({len(group)})"
            )

        picks = _sample_positions(group, n_indices, sequential_sampling, rng)
        filtered_dfs.append(group.iloc[picks].reset_index(drop=True))
        filtered_sfts.append(sftm[:, picks])

    out_df = pd.concat(filtered_dfs, ignore_index=True)
    return out_df, filtered_sfts


    def _sample_df_only(
        df: pd.DataFrame,
        n_indices: int,
        sequential_sampling: bool,
        random_seed: int
    ) -> pd.DataFrame:
        """
        Helper to sample only the DataFrame (no softmax matrices).
        """
        rng = random.Random(random_seed)
        pieces: List[pd.DataFrame] = []
    
        for case, group in df.groupby('case:concept:name', sort=False):
            group = group.reset_index(drop=True)
            picks = _sample_positions(group, n_indices, sequential_sampling, rng)
            pieces.append(group.iloc[picks])
    
        return pd.concat(pieces, ignore_index=True)


def _sample_positions(
    trace: pd.DataFrame,
    n_indices: int,
    sequential_sampling: bool,
    rng: random.Random
) -> List[int]:
    """
    Return a sorted list of positions to sample in a single trace.

    If `sequential_sampling` is True, up to `n_indices` positions
    are sampled from each run of identical activities; otherwise
    uniformly sampled across the entire trace.
    """
    length = len(trace)
    if length == 0:
        return []

    if not sequential_sampling:
        k = min(n_indices, length)
        return sorted(rng.sample(range(length), k))

    # Sequential sampling: split by runs of the same activity
    names = trace['concept:name'].tolist()
    picks: List[int] = []
    start = 0

    for idx in range(1, length):
        if names[idx] != names[idx - 1]:
            segment = list(range(start, idx))
            k = min(n_indices, len(segment))
            picks.extend(rng.sample(segment, k))
            start = idx

    # Last segment
    segment = list(range(start, length))
    k = min(n_indices, len(segment))
    picks.extend(rng.sample(segment, k))

    return sorted(picks)


def split_train_test(
    df: pd.DataFrame,
    n_train_traces: int,
    n_test_traces: int,
    train_traces: Optional[List[Any]],
    test_traces: Optional[List[Any]],
    random_selection: bool,
    random_seed: int,
    allow_duplicate_variants: bool,
    allow_intersection: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the log into train/test DataFrames of traces.
    """
    logger.debug("Splitting data into train/test sets.")
    result = train_test_log_split(
        dk_df=df,
        n_train_traces=n_train_traces,
        n_test_traces=n_test_traces,
        train_traces=train_traces,
        test_traces=test_traces,
        random_selection=random_selection,
        random_seed=random_seed,
        allow_duplicate_variants=allow_duplicate_variants,
        allow_intersection=allow_intersection
    )
    train_df = result['train_df']
    test_df = result['test_df']
    logger.info(
        f"Train traces: {train_df[CASE_COLUMN].nunique()}, "
        f"Test traces: {test_df[CASE_COLUMN].nunique()}"
    )
    return train_df, test_df


def sfmx_mat_to_sk_trace(
    softmax_matrix: Union[np.ndarray, list],
    case_identifier: str,
    round_precision: int = 2,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Convert a softmax matrix into a Stochastically-Known Trace DataFrame.

    Parameters
    ----------
    softmax_matrix : array-like of shape (n_activities, sequence_length)
        Softmax probabilities for each activity at each step.
    case_identifier : any
        Identifier to assign to every row in the resulting trace.
    round_precision : int, default=2
        Decimal places to round probabilities to.
    threshold : float, default=0.0
        Minimum probability to include an activity (zeros are kept when threshold=0.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'case:concept:name': all values == case_identifier
        - 'concept:name': list of activity labels at each position
        - 'probs': list of rounded probabilities at each position

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a (n_activities, sequence_length) matrix,
        or if any column has no probabilities ≥ threshold after rounding.
    """
    # Convert to NumPy array
    arr = np.asarray(softmax_matrix)

    # Remove leading batch dimension if present
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array but got shape {arr.shape!r}")

    num_acts, seq_len = arr.shape
    labels = [str(i) for i in range(num_acts)]

    out_activities = []
    out_probs = []

    for t in range(seq_len):
        probs = arr[:, t]
        # Keep all activities with prob >= threshold
        mask = probs >= threshold
        selected = probs[mask]
        rounded = np.round(selected, round_precision)
        # Drop any that rounded to exactly zero
        nonzero = rounded > 0
        if not nonzero.any():
            raise ValueError(f"No probabilities ≥ {threshold} (after rounding) at position {t}")
        final_probs = rounded[nonzero].tolist()
        kept_indices = np.where(mask)[0][nonzero]
        final_acts = [labels[i] for i in kept_indices]

        out_probs.append(final_probs)
        out_activities.append(final_acts)

    return pd.DataFrame({
        'case:concept:name': [case_identifier] * seq_len,
        'concept:name': out_activities,
        'probs': out_probs
    })


def select_softmax_matrices(
    softmax_matrices: Sequence[np.ndarray],
    df: pd.DataFrame,
) -> List[np.ndarray]:
    """
    Select softmax matrices corresponding to the case IDs in a given DataFrame,
    assuming case IDs are string representations of integer indices.

    Assumes:
    - `softmax_matrices[i]` corresponds to case ID == i.
    - `df` contains a 'case:concept:name' column with case IDs as strings.

    :param softmax_matrices: Sequence of softmax matrices indexed by case ID.
    :param df: DataFrame containing traces to select matrices for.
    :return: List of softmax matrices corresponding to the case IDs.
    :raises ValueError: If required columns are missing or IDs are invalid.
    """
    case_column = "case:concept:name"
    if case_column not in df.columns:
        raise ValueError(f"DataFrame must contain a '{case_column}' column.")

    # Extract unique case IDs while preserving order, and convert to integers efficiently
    case_ids_series = df[case_column].drop_duplicates()
    try:
        case_ids = case_ids_series.astype(int).tolist()
    except ValueError as e:
        raise ValueError(f"All case IDs must be convertible to integers. Problematic ID: {e.args[0]}") from e

    # Validate that all indices are within bounds
    max_index = len(softmax_matrices) - 1
    invalid_ids = [case_id for case_id in case_ids if case_id < 0 or case_id > max_index]
    if invalid_ids:
        raise ValueError(f"Invalid case IDs found (out of bounds): {invalid_ids}")

    return [softmax_matrices[case_id] for case_id in case_ids]


def construct_stochastic_trace_model(stochastic_trace_df: pd.DataFrame, non_sync_penalty: int = 1):
    """
    Constructs a stochastic trace model based on the given stochastic trace DataFrame.

    Parameters:
    stochastic_trace_df (pd.DataFrame): DataFrame containing 'concept:name' and 'probs' columns.
    non_sync_penalty (int, optional): Penalty for non-synchronous moves. Default is 1.

    Returns:
    stochastic_trace_model: Constructed stochastic trace model.
    """
    # Select relevant columns and reset index
    processed_df = stochastic_trace_df[['concept:name', 'probs']].reset_index(drop=True)
    
    # Construct the trace model
    stochastic_trace_model = construct_trace_model(processed_df, non_sync_penalty)
    
    return stochastic_trace_model


def construct_trace_model(trace_df, non_sync_move_penalty=1, add_heuristic=False):   
    trace_df = trace_df.reset_index(drop=True)
    places = [Place(f'place_{i}') for i in range(len(trace_df) + 1)]
    transitions = []
    transition_to_idx_dict = {}
    curr_idx = 0
    
    if not isinstance(trace_df['concept:name'].iloc[0], list):
        trace_df['concept:name'] = trace_df['concept:name'].apply(lambda x: [x])

    
    if add_heuristic:
        for i in trace_df.index:
            for idx, activity in enumerate(trace_df.loc[i,'concept:name']):
                prob = trace_df.loc[i,'probs'][idx]
                weight = non_sync_move_penalty - np.log(prob) / 10**5
                new_transition = Transition(f'{activity}_{i+1}', activity, move_type='trace', prob=prob,
                                            weight=weight)
                transitions.append(new_transition)
                transition_to_idx_dict[f'{activity}_{i+1}'] = curr_idx
                curr_idx += 1 

    else:
        for i in trace_df.index:
            for idx, activity in enumerate(trace_df.loc[i,'concept:name']):
                new_transition = Transition(f'{activity}_ts={i}', move_type='trace', label=activity,
                                            prob=trace_df.loc[i,'probs'][idx], weight=non_sync_move_penalty)
                transitions.append(new_transition)
                transition_to_idx_dict[f'{activity}_{i+1}'] = curr_idx
                curr_idx += 1    
    
    trace_model_net = PetriNet('trace_model', places, transitions)    
    
    for i in trace_df.index:
        for activity in trace_df.loc[i, 'concept:name']: 
            trace_model_net.add_arc_from_to(places[i], transitions[transition_to_idx_dict[f'{activity}_{i+1}']])
            trace_model_net.add_arc_from_to(transitions[transition_to_idx_dict[f'{activity}_{i+1}']], places[i+1])
    
    init_mark = tuple([1] + [0] * len(trace_df))
    final_mark = tuple([0] * len(trace_df) + [1])
    
    trace_model_net.init_mark = Marking(init_mark)
    trace_model_net.final_mark = Marking(final_mark)
    
    return trace_model_net
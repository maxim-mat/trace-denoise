"""
General utility functions for the sktr package.
"""

import re
import pickle
import random

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    Dict
)

import numpy as np
import pandas as pd
import torch
import os
import io
from pathlib import Path

from sktr_update.config import logger


def validate_input_parameters(
    n_indices: int,
    round_precision: int,
    non_sync_penalty: float,
    alpha: float,
    temp_bounds: Tuple[int, int]
) -> None:
    """
    Validate common input parameters for stochastic comparison.
    """
    logger.debug("Validating input parameters.")
    if not isinstance(n_indices, int) or n_indices <= 0:
        logger.error("n_indices must be a positive integer.")
        raise ValueError("n_indices must be a positive integer")
    if not isinstance(round_precision, int) or round_precision < 0:
        logger.error("round_precision must be a non-negative integer.")
        raise ValueError("round_precision must be a non-negative integer")
    if not isinstance(non_sync_penalty, (int, float)) or non_sync_penalty < 0:
        logger.error("non_sync_penalty must be a non-negative number.")
        raise ValueError("non_sync_penalty must be a non-negative number")
    if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        logger.error("alpha must be a number between 0 and 1.")
        raise ValueError("alpha must be a number between 0 and 1")
    if (
        not isinstance(temp_bounds, tuple)
        or len(temp_bounds) != 2
        or temp_bounds[0] > temp_bounds[1]
    ):
        logger.error(
            "temp_bounds must be a tuple of two integers "
            "where the first is <= the second."
        )
        raise ValueError(
            "temp_bounds must be a tuple of two integers where the first "
            "is less than or equal to the second"
        )
    logger.debug("Input parameters validation passed.")


def process_cost_function(
    cf: Optional[Union[str, Callable[[float], float]]]
) -> Optional[Callable[[float], float]]:
    """
    Convert cost‑function identifier to a callable.
    """
    logger.debug(f"Processing cost function: {cf}")
    if isinstance(cf, str):
        if cf == 'logarithmic':
            logger.debug("Using logarithmic cost function.")
            return lambda x: -np.log(x) / 4.7
        if cf == 'linear':
            logger.debug("Using linear cost function.")
            return lambda x: 1 - x
        logger.error(f"Unknown cost_function string: {cf}")
        raise ValueError(f"Unknown cost_function string: {cf}")
    logger.debug("Using provided callable as cost function.")
    return cf


def set_random_seed(seed: int) -> None:
    """
    Set the NumPy random seed for reproducibility.
    """
    logger.debug(f"Setting random seed to {seed}.")
    np.random.seed(seed)


def calculate_accuracy(
    correct: int,
    total: int,
    round_precision: int
) -> float:
    """
    Compute accuracy = correct/total, rounded.
    """
    if total < 0:
        logger.error("Total count cannot be negative.")
        raise ValueError("Total count cannot be negative")
    acc = round(correct / total, round_precision) if total > 0 else 0.0
    logger.debug(f"Calculated accuracy: {acc} (correct={correct}, total={total})")
    return acc


def compute_average_accuracies(
    df: pd.DataFrame,
    trace_summary_column: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    From every 'trace_summary' like "TC: 3.2, SKTR Acc: 0.75, Argmax Acc: 0.60",
    compute the overall average SKTR and Argmax accuracies.
    """
    sktr_accs: List[float] = []
    argmax_accs: List[float] = []
    pattern = r"SKTR Acc:\s*(\d+\.\d+), Argmax Acc:\s*(\d+\.\d+)"
    for txt in df[trace_summary_column]:
        m = re.search(pattern, txt)
        if m:
            sktr_accs.append(float(m.group(1)))
            argmax_accs.append(float(m.group(2)))
    avg_sktr = sum(sktr_accs) / len(sktr_accs) if sktr_accs else None
    avg_argmax = sum(argmax_accs) / len(argmax_accs) if argmax_accs else None
    return avg_sktr, avg_argmax


def inverse_softmax(
    softmax_probs: np.ndarray,
    epsilon: float = 1e-9
) -> np.ndarray:
    """
    Convert softmax probabilities to logits (inverse of softmax).

    Args:
        softmax_probs (np.ndarray): Array of softmax probabilities,
            shape (n_classes, n_events) or (n_classes,).
        epsilon (float): Small value to avoid log(0) or log(1).

    Returns:
        np.ndarray: Logits, same shape as input.

    Notes:
        - Each probability should be in (0, 1) and columns should sum to 1.
        - This is log(p), not logit(p). Softmax logit recovery can only retrieve
          logits *up to a constant shift* (since softmax is invariant to shifts).
    """
    # To avoid log(0) and log(1)
    probs = np.clip(softmax_probs, epsilon, 1.0 - epsilon)
    return np.log(probs)


def pickle_to_log_df(pickle_path: str) -> pd.DataFrame:
    """
    Load a pickle of PyTorch tensors list into a pandas DataFrame log.
    """
    with open(pickle_path, 'rb') as f:
        tensor_list = pickle.load(f)
    if not isinstance(tensor_list, list) or not all(isinstance(t, torch.Tensor) for t in tensor_list):
        raise ValueError("Pickle must contain a list of torch.Tensor")
    rows = []
    for case_id, tensor in enumerate(tensor_list):
        for val in tensor.tolist():
            rows.append({"case:concept:name": case_id, "concept:name": val})
    return pd.DataFrame(rows)


def convert_tensors_to_numpy(
    softmax_list: List[Union[torch.Tensor, np.ndarray]]
) -> List[np.ndarray]:
    """
    Convert a list of PyTorch tensors or NumPy arrays into NumPy arrays,
    squeezing out the leading singleton dimension.

    Parameters
    ----------
    softmax_list : List[Union[torch.Tensor, np.ndarray]]
        A sequence where each element is either:
        - a PyTorch Tensor of shape (1, …), possibly on GPU, or
        - a NumPy array (any shape).

    Returns
    -------
    List[np.ndarray]
        A list of NumPy arrays. For tensor inputs, the returned array is
        `tensor.cpu().numpy().squeeze(0)`. For array inputs, `np.asarray`
        is used to ensure a NumPy array.
    """
    return [
        # GPU → CPU → NumPy → squeeze leading dim
        item.cpu().numpy().squeeze(0)  # type: ignore[attr-defined]
        if isinstance(item, torch.Tensor) else np.asarray(item)
        for item in softmax_list
    ]


def train_test_log_split(
    dk_df: pd.DataFrame,
    n_train_traces: Optional[int] = None,
    n_test_traces: Optional[int] = None,
    train_traces: Optional[List[str]] = None,
    test_traces: Optional[List[str]] = None,
    random_selection: bool = True,
    random_seed: int = 42,
    allow_duplicate_variants: bool = True,
    allow_intersection: bool = False,
    sk_df: Optional[pd.DataFrame] = None,
    sftmax_lst: Optional[List[np.ndarray]] = None
) -> Dict[str, Union[pd.DataFrame, List[np.ndarray]]]:
    """
    Split a log into train/test sets of whole traces, optionally handling a second DataFrame (sk_df)
    or aligning softmax matrices (sftmax_lst).

    Returns a dict containing:
    - 'train_df': DataFrame of training events
    - 'test_df': DataFrame of testing events (from dk_df or sk_df)
    - 'test_ground_truth' (only if sk_df is provided): events from dk_df for test cases
    - 'test_sftmax_lst' (only if sftmax_lst is provided): list of softmax matrices for test cases
    """
    result: Dict[str, Any] = {}

    # 1) Determine case ordering & deduplication
    cases_list = get_unique_cases(dk_df, allow_duplicate_variants)

    # 2) Resolve which cases go to train vs. test
    def resolve_cases(
        all_cases: List[str],
        n_train: Optional[int],
        n_test: Optional[int],
        train_list: Optional[List[str]],
        test_list: Optional[List[str]],
        allow_intersect: bool,
        rand: bool,
        seed: int
    ) -> Tuple[List[str], List[str]]:
        if train_list is not None:
            train = train_list
            pool = all_cases if allow_intersect else [c for c in all_cases if c not in train_list]
            test = test_list if test_list is not None else select_cases(pool, n_test, rand, seed)
        elif test_list is not None:
            test = test_list
            pool = all_cases if allow_intersect else [c for c in all_cases if c not in test_list]
            train = select_cases(pool, n_train, rand, seed)
        else:
            train, test = split_cases(all_cases, n_train, n_test, allow_intersect, rand, seed)
        return train, test

    train_cases, test_cases = resolve_cases(
        cases_list,
        n_train_traces,
        n_test_traces,
        train_traces,
        test_traces,
        allow_intersection,
        random_selection,
        random_seed
    )

    # 3) Early check for softmax alignment
    if sftmax_lst is not None:
        unique_ids = list(dk_df['case:concept:name'].unique())
        if len(unique_ids) != len(sftmax_lst):
            raise ValueError(
                f"dk_df has {len(unique_ids)} unique cases, but sftmax_lst has {len(sftmax_lst)} matrices"
            )

    # 4) Build train/test DataFrames
    df_pairs = [
        ('train_df', dk_df, train_cases),
        ('test_df', sk_df if sk_df is not None else dk_df, test_cases)
    ]
    for key, source_df, cases in df_pairs:
        result[key] = filter_dataframe(source_df, cases)

    # 5) If sk_df was used, add ground truth and return
    if sk_df is not None:
        result['test_ground_truth'] = filter_dataframe(dk_df, test_cases)
        return result

    # 6) If softmax matrices provided, pick out test ones and return
    if sftmax_lst is not None:
        unique_ids = list(dk_df['case:concept:name'].unique())
        idx_map = {case: i for i, case in enumerate(unique_ids)}
        test_idx = [idx_map[c] for c in test_cases]
        result['test_sftmax_lst'] = [sftmax_lst[i] for i in test_idx]
        return result

    # 7) Otherwise just return train/test DataFrames
    return result
    

def split_cases(
    cases_list: List[str], 
    n_train: Optional[int] = None, 
    n_test: Optional[int] = None, 
    allow_intersection: bool = True, 
    random_selection: bool = True, 
    random_seed: Optional[int] = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits cases into train and test sets with configurable randomness and overlap.

    Args:
        cases_list: List of cases to split.
        n_train: Number of training cases. If None, defaults to half the dataset.
        n_test: Number of test cases. If None, defaults to remaining cases.
        allow_intersection: If True, train and test sets can share cases.
        random_selection: If True, shuffles cases before splitting.
        random_seed: Seed for reproducibility (if `random_selection=True`).

    Returns:
        Tuple of (train_cases, test_cases).

    Raises:
        ValueError: If `n_train + n_test` exceeds available cases (when `allow_intersection=False`).
    """

    if random_selection:
        if random_seed is not None:
            random.seed(random_seed)
        cases_list = random.sample(cases_list, len(cases_list))  # Shuffle, creating a new list

    # Set defaults if not provided
    if n_train is None and n_test is None:
        n_train = len(cases_list) // 2
        n_test = len(cases_list) - n_train
    elif n_train is None:
        n_train = len(cases_list) - n_test
    elif n_test is None:
        n_test = len(cases_list) - n_train

    # Check for impossible splits (only when allow_intersection=False)
    if not allow_intersection and (n_train + n_test > len(cases_list)):
        raise ValueError(
            f"Cannot split {len(cases_list)} cases into {n_train} train + {n_test} test without intersection."
        )

    # Generate train and test sets
    if allow_intersection:
        train_cases = random.sample(cases_list, n_train)
        test_cases = random.sample(cases_list, n_test)
    else:
        train_cases = cases_list[:n_train]
        test_cases = cases_list[n_train : n_train + n_test]

    return train_cases, test_cases


def select_cases(
    cases_list: List[str],
    n_cases: Optional[int] = None,
    random_selection: bool = True,
    random_seed: int = 42
) -> List[str]:
    """
    Selects up to `n_cases` case IDs from `cases_list`, optionally at random,
    without mutating the input list.
    """
    pool = cases_list[:]  # work on a copy
    if random_selection:
        rng = random.Random(random_seed)
        rng.shuffle(pool)

    # If n_cases is None or exceeds length, return all
    if n_cases is None or n_cases > len(pool):
        return pool
    return pool[:n_cases]


def get_unique_cases(
    log: pd.DataFrame,
    allow_duplicate_variants : bool
) -> List[str]:
    """
    Returns case IDs in first-appearance order.
    If allow_duplicate_variants=False, returns only one case per unique activity sequence ('concept:name' values),
    keeping the first to appear (deduplicates by trace variant).
    """
    # Build one tuple of activities per case
    sequences = (
        log
        .groupby('case:concept:name')['concept:name']
        .apply(tuple)
    )  # Series: index=case ID → value=tuple(activity names)

    if allow_duplicate_variants:
        # Every occurrence of a case ID in the original order
        return list(sequences.index)
    else:
        seen = set()
        unique = []
        # sequences.items() preserves the order of first appearance
        for case_id, seq in sequences.items():
            if seq not in seen:
                seen.add(seq)
                unique.append(case_id)
        return unique
    

def filter_dataframe(df, cases):
    """
    Keep only rows for the specified cases, in the order provided.
    
    Parameters:
    - df: DataFrame with a 'case:concept:name' column
    - cases: List of case IDs to keep
    
    Returns:
    - Filtered DataFrame with only the specified cases
    """
    # Keep only rows matching our cases
    filtered = df[df['case:concept:name'].isin(cases)]
    
    # Make sure cases appear in our desired order
    filtered = filtered.set_index('case:concept:name')
    ordered = filtered.loc[cases].reset_index()
    
    return ordered


def build_conditioned_prob_dict(df_train, max_hist_len=2, precision=2):
    
    def get_histories_up_to_length_k(activities_seq_list, k):
        """
        Generate all possible histories up to length k from the activities sequence,
        including the first activity with an empty history represented as ().
        """
        histories = []
        
        # Include the first activity with an empty history represented as ()
        if activities_seq_list:
            first_activity = activities_seq_list[0]
            histories.append(((), first_activity))
        
        # Generate histories for subsequent activities
        for i in range(1, len(activities_seq_list)):
            for j in range(1, min(i, k) + 1):
                history = tuple(activities_seq_list[i-j:i])
                histories.append((history, activities_seq_list[i]))
        
        return histories

    def get_relative_freq_dict(counter, precision=2):
        """
        Convert absolute counts to relative frequencies.
        """
        frequencies_dict = dict(counter)
        rel_freq_dict = defaultdict(dict)
        
        for key, freq in frequencies_dict.items():
            history, activity = key
            
            # Calculate total frequency for the current history
            total_history_freq = sum(
                frequencies_dict[sub_key] for sub_key in frequencies_dict 
                if sub_key[0] == history
            )
            
            # Avoid division by zero
            if total_history_freq == 0:
                continue
            
            # Calculate probability
            probability = round(freq / total_history_freq, precision)
            if probability > 0:
                rel_freq_dict[history][activity] = probability
        
        # Remove any empty dictionaries that may have resulted from filtering
        rel_freq_dict = {k: v for k, v in rel_freq_dict.items() if v}
                
        return rel_freq_dict


def group_cases_by_trace(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'case:concept:name' and aggregate the 'concept:name' into a tuple
    grouped = df.groupby('case:concept:name')['concept:name'].apply(tuple).reset_index()
    
    # Group by the trace (sequence of activities) and aggregate the case IDs
    trace_groups = grouped.groupby('concept:name')['case:concept:name'].apply(list).reset_index()
    
    # Add a column for the length of each trace
    trace_groups['trace_length'] = trace_groups['concept:name'].apply(len)
    
    # Sort case_list numerically
    trace_groups['case:concept:name'] = trace_groups['case:concept:name'].apply(lambda x: sorted(x, key=int))
    
    # Rename columns for clarity
    trace_groups.columns = ['trace', 'case_list', 'trace_length']
    
    return trace_groups[['case_list', 'trace_length']]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

            
# def prepare_df(
#     dataset_name: str,
#     path: str = "",
#     read_mapping: bool = False,
#     return_mapping: bool = False,
# ) -> Union[
#     pd.DataFrame,
#     Tuple[pd.DataFrame, List[np.ndarray], Dict[str, str]]
# ]:
#     """
#     Load a video dataset's target labels and softmax predictions,
#     build a combined DataFrame, auto‑fix matrix orientation, and
#     optionally return an activity mapping.
    
#     Parameters
#     ----------
#     dataset_name
#         One of: '50salads', 'gtea', 'breakfast'.
#     path
#         Base directory containing:
#           - `<dataset>_softmax_lst.pickle`
#           - `<dataset>_target_lst.pickle`
#         Defaults to ../../Datasets/video (relative to the utils.py file).
#     read_mapping
#         If True, load `mapping_<dataset>.txt` from the same folder.
#     return_mapping
#         If True, return the mapping dict as the third element.
        
#     Returns
#     -------
#     df
#         DataFrame with columns:
#           - 'case:concept:name' (trace IDs '0','1',…)
#           - 'concept:name'      (string activity codes)
#     softmax_list
#         List of np.ndarray, each shaped (n_classes, n_events).
#     mapping_dict (optional)
#         If requested, dict mapping original labels → integer strings.
#     """
#     # 1) Resolve file paths
#     if not path:
#         here = os.path.dirname(__file__)
#         # Need to go up two levels from SKTR/sktr to get to Jupyter Projects
#         path = os.path.abspath(os.path.join(here, "..", "..", "Datasets", "video"))
    
#     sf_path = os.path.join(path, f"{dataset_name}_softmax_lst.pickle")
#     tg_path = os.path.join(path, f"{dataset_name}_target_lst.pickle")
    
#     # 2) Load pickles on CPU
#     with open(sf_path, "rb") as f:
#         raw_softmax = CPU_Unpickler(f).load()
#     with open(tg_path, "rb") as f:
#         target_list = CPU_Unpickler(f).load()
    
#     # 3) Convert & auto‑fix orientation
#     softmax_list: List[np.ndarray] = []
#     for idx, entry in enumerate(raw_softmax):
#         # a) to numpy & squeeze
#         if isinstance(entry, torch.Tensor):
#             arr = entry.cpu().numpy().squeeze(0)
#         else:
#             arr = np.asarray(entry)
        
#         # b) ensure arr.shape == (n_classes, n_events)
#         trace_len = len(target_list[idx])
#         n_classes, n_events = arr.shape
#         if n_events != trace_len and n_classes == trace_len:
#             # transpose if the event‐axis was flipped
#             arr = arr.T
#             n_classes, n_events = arr.shape
        
#         # c) now both dims must match
#         if n_events != trace_len:
#             raise ValueError(
#                 f"prepare_df mismatch for case '{idx}': "
#                 f"{trace_len} events vs {n_events} softmax columns"
#             )
#         softmax_list.append(arr)
    
#     # 4) Optionally load mapping
#     mapping_dict: Optional[Dict[str, str]] = None
#     if read_mapping:
#         mapping_dict = read_mapping_file(dataset_name, path)
    
#     # 5) Build the DataFrame
#     records = []
#     for idx, trace in enumerate(target_list):
#         trace_id   = str(idx)
#         activities = [str(int(x)) for x in trace.tolist()]
#         for act in activities:
#             records.append({
#                 "case:concept:name": trace_id,
#                 "concept:name":      act,
#             })
#     df = pd.DataFrame.from_records(records)
    
#     # 6) Ensure 'concept:name' are string‐integers
#     df, mapping_dict = map_to_string_numbers(
#         df,
#         mapping_dict=mapping_dict,
#         map_strings_to_integer_strings=False
#     )
    
#     if return_mapping:
#         return df, softmax_list, mapping_dict  # type: ignore
#     return df, softmax_list


def prepare_df(
    dataset_name: str,
    path: Optional[Union[str, Path]] = None,
    return_mapping: bool = False,
    read_mapping: bool = False,
) -> Union[
    Tuple[pd.DataFrame, List[np.ndarray]],
    Tuple[pd.DataFrame, List[np.ndarray], Dict[str, str]]
]:
    """
    Prepare a DataFrame from a specified video dataset.

    Loads target labels and softmax predictions, builds a combined DataFrame,
    auto-fixes matrix orientation, and optionally returns activity mapping.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'.
    path : str or Path, optional
        Base directory containing:
          - {dataset_name}_softmax_lst.pickle
          - {dataset_name}_target_lst.pickle
        If None, tries three common relative locations.
    return_mapping : bool, optional
        If True, returns the mapping dict as the third element.
    read_mapping : bool, optional
        If True, attempts to load mapping_{dataset_name}.txt.

    Returns
    -------
    (df, softmax_list) or (df, softmax_list, mapping_dict)
    """
    # 1) validate dataset
    valid = {'salads', 'gtea', 'breakfast'}
    if dataset_name not in valid:
        raise ValueError(f"dataset_name must be one of {valid}")

    # 2) resolve path
    dirs_to_try = []
    if path:
        dirs_to_try.append(Path(path))
    else:
        here = Path(__file__).resolve().parent
        dirs_to_try += [
            here / 'Datasets' / 'video',
            here.parent / 'Datasets' / 'video',
            here.parents[1] / 'Datasets' / 'video'
        ]

    for d in dirs_to_try:
        if d.is_dir():
            data_dir = d
            break
    else:
        raise FileNotFoundError(f"Could not find Datasets/video in {dirs_to_try}")

    sf_path = data_dir / f"{dataset_name}_softmax_lst.pickle"
    tg_path = data_dir / f"{dataset_name}_target_lst.pickle"
    if not sf_path.exists() or not tg_path.exists():
        raise FileNotFoundError(f"Missing files in {data_dir}")

    # 3) load pickles
    with open(sf_path, "rb") as f:
        raw_softmax = CPU_Unpickler(f).load()
    with open(tg_path, "rb") as f:
        target_list = CPU_Unpickler(f).load()

    # 4) fix orientation
    softmax_list: List[np.ndarray] = []
    for idx, entry in enumerate(raw_softmax):
        arr = entry.cpu().numpy() if isinstance(entry, torch.Tensor) else np.asarray(entry)
        arr = np.squeeze(arr)
        L = len(target_list[idx])
        if arr.ndim == 1:
            # reshape 1D → (n_classes, L)
            c = arr.size // L
            if arr.size % L:
                raise ValueError(f"Cannot reshape array of size {arr.size} to match length {L}")
            arr = arr.reshape(c, L)
        else:
            c, e = arr.shape
            if e != L and c == L:
                arr = arr.T
                c, e = arr.shape
            if e != L:
                raise ValueError(f"Case {idx}: expected {L} columns but got {e}")

        softmax_list.append(arr)

    # 5) build df
    recs = []
    for idx, trace in enumerate(target_list):
        cid = str(idx)
        recs += [{"case:concept:name": cid, "concept:name": str(int(x))} for x in trace.tolist()]
    df = pd.DataFrame.from_records(recs)

    # 6) mapping
    mapping_dict = None
    if read_mapping:
        mapping_dict = read_mapping_file(dataset_name, data_dir)

    df, mapping_dict = map_to_string_numbers(df, mapping_dict=mapping_dict)

    return (df, softmax_list, mapping_dict) if return_mapping else (df, softmax_list)


def prepare_df_from_dataset(
    target_list,
    softmax_list,
    path=None,
    return_mapping=False,
    read_mapping=False,
):
    # 5) build df
    recs = []
    for idx, trace in enumerate(target_list):
        cid = str(idx)
        recs += [{"case:concept:name": cid, "concept:name": str(int(x))} for x in trace.tolist()]
    df = pd.DataFrame.from_records(recs)

    # 6) mapping
    mapping_dict = None

    df, mapping_dict = map_to_string_numbers(df, mapping_dict=mapping_dict)

    return (df, softmax_list, mapping_dict) if return_mapping else (df, softmax_list)

    

def map_to_string_numbers(
    df: pd.DataFrame,
    mapping_dict: Optional[Dict[str, str]] = None,
    map_strings_to_integer_strings: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Ensure `df['concept:name']` contains stringified integers.

    - If `mapping_dict` is provided and `map_strings_to_integer_strings=False`,
      values are looked up in `mapping_dict`.
    - If `map_strings_to_integer_strings=True`, every unique label is assigned
      a new integer string, updating/returning `mapping_dict`.

    Returns
    -------
    df
        Modified DataFrame (copy-on-write semantics).
    mapping_dict
        The mapping used (newly generated or provided).
    """
    df = df.copy()  # avoid in‑place changes

    if map_strings_to_integer_strings:
        if mapping_dict is None:
            mapping_dict = {}
        next_id = max((int(v) for v in mapping_dict.values()), default=-1) + 1

        def _map_new(label: str) -> str:
            nonlocal next_id
            if label not in mapping_dict:
                mapping_dict[label] = str(next_id)
                next_id += 1
            return mapping_dict[label]

        df["concept:name"] = df["concept:name"].astype(str).map(_map_new)
    else:
        # default: either use provided mapping_dict or just cast to int→str
        if mapping_dict:
            df["concept:name"] = df["concept:name"].map(mapping_dict)
        else:
            df["concept:name"] = (
                df["concept:name"]
                .astype(int)
                .astype(str)
            )

    return df, mapping_dict or {}
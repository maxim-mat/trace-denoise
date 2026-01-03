"""
Core comparison: SKTR (stochastic) vs argmax alignments.
"""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
    Dict
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from sktr_update.config import logger
from sktr_update.constants import VALID_MOVE_TYPES, CASE_COLUMN, LABEL_COLUMN
from sktr_update.utils import (
    validate_input_parameters,
    process_cost_function,
    set_random_seed,
    calculate_accuracy
)
from sktr_update.data_processing import (
    prepare_softmax,
    filter_indices,
    split_train_test,
    sfmx_mat_to_sk_trace,
    select_softmax_matrices,
    construct_stochastic_trace_model,
)
from sktr_update.petri_model import (
    discover_petri_net,
    build_probability_dict
)
from sktr_update.calibration import calibrate_softmax
from sktr_update.classes import SyncProduct


def compare_stochastic_vs_argmax_random_indices(
        df: pd.DataFrame,
        softmax_lst: Optional[List[np.ndarray]] = None,
        cost_function: Optional[Union[str, Callable[[float], float]]] = None,
        n_train_traces: int = 10,
        n_test_traces: int = 10,
        train_cases: Optional[List[Any]] = None,
        test_cases: Optional[List[Any]] = None,
        n_indices: int = 100,
        round_precision: int = 2,
        random_trace_selection: bool = True,
        random_seed: int = 42,
        non_sync_penalty: float = 1.0,
        activity_prob_threshold: float = 0.0,
        allow_duplicate_variants: bool = False,
        sequential_sampling: bool = False,
        allow_train_test_case_overlap: bool = True,
        only_return_model: bool = False,
        return_model: bool = False,
        lambdas: Optional[List[float]] = None,
        alpha: float = 0.5,
        use_cond_probs: bool = False,
        use_calibration: bool = False,
        temp_bounds: Tuple[int, int] = (1, 10),
        use_ngram_smoothing: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, Any]]:
    """
    Main entrypoint: discovers a Petri net on a train split, then for each
    test trace builds a stochastic model, aligns both SKTR (stochastic) and
    argmax, and returns two DataFrames:
    - recovery_df: step-by-step chosen activities & predictions
    - alignment_df: the alignment transitions and costs
    
    If return_model=True, also returns the discovered model.
    """
    logger.info("Starting comparison.")
    # 0. Validate inputs
    validate_input_parameters(n_indices, round_precision, non_sync_penalty, alpha, temp_bounds)

    # 1. Process cost function & random seed
    cost_fn = process_cost_function(cost_function)
    set_random_seed(random_seed)

    # 2. Filter / sample indices
    logger.debug("=== Entering filter_indices block ===")
    logger.debug("Input softmax_lst: %d matrices", len(softmax_lst))

    softmax_np = prepare_softmax(softmax_lst)

    # inspect prepare_softmax output
    if isinstance(softmax_np, list):
        shapes = [m.shape for m in softmax_np]
        logger.debug("prepare_softmax returned list of %d arrays, shapes=%s",
                     len(softmax_np), shapes)
    else:
        logger.debug("prepare_softmax returned ndarray of shape=%s",
                     getattr(softmax_np, "shape", None))

    logger.debug("Input DataFrame before filtering: %d rows, columns=%s",
                 len(df), df.columns.tolist())

    log_filt, sm_filt = filter_indices(
        df, softmax_np,
        n_indices, sequential_sampling,
        random_seed
    )

    # inspect outputs
    logger.debug("filter_indices → log_filt: %d rows", len(log_filt))
    if isinstance(sm_filt, list):
        shapes_filt = [m.shape for m in sm_filt]
        logger.debug("filter_indices → sm_filt: list of %d arrays, shapes=%s",
                     len(sm_filt), shapes_filt)
    elif sm_filt is None:
        logger.debug("filter_indices → sm_filt is None")
    else:
        logger.debug("filter_indices → sm_filt ndarray of shape=%s",
                     getattr(sm_filt, "shape", None))
    logger.debug("=== Exiting filter_indices block ===")

    # 3. Train/test split
    train_df, test_df = split_train_test(
        log_filt,
        n_train_traces,
        n_test_traces,
        train_cases,
        test_cases,
        random_trace_selection,
        random_seed,
        allow_duplicate_variants,
        allow_train_test_case_overlap
    )

    # 4. Discover model & optional prob-dict
    model = discover_petri_net(train_df, non_sync_penalty)
    prob_dict = build_probability_dict(train_df, use_cond_probs, lambdas)
    if only_return_model:
        logger.info("Returning model only.")
        return model

    # 5. Calibration if any
    if sm_filt is not None:
        sm_test = select_softmax_matrices(
            softmax_matrices=sm_filt,
            df=test_df,
        )
    else:
        sm_test = None

    if use_calibration:
        if sm_filt is None:
            raise ValueError("No softmax provided for calibration.")
        sm_train = select_softmax_matrices(sm_filt, train_df)
        sm_test = calibrate_softmax(train_df, test_df, sm_train, sm_test, temp_bounds)

    # 6. Iterate test cases with their matching softmax matrix
    rec_records: List[dict] = []
    aln_records: List[dict] = []

    cases = test_df[CASE_COLUMN].drop_duplicates().tolist()
    if len(cases) != len(sm_test):
        raise ValueError(f"Found {len(cases)} traces but {len(sm_test)} matrices")

    logger.info(f"Processing {len(cases)} test cases.")
    for idx, (case, mat) in enumerate(zip(cases, sm_test), start=1):
        logger.debug(f"Case {idx}/{len(cases)}: {case}, matrix shape={mat.shape}")
        rec, aln = _process_test_case(
            trace_case=case,
            test_df=test_df,
            softmax_mat=mat,
            model=model,
            cost_function=cost_fn,
            lambdas=lambdas or [],
            alpha=alpha,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            use_ngram_smoothing=use_ngram_smoothing,
            non_sync_penalty=non_sync_penalty,
            round_precision=round_precision,
            activity_prob_threshold=activity_prob_threshold
        )
        rec_records.extend(rec)
        aln_records.extend(aln)

    rec_df = pd.DataFrame(rec_records)
    aln_df = pd.DataFrame(aln_records)

    if return_model:
        return rec_df, aln_df, model
    return rec_df, aln_df


def train_sktr(
    df_train,
    softmax_train,
    n_indices,
    sequential_sampling,
    random_seed,
    non_sync_penalty,
    use_cond_probs,
    lambdas,
):
    set_random_seed(random_seed)
    softmax_array = prepare_softmax(softmax_train)
    # df_filtered, softmax_fildered = filter_indices(
    #     df_train, softmax_array,
    #     n_indices, sequential_sampling,
    #     random_seed
    # )
    prev = df_train.groupby("case:concept:name")["concept:name"].shift()
    df_dedup = df_train[df_train["concept:name"] != prev].reset_index(drop=True)
    model = discover_petri_net(df_dedup, non_sync_penalty)
    prob_dict = build_probability_dict(df_dedup, use_cond_probs, lambdas)
    return model, prob_dict


def _worker(case, mat,
            test_df, model, cost_fn, lambdas, alpha,
            use_cond_probs, prob_dict, use_ngram_smoothing,
            non_sync_penalty, round_precision, activity_prob_threshold):
    return _process_test_case(
        trace_case=case,
        test_df=test_df,
        softmax_mat=mat,
        model=model,
        cost_function=cost_fn,
        lambdas=lambdas or (),
        alpha=alpha,
        use_cond_probs=use_cond_probs,
        prob_dict=prob_dict,
        use_ngram_smoothing=use_ngram_smoothing,
        non_sync_penalty=non_sync_penalty,
        round_precision=round_precision,
        activity_prob_threshold=activity_prob_threshold
    )


def test_sktr_parallel(
    df_test,
    softmax_test,
    model,
    cost_function,
    lambdas,
    alpha,
    use_cond_probs,
    prob_dict,
    use_ngram_smoothing,
    non_sync_penalty,
    round_precision,
    activity_prob_threshold,
    n_indices,
    sequential_sampling,
    random_seed,
    n_jobs,
):
    cost_fn = cost_function
    set_random_seed(random_seed)
    softmax_array = prepare_softmax(softmax_test)
    df_filtered, softmax_fildered = filter_indices(
        df_test, softmax_array,
        n_indices, sequential_sampling,
        random_seed
    )
    rec_records: List[dict] = []
    aln_records: List[dict] = []
    cases = df_filtered[CASE_COLUMN].drop_duplicates().tolist()
    # for idx, (case, mat) in enumerate(zip(cases, softmax_fildered)):
    #     rec, aln = _process_test_case(
    #         trace_case=case,
    #         test_df=df_filtered,
    #         softmax_mat=mat,
    #         model=model,
    #         cost_function=cost_fn,
    #         lambdas=lambdas or [],
    #         alpha=alpha,
    #         use_cond_probs=use_cond_probs,
    #         prob_dict=prob_dict,
    #         use_ngram_smoothing=use_ngram_smoothing,
    #         non_sync_penalty=non_sync_penalty,
    #         round_precision=round_precision,
    #         activity_prob_threshold=activity_prob_threshold
    #     )
    #     rec_records.extend(rec)
    #     aln_records.extend(aln)

    worker = partial(_worker,
                     test_df=df_filtered,
                     model=model,
                     cost_fn=cost_fn,
                     lambdas=lambdas,
                     alpha=alpha,
                     use_cond_probs=use_cond_probs,
                     prob_dict=prob_dict,
                     use_ngram_smoothing=use_ngram_smoothing,
                     non_sync_penalty=non_sync_penalty,
                     round_precision=round_precision,
                     activity_prob_threshold=activity_prob_threshold)

    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        futures = {exe.submit(worker, case, mat): case
                   for case, mat in zip(cases, softmax_fildered)}
        # as each finishes, collect results
        for fut in as_completed(futures):
            rec, aln = fut.result()
            rec_records.extend(rec)
            aln_records.extend(aln)

    rec_df = pd.DataFrame(rec_records)
    aln_df = pd.DataFrame(aln_records)
    return rec_df, aln_df, model


def test_sktr(
        df_test,
        softmax_test,
        model,
        cost_function,
        lambdas,
        alpha,
        use_cond_probs,
        prob_dict,
        use_ngram_smoothing,
        non_sync_penalty,
        round_precision,
        activity_prob_threshold,
        n_indices,
        sequential_sampling,
        random_seed,
        limit=None,
):
    stop_after = limit if limit is not None else np.inf
    cost_fn = cost_function
    set_random_seed(random_seed)
    softmax_array = prepare_softmax(softmax_test)
    df_filtered, softmax_fildered = filter_indices(
        df_test, softmax_array,
        n_indices, sequential_sampling,
        random_seed
    )
    rec_records: List[dict] = []
    aln_records: List[dict] = []
    cases = df_filtered[CASE_COLUMN].drop_duplicates().tolist()
    for idx, (case, mat) in enumerate(zip(cases, softmax_fildered)):
        if idx >= stop_after:
            break
        rec, aln = _process_test_case(
            trace_case=case,
            test_df=df_filtered,
            softmax_mat=mat,
            model=model,
            cost_function=cost_fn,
            lambdas=lambdas or [],
            alpha=alpha,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            use_ngram_smoothing=use_ngram_smoothing,
            non_sync_penalty=non_sync_penalty,
            round_precision=round_precision,
            activity_prob_threshold=activity_prob_threshold
        )
        rec_records.extend(rec)
        aln_records.extend(aln)
    rec_df = pd.DataFrame(rec_records)
    aln_df = pd.DataFrame(aln_records)
    return rec_df, aln_df, model

def _process_test_case(
        trace_case: str,
        test_df: pd.DataFrame,
        softmax_mat: Optional[np.ndarray],
        model: Any,
        cost_function,
        lambdas: List[float],
        alpha: float,
        use_cond_probs: bool,
        prob_dict: Dict[Tuple[Any, ...], Dict[Any, float]],
        use_ngram_smoothing: bool,
        non_sync_penalty: float,
        round_precision: int,
        activity_prob_threshold: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Align a single trace under SKTR and argmax, preserving original behavior.

    Returns
    -------
    recovery_records : List[dict]
        Detailed per‐step recovery and prediction info.
    alignment_records : List[dict]
        Full alignment moves, including non‐synchronous transitions.
    """
    cost_function = process_cost_function(cost_function)
    # Extract the true trace
    true_trace = (
        test_df[test_df[CASE_COLUMN] == trace_case]
        .reset_index(drop=True)
    )
    length = len(true_trace)

    # Validate softmax matrix dimensions
    if softmax_mat is not None and softmax_mat.shape[1] != length:
        raise ValueError(
            f"Length mismatch for trace {trace_case}: "
            f"expected {length} columns but got {softmax_mat.shape[1]}"
        )

    # Build a DataFrame of (activities, probabilities) for each step
    if softmax_mat is None:
        stoch_df = true_trace.copy()
        stoch_df['probs'] = [[1.0]] * length
    else:
        stoch_df = sfmx_mat_to_sk_trace(
            softmax_mat,
            trace_case,
            round_precision,
            activity_prob_threshold
        )

    # Construct the stochastic trace model and run the alignment
    stoch_model = construct_stochastic_trace_model(stoch_df, non_sync_penalty)
    sync_prod = SyncProduct(model, stoch_model, cost_function=cost_function)
    alignment, _ = sync_prod._dijkstra_no_rg_construct(
        lambdas=lambdas,
        alpha=alpha,
        use_cond_probs=use_cond_probs,
        prob_dict=prob_dict,
        use_ngram_smoothing=use_ngram_smoothing,
        trace_recovery=True
    )

    # Filter only valid move types
    valid_moves = [m for m in alignment if m.move_type in VALID_MOVE_TYPES]

    # Collect per‐step recovery records
    recovery_records: List[Dict[str, Any]] = []
    correct_sktr = correct_argmax = 0

    for step, (acts, probs) in enumerate(zip(stoch_df[LABEL_COLUMN], stoch_df['probs'])):
        if step >= len(valid_moves):
            logger.warning(f"Overshot moves at step {step}; stopping.")
            break

        move = valid_moves[step]

        # Ground truth and predictions
        gt = true_trace[LABEL_COLUMN].iloc[step]
        sktr_pred = move.label
        if sktr_pred == gt:
            correct_sktr += 1

        argmax_idx = int(np.argmax(probs))
        argmax_pred = acts[argmax_idx]
        if argmax_pred == gt:
            correct_argmax += 1

        recovery_records.append({
            CASE_COLUMN: trace_case,
            'step': step,
            'possible_activities': acts,
            'probabilities': probs,
            'sktr_pred': sktr_pred,
            'argmax_pred': argmax_pred,
            'ground_truth': gt,
            'chosen_activity_weight': getattr(move, 'weight', None),
            'chosen_activity_name': getattr(move, 'name', None),
            'trace_summary': None
        })

    # Append summary row
    sktr_acc = calculate_accuracy(correct_sktr, length, round_precision)
    argmax_acc = calculate_accuracy(correct_argmax, length, round_precision)
    total_cost = round(
        sum(getattr(m, 'weight', 0) for m in valid_moves),
        round_precision
    )
    recovery_records.append({
        CASE_COLUMN: trace_case,
        'step': 'Separator',
        'possible_activities': None,
        'probabilities': None,
        'sktr_pred': None,
        'argmax_pred': None,
        'ground_truth': None,
        'chosen_activity_weight': None,
        'chosen_activity_name': None,
        'trace_summary': f"TC: {total_cost}, SKTR Acc: {sktr_acc}, Argmax Acc: {argmax_acc}"
    })

    # Build alignment records
    alignment_records: List[Dict[str, Any]] = []
    idx_st = 0
    for move in alignment:
        if move.move_type in VALID_MOVE_TYPES:
            acts = stoch_df[LABEL_COLUMN].iloc[idx_st]
            probs = stoch_df['probs'].iloc[idx_st]
            gt = true_trace[LABEL_COLUMN].iloc[idx_st]  # Get ground truth
            idx_st += 1
        else:
            acts = probs = gt = None

        alignment_records.append({
            CASE_COLUMN: trace_case,
            'transition_name': getattr(move, 'name', str(move)),
            'weight': getattr(move, 'weight', None),
            'possible_activities': acts,
            'probabilities': probs,
            'ground_truth': gt  # Add ground truth
        })

    # End‐of‐alignment marker
    alignment_records.append({
        CASE_COLUMN: trace_case,
        'transition_name': f"END OF ALIGNMENT (TC: {total_cost})",
        'weight': None,
        'possible_activities': None,
        'probabilities': None,
        'ground_truth': None  # Add ground truth field for consistency
    })

    return recovery_records, alignment_records


def log_test_data_info(
        test_df: pd.DataFrame,
        sm_test: Sequence[np.ndarray],
        case_column: str = CASE_COLUMN,
) -> None:
    """
    Log trace lengths (in order of first appearance) and the shape of each softmax matrix.

    :param test_df: DataFrame of test events, with one column identifying the case.
    :param sm_test: Sequence of numpy arrays (e.g., softmax matrices) aligned with those cases.
    :param case_column: Name of the column in test_df that holds the case ID.
    """
    # 1. Trace lengths in appearance order
    unique_cases = test_df[case_column].drop_duplicates().tolist()
    counts = test_df.groupby(case_column).size()
    total_cases = len(unique_cases)
    for idx, case_id in enumerate(unique_cases, start=1):
        length = int(counts[case_id])
        logger.info(f"[{idx}/{total_cases}] Trace '{case_id}' has {length} events")

    # 2. Shapes of softmax matrices
    total_matrices = len(sm_test)
    for idx, mat in enumerate(sm_test, start=1):
        logger.info(f"[{idx}/{total_matrices}] sm_test[{idx - 1}] shape: {mat.shape}")


def display_dataframe_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Display statistics about train and test dataframes including unique cases and basic metrics.
    
    Args:
        train_df: The training dataframe containing process traces
        test_df: The testing dataframe containing process traces
    """
    # Get case ID column (assuming it's the standard 'case:concept:name')
    case_column = 'case:concept:name'
    if case_column not in train_df.columns:
        # Try to find a case column if standard name isn't used
        case_candidates = [col for col in train_df.columns if 'case' in col.lower()]
        if case_candidates:
            case_column = case_candidates[0]
        else:
            raise ValueError("Could not identify case column in dataframes")

    # Get unique cases
    train_cases = train_df[case_column].unique()
    test_cases = test_df[case_column].unique()

    # Basic statistics
    train_events = len(train_df)
    test_events = len(test_df)
    train_unique_cases = len(train_cases)
    test_unique_cases = len(test_cases)

    # Calculate average trace length
    train_avg_length = train_events / train_unique_cases if train_unique_cases > 0 else 0
    test_avg_length = test_events / test_unique_cases if test_unique_cases > 0 else 0

    # Check for intersection between train and test sets
    intersection = set(train_cases).intersection(set(test_cases))
    intersection_size = len(intersection)

    # Activity diversity
    activity_column = 'concept:name'
    train_activities = train_df[activity_column].nunique() if activity_column in train_df.columns else "N/A"
    test_activities = test_df[activity_column].nunique() if activity_column in test_df.columns else "N/A"

    # Display results
    print("\n=== Dataset Statistics ===")
    print(f"{'Metric':<25} {'Training Set':<15} {'Testing Set':<15}")
    print(f"{'-' * 25} {'-' * 15} {'-' * 15}")
    print(f"{'Unique cases':<25} {train_unique_cases:<15} {test_unique_cases:<15}")
    print(f"{'Total events':<25} {train_events:<15} {test_events:<15}")
    print(f"{'Avg. trace length':<25} {train_avg_length:<15.2f} {test_avg_length:<15.2f}")  # Fixed line
    print(f"{'Unique activities':<25} {train_activities:<15} {test_activities:<15}")
    print(f"{'Cases in both sets':<25} {intersection_size:<15}")

    # Get distribution of trace lengths
    print("\n=== Trace Length Distribution ===")

    # Function to calculate trace length stats
    def get_trace_length_stats(df, case_col):
        trace_lengths = df.groupby(case_col).size()
        percentiles = trace_lengths.quantile([0.25, 0.5, 0.75]).to_dict()
        return {
            'min': trace_lengths.min(),
            'max': trace_lengths.max(),
            'mean': trace_lengths.mean(),
            'median': percentiles[0.5],
            '25th': percentiles[0.25],
            '75th': percentiles[0.75]
        }

    train_length_stats = get_trace_length_stats(train_df, case_column)
    test_length_stats = get_trace_length_stats(test_df, case_column)

    print(f"{'Statistic':<25} {'Training Set':<15} {'Testing Set':<15}")
    print(f"{'-' * 25} {'-' * 15} {'-' * 15}")
    for stat in ['min', 'max', 'mean', 'median', '25th', '75th']:
        train_val = f"{train_length_stats[stat]:<15.2f}" if stat == 'mean' else f"{train_length_stats[stat]:<15}"
        test_val = f"{test_length_stats[stat]:<15.2f}" if stat == 'mean' else f"{test_length_stats[stat]:<15}"
        print(f"{stat.capitalize():<25} {train_val} {test_val}")

    # Optional: Show first few cases from each set
    print("\n=== Sample Cases ===")
    print("Training set (first 5):", train_cases[:5])
    print("Testing set (first 5):", test_cases[:5])

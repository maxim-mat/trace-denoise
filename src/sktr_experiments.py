import numpy as np
import pickle as pkl
from sklearn import metrics
from tqdm import tqdm

from sktr_update.utils import prepare_df_from_dataset
from sktr_update.core import compare_stochastic_vs_argmax_random_indices

#'50_salads_unified', 'gtea_unified', 'ava_unified', 'bpi12_improved',
dataset_names = ['breakfast_unified']

for ds_name in tqdm(dataset_names):
    ds_results = []
    data_path = f"../data/pickles/{ds_name}.pkl"
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    target, source = data['target'], data['stochastic']
    target_for_sktr = [np.argmax(t, axis=1) for t in target]
    source_for_sktr = [np.array(x).T for x in source]
    df, softmax_lst = prepare_df_from_dataset(target_for_sktr, source_for_sktr)
    recovery_results_df, alignment_results_df, model = compare_stochastic_vs_argmax_random_indices(
        df=df,
        softmax_lst=softmax_lst,
        n_indices=60,
        activity_prob_threshold=0.01,
        cost_function='logarithmic',
        random_seed=42,
        train_cases=None,
        test_cases=None,
        n_train_traces=5,
        n_test_traces=10,
        allow_train_test_case_overlap=False,
        allow_duplicate_variants=True,
        sequential_sampling=True,
        round_precision=4,
        return_model=True,
        lambdas=[0.1, 0.3, 0.6],
        alpha=0.6,
        use_cond_probs=True,
        use_calibration=True,
        use_ngram_smoothing=False,
        temp_bounds=(1, 5)
    )
    case_lists = recovery_results_df.groupby("case:concept:name")[["sktr_pred", "argmax_pred", "ground_truth"]].apply(lambda g: g.values.tolist()).tolist()
    sktr_result = [[int(x[0]) for x in y if x[0] is not None] for y in case_lists]
    sktr_gt = [[int(x[2]) for x in y if x[0] is not None] for y in case_lists]
    sktr_acc = np.mean([metrics.accuracy_score(t, s) for t, s in zip(sktr_gt, sktr_result)])
    sktr_pre = np.mean([metrics.precision_score(t, s, average='macro', zero_division=0) for t, s in zip(sktr_gt, sktr_result)])
    sktr_rec = np.mean([metrics.recall_score(t, s, average='macro', zero_division=0) for t, s in zip(sktr_gt, sktr_result)])
    with open(f"{ds_name}_sktr_base_results.pkl", "wb") as f:
        pkl.dump([sktr_gt, sktr_result, sktr_acc, sktr_pre, sktr_rec], f)

    with open(f"{ds_name}_noisy.pkl", "rb") as f:
        noisy_datasets = pkl.load(f)
    for noisy_ds in tqdm(noisy_datasets):
        df, softmax_lst = prepare_df_from_dataset(target_for_sktr, [np.array(x).T for x in noisy_ds])
        recovery_results_df, alignment_results_df, model = compare_stochastic_vs_argmax_random_indices(
            df=df,
            softmax_lst=softmax_lst,
            n_indices=60,
            activity_prob_threshold=0.01,
            cost_function='logarithmic',
            random_seed=42,
            train_cases=None,
            test_cases=None,
            n_train_traces=5,
            n_test_traces=10,
            allow_train_test_case_overlap=False,
            allow_duplicate_variants=True,
            sequential_sampling=True,
            round_precision=4,
            return_model=True,
            lambdas=[0.1, 0.3, 0.6],
            alpha=0.6,
            use_cond_probs=True,
            use_calibration=True,
            use_ngram_smoothing=False,
            temp_bounds=(1, 5)
        )
        case_lists = recovery_results_df.groupby("case:concept:name")[["sktr_pred", "argmax_pred", "ground_truth"]].apply(lambda g: g.values.tolist()).tolist()
        sktr_result = [[int(x[0]) for x in y if x[0] is not None] for y in case_lists]
        sktr_gt = [[int(x[2]) for x in y if x[0] is not None] for y in case_lists]
        sktr_acc = np.mean([metrics.accuracy_score(t, s) for t, s in zip(sktr_gt, sktr_result)])
        sktr_pre = np.mean([metrics.precision_score(t, s, average='macro', zero_division=0) for t, s in zip(sktr_gt, sktr_result)])
        sktr_rec = np.mean([metrics.recall_score(t, s, average='macro', zero_division=0) for t, s in zip(sktr_gt, sktr_result)])
        ds_results.append((sktr_gt, sktr_result, sktr_acc, sktr_pre, sktr_rec))
    with open(f"{ds_name}_sktr_noisy_results.pkl", "wb") as f:
        pkl.dump([sktr_gt, sktr_result, sktr_acc, sktr_pre, sktr_rec], f)

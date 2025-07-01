import pandas as pd
import numpy as np
from itertools import combinations
import time
import gc
import sys
import logging
import warnings
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mutual_info_score
from mostlyai.qa._accuracy import bin_data
from .utils import time_it, print_memory_consumption, calculate_accuracy

logger = logging.getLogger(__name__)

def _make_spec(df: pd.DataFrame, bins: int = 10) -> Dict[str, Tuple]:
    spec = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            edges = np.unique(np.quantile(s.dropna().astype("float64"), np.linspace(0, 1, bins + 1)))
            if len(edges) < 2:
                edges = np.linspace(s.min(), s.max(), 2)
            spec[col] = ("num", edges)
        else:
            top_categories = s.value_counts(dropna=False).index[:bins - 1]
            mapping = {val: i for i, val in enumerate(top_categories)}
            spec[col] = ("cat", mapping, bins - 1)
    return spec


def _bin_df(df: pd.DataFrame, spec: Dict[str, Tuple]) -> pd.DataFrame:
    binned_df = pd.DataFrame(index=df.index)
    for col, info in spec.items():
        kind = info[0]
        if kind == "num":
            edges = info[1]
            binned_df[col] = np.searchsorted(edges[1:-1], df[col].values, side="right")
        else:
            mapping, other_code = info[1], info[2]
            binned_df[col] = df[col].map(mapping).fillna(other_code)
    return binned_df.astype("int64")


def _top_triples(df: pd.DataFrame, cols: List[str], k: int = 50, bins: int = 10) -> List[Tuple[str, str, str]]:
    if k == 0 or len(cols) < 3:
        return []
    mi_pairs = []
    for c1, c2 in combinations(cols, 2):
        mi = mutual_info_score(df[c1], df[c2])
        mi_pairs.append(((c1, c2), mi))
    mi_pairs.sort(key=lambda t: -t[1])
    top_cols = set([c for pair, _ in mi_pairs[:min(len(mi_pairs), 4 * k)] for c in pair])
    if len(top_cols) < 3:
        return []
    triples = []
    for c1, c2, c3 in combinations(top_cols, 3):
        joint_numerical = df[c1].values * bins + df[c2].values
        mi = mutual_info_score(joint_numerical, df[c3].values)
        triples.append(((c1, c2, c3), mi))
    triples.sort(key=lambda t: -t[1])
    return [t[0] for t in triples[:k]]


@time_it
def choose_rows_by_refinement(
        train_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        *,
        target_size: int,
        bins: int = 10,
        iterations: int = 500,
        swap_size: int = 100,
        top_k_pairs: int = 40,
        top_k_triples: int = 5,
        swap_size_multiplier: int = 3,
        initial_mask: Optional[np.ndarray] = None,
        max_time: Optional[int] = None
) -> np.ndarray:
    """Selects a subset of rows from a pool to match a training set's distribution.

    This function implements a greedy, iterative refinement algorithm. It starts
    with an initial subset (random or provided) and repeatedly swaps rows with
    those in the larger pool to minimize the L1 distance between the univariate,
    bivariate, and trivariate distributions of the subset and the original
    training data. It can also be used in a "trimming" mode to greedily remove
    the worst-fitting rows by setting the swap_size_multiplier to 0.

    Args:
        train_df: The original data to match the statistics of.
        pool_df: The larger pool of synthetic data to select rows from.
        target_size: The desired number of rows in the final subset.
        bins: The number of bins to use for discretizing numeric features.
        iterations: The maximum number of swap iterations to perform.
        swap_size: The number of rows to swap in each iteration.
        top_k_pairs: The number of top bivariate features (by mutual information)
                     to consider for matching.
        top_k_triples: The number of top trivariate features to consider.
        swap_size_multiplier: Factor to determine the size of the candidate pool
                              for additions. A value of 0 enables trimming mode.
        initial_mask: A boolean mask indicating the starting subset of rows from
                      the pool. If None, a random subset is chosen.
        max_time: Maximum execution time in minutes.

    Returns:
        An array of indices corresponding to the selected rows from pool_df.
    """
    start_time = time.time()
    feat_cols = list(train_df.columns)

    spec = _make_spec(train_df[feat_cols], bins)
    tr_bin = _bin_df(train_df[feat_cols], spec)
    pl_bin = _bin_df(pool_df[feat_cols], spec)
    del spec
    gc.collect()

    all_biv_feats = list(combinations(feat_cols, 2))
    biv_mi_pairs = sorted([(c, mutual_info_score(tr_bin[c[0]], tr_bin[c[1]])) for c in all_biv_feats],
                          key=lambda t: -t[1])
    bivariate_features = [p[0] for p in biv_mi_pairs[:top_k_pairs]]
    trivariate_features = _top_triples(tr_bin, feat_cols, k=top_k_triples, bins=bins)
    del all_biv_feats, biv_mi_pairs
    gc.collect()

    cols = {
        'uni': feat_cols,
        'bi': [f"{c1}×{c2}" for c1, c2 in bivariate_features],
        'tri': [f"{c1}×{c2}×{c3}" for c1, c2, c3 in trivariate_features]
    }

    tr_bin_np, pl_bin_np = {c: tr_bin[c].values for c in feat_cols}, {c: pl_bin[c].values for c in feat_cols}
    del tr_bin, pl_bin  # Free up the large binned DataFrames
    gc.collect()

    for c1, c2 in bivariate_features:
        name = f"{c1}×{c2}";
        tr_bin_np[name] = (tr_bin_np[c1] * bins + tr_bin_np[c2]).astype(np.int32)
        pl_bin_np[name] = (pl_bin_np[c1] * bins + pl_bin_np[c2]).astype(np.int32)

    for c1, c2, c3 in trivariate_features:
        name = f"{c1}×{c2}×{c3}";
        tr_bin_np[name] = (tr_bin_np[c1] * bins * bins + tr_bin_np[c2] * bins + tr_bin_np[c3]).astype(np.int32)
        pl_bin_np[name] = (pl_bin_np[c1] * bins * bins + pl_bin_np[c2] * bins + pl_bin_np[c3]).astype(np.int32)
    del bivariate_features, trivariate_features
    gc.collect()

    all_cols_by_phase = {
        'uni': cols['uni'], 'bi': cols['bi'], 'tri': cols['tri']
    }
    targets, max_bins = {}, {}
    for phase, p_cols in all_cols_by_phase.items():
        if not p_cols: continue
        targets[phase], max_bins[phase] = [], []
        for c in p_cols:
            n_bins = bins if phase == 'uni' else (bins ** 2 if phase == 'bi' else bins ** 3)
            targets[phase].append(np.bincount(tr_bin_np[c], minlength=n_bins))
            max_bins[phase].append(n_bins)

    G = len(pool_df)
    if target_size > G:
        warnings.warn(f"target_size ({target_size}) is larger than the pool size ({G}). Returning all rows.")
        return np.arange(G)

    if initial_mask is not None and initial_mask.shape == (G,):
        logger.info("Starting with the provided initial set...")
        chosen_mask = initial_mask.copy()
        if chosen_mask.sum() != target_size:
            warnings.warn(
                f"Provided initial_mask has {chosen_mask.sum()} items, but target_size is {target_size}. Refinement will proceed with {chosen_mask.sum()} items.")
    else:
        logger.info("Starting with a random initial set...")
        chosen_mask = np.zeros(G, dtype=bool)
        initial_indices = np.random.choice(G, size=target_size, replace=False)
        chosen_mask[initial_indices] = True

    def calculate_normalized_l1(hists_dict, targets_dict):
        total_norm_error = 0.0
        num_phases_with_features = 0
        for phase in ['uni', 'bi', 'tri']:
            if not targets_dict.get(phase): continue
            num_phases_with_features += 1
            phase_error = sum(
                np.abs(hists_dict[phase][j] - targets_dict[phase][j]).sum() for j in range(len(targets_dict[phase])))
            max_possible_phase_error = 2 * len(train_df) * len(targets_dict[phase])
            if max_possible_phase_error > 0:
                total_norm_error += phase_error / max_possible_phase_error
        return total_norm_error / num_phases_with_features if num_phases_with_features > 0 else 0

    current_hists = {}
    for phase, p_cols in all_cols_by_phase.items():
        if not p_cols: continue
        current_hists[phase] = [np.bincount(pl_bin_np[c][chosen_mask], minlength=max_bins[phase][j]).astype(np.int32)
                                for j, c in enumerate(p_cols)]

    current_error = calculate_normalized_l1(current_hists, targets)
    logger.info(f"Initial solution error (normalized): {current_error:.6f}")

    current_swap_size, min_swap_size, initial_temp = swap_size, 1, 0.00001
    is_trimming = swap_size_multiplier == 0
    for i in range(iterations):
        if max_time is not None and ((time.time() - start_time) / 60) > max_time:
            logger.info(f"Refinement time limit of {max_time} minutes reached. Stopping early.")
            break

        temperature = initial_temp * (1 - (i / iterations)) ** 2
        idx_chosen, = np.where(chosen_mask)
        idx_pool, = np.where(~chosen_mask)
        if len(idx_pool) < current_swap_size: break
        removal_gains = np.zeros(len(idx_chosen), dtype=np.float64)
        for phase in ['uni', 'bi', 'tri']:
            if not targets.get(phase): continue
            for j, c_name in enumerate(all_cols_by_phase[phase]):
                max_l1_dist = targets[phase][j].sum() * 2
                if max_l1_dist == 0: continue
                resid = targets[phase][j] - current_hists[phase][j]
                vals_chosen = pl_bin_np[c_name][idx_chosen]
                removal_gains += (np.abs(resid[vals_chosen]) - np.abs(resid[vals_chosen] + 1)) / max_l1_dist

        worst_indices = idx_chosen[np.argsort(removal_gains)[-current_swap_size:]]
        hists_of_worst = {
            p: [np.bincount(pl_bin_np[c][worst_indices], minlength=max_bins[p][j]).astype(np.int32) for j, c in
                enumerate(all_cols_by_phase[p])]
            for p in ['uni', 'bi', 'tri'] if targets.get(p)
        }

        cand_indices = np.random.choice(idx_pool, size=min(len(idx_pool), current_swap_size * swap_size_multiplier),
                                        replace=False)
        addition_gains = np.zeros(len(cand_indices), dtype=np.float32)
        for phase in ['uni', 'bi', 'tri']:
            if not targets.get(phase): continue
            for j, c_name in enumerate(all_cols_by_phase[phase]):
                max_l1_dist = 2 * targets[phase][j].sum()
                if max_l1_dist == 0: continue
                resid_after_removal = targets[phase][j] - (current_hists[phase][j] - hists_of_worst[phase][j])
                vals_cand = pl_bin_np[c_name][cand_indices]
                addition_gains += (np.abs(resid_after_removal[vals_cand]) - np.abs(
                    resid_after_removal[vals_cand] - 1)) / max_l1_dist

        best_replacements = cand_indices[np.argsort(addition_gains)[-current_swap_size:]]
        hists_of_cand = {
            p: [np.bincount(pl_bin_np[c][best_replacements], minlength=max_bins[p][j]).astype(np.int32) for j, c in
                enumerate(all_cols_by_phase[p])] for p in
            ['uni', 'bi', 'tri'] if targets.get(p)}
        new_hists = {p: [current_hists[p][j] - hists_of_worst[p][j] + hists_of_cand[p][j] for j in range(len(C))] for
                     p, C in targets.items() if C}
        new_error = calculate_normalized_l1(new_hists, targets)

        accepted = False
        if new_error < current_error:
            accepted = True
            # logger.info(f"Iter {i + 1:3d}/{iterations}: ACCEPTED (IMPROVED) Swapped {current_swap_size}. Err: {current_error:.6f} -> {new_error:.6f}.")
        elif temperature > 1e-9 and np.exp((current_error - new_error) / temperature) > np.random.random():
            accepted = True
            # logger.info(f"Iter {i + 1:3d}/{iterations}: ACCEPTED (ANNEALING) Swapped {current_swap_size}. Err: {current_error:.6f} -> {new_error:.6f}.")

        if accepted:
            if is_trimming:
                num_current_groups = sum(chosen_mask)
                # logger.info(f"Number of elements left: {num_current_groups} ({num_current_groups - target_size} to remove)")

                # check if we need to stop the loop in the trimming phase
                if (num_current_groups - len(worst_indices)) <= target_size:
                    num_needed_groups = num_current_groups - target_size
                    logger.info(f"Trimming stopped, remove last {num_needed_groups} elements")
                    if num_needed_groups > 0:
                        needed_worst_indices = worst_indices[-num_needed_groups:]
                        chosen_mask[needed_worst_indices] = False
                    break

            chosen_mask[worst_indices], chosen_mask[best_replacements] = False, True
            current_hists, current_error = new_hists, new_error

            if is_trimming:
                current_swap_size = max(current_swap_size - 1, 5)  # should get smaller over time
            else:
                current_swap_size = min(swap_size * 2, current_swap_size + 1)
        else:
            current_swap_size = max(min_swap_size, current_swap_size - 5)
        if (i + 1) % 100 == 0:
            temp_subset_df = pool_df.iloc[np.where(chosen_mask)[0]]
            acc = calculate_accuracy(train_df, temp_subset_df)
            logger.info(f"Iter {i + 1:4d}/{iterations}: Swap Size: {current_swap_size:3d}, Norm. L1 Err: {current_error:.6f}, Accuracy vs Original: {acc.get('overall_accuracy', 0):.6f}")

    logger.info(f"Finished refinement in {time.time() - start_time:.2f} seconds.")
    return np.where(chosen_mask)[0]


@time_it
def ipf_pairs_full(
        train_bin: pd.DataFrame,
        pool_bin: pd.DataFrame,
        k_final: int,
        top_pairs: int = 150,
        max_iter: int = 6,
        tol: float = 1e-4,
) -> np.ndarray:
    """Selects a subset of data using Iterative Proportional Fitting (IPF).

    This function performs IPF on the full data pool to align the bivariate
    distributions of the most important feature pairs with those of the
    training data. It calculates fractional weights for each row in the pool
    and then uses controlled rounding (expectation-rounding) to select a final
    integer number of rows.

    Args:
        train_bin: Binned training data.
        pool_bin: Binned synthetic data pool.
        k_final: The target number of rows to select.
        top_pairs: The number of highest mutual information pairs to use as
                   targets for IPF.
        max_iter: The maximum number of IPF iterations.
        tol: The tolerance for convergence.

    Returns:
        An array of indices for the selected rows from the pool, potentially
        with duplicates.
    """
    C = len(pool_bin.columns)
    codes_tr = train_bin.apply(lambda s: s.cat.codes, axis=0).to_numpy()
    codes_pl = pool_bin.apply(lambda s: s.cat.codes, axis=0).to_numpy()
    N = len(pool_bin)

    # pick top-MI pairs
    mi_rank = [
        (mutual_info_score(codes_tr[:, i], codes_tr[:, j]), i, j)
        for i in range(C) for j in range(i + 1, C)
    ]
    mi_rank.sort(reverse=True)
    pairs = [(i, j) for _, i, j in mi_rank[:top_pairs]]

    # 2-D targets
    n_tr = len(train_bin)
    targets = {}
    for i, j in pairs:
        di = train_bin.iloc[:, i].cat.categories.size
        dj = train_bin.iloc[:, j].cat.categories.size
        tgt = np.zeros((di, dj), float)
        np.add.at(tgt, (codes_tr[:, i], codes_tr[:, j]), 1.0)
        tgt *= (k_final / n_tr) / tgt.sum()
        targets[(i, j)] = tgt

    # IPF
    w = np.full(N, k_final / N, float)  # start uniform, sum = k_final
    for _ in range(max_iter):
        delta = 0.0
        for (i, j), tgt in targets.items():
            di, dj = tgt.shape
            cur = np.zeros_like(tgt)
            np.add.at(cur, (codes_pl[:, i], codes_pl[:, j]), w)
            scale = np.divide(tgt, cur, out=np.ones_like(cur), where=cur > 0)
            w *= scale[codes_pl[:, i], codes_pl[:, j]]
            delta = max(delta, np.max(np.abs(cur - tgt)))
        if delta < tol:
            break
        # renormalise
        w *= k_final / w.sum()

    # Expectation-rounding
    copy_cnt = np.floor(w).astype(int)
    remainder = w - copy_cnt
    shortfall = k_final - copy_cnt.sum()  # how many still missing
    if shortfall > 0:
        extra_idx = np.random.choice(
            np.arange(N), size=shortfall, replace=False, p=remainder / remainder.sum()
        )
        copy_cnt[extra_idx] += 1

    # materialise the sample
    chosen = np.repeat(np.arange(N), copy_cnt)
    if len(chosen) > k_final:
        chosen = np.random.choice(chosen, k_final, replace=False)
    return chosen


@time_it
def select_rows_with_ipf_and_refinement(
        train_df: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        ipf_top_pairs: int,
        refinement_top_pairs: int,
        refinement_top_triples: int,
        refinement_iterations: int,
        trimming_data_multiplier: float,
        trimming_swapsize: int,
        max_trimming_time: int,
        max_refinement_time: int
) -> np.ndarray:
    """Orchestrates the post-processing pipeline for flat data.

    This function executes a multi-step process to select a high-quality
    subset from a large pool of synthetic data.
    1.  IPF Selection: Selects an initial, oversized subset using IPF to
        match bivariate distributions.
    2.  Trimming: Reduces the subset to the target size by greedily removing
        rows that contribute most to statistical error.
    3.  Refinement: Further improves the subset by iteratively swapping rows
        with the pool to minimize multi-variate statistical errors.

    Returns:
        An array of the final selected indices from the synthetic_data pool.
    """
    target_size = len(train_df)

    logger.info("--- Step 0: Binning data for IPF ---")
    tr_int_bin, bins = bin_data(train_df, bins=10)
    pl_int_bin, _ = bin_data(synthetic_data, bins=bins)
    gc.collect()

    logger.info(f"--- Step 1: Generating initial subset with IPF (top {ipf_top_pairs} pairs) ---")
    trimming_size = int(target_size * trimming_data_multiplier)
    indices_from_ipf = ipf_pairs_full(
        train_bin=tr_int_bin,
        pool_bin=pl_int_bin,
        k_final=trimming_size,
        top_pairs=ipf_top_pairs,
    )
    del tr_int_bin, pl_int_bin
    gc.collect()

    logger.info("--- Step 2: Preparing IPF result for refinement ---")
    initial_mask = np.zeros(len(synthetic_data), dtype=bool)
    unique_indices = np.unique(indices_from_ipf)
    logger.info(f"IPF returned {len(indices_from_ipf)} indices with {len(unique_indices)} unique ones.")

    initial_mask[unique_indices] = True
    if len(unique_indices) < target_size:
        logger.warning(f"IPF produced {len(unique_indices)} unique rows, but target is {target_size}. Adding random rows to meet target size.")
        num_needed = target_size - len(unique_indices)
        available_pool, = np.where(~initial_mask)
        if num_needed > len(available_pool):
            raise ValueError("Not enough unique rows in the entire synthetic data pool to meet target_size.")
        random_fill = np.random.choice(available_pool, size=num_needed, replace=False)
        initial_mask[random_fill] = True

    logger.info("--- Accuracy of IPF-selected subset (before refinement) ---")
    ipf_subset_df = synthetic_data[initial_mask]
    ipf_accuracy = calculate_accuracy(train_df, ipf_subset_df)
    logger.info(f"Overall accuracy: {ipf_accuracy['overall_accuracy']}")
    del ipf_subset_df, ipf_accuracy
    gc.collect()

    logger.info("--- Step 3: Trimming the subset ---")
    trimming_indices = choose_rows_by_refinement(
        train_df=train_df,
        pool_df=synthetic_data,
        target_size=target_size,
        iterations=100_000,  # trimming needs to finish
        initial_mask=initial_mask,
        swap_size_multiplier=0,  # leads to trimming
        swap_size=trimming_swapsize,
        top_k_pairs=refinement_top_pairs,
        top_k_triples=refinement_top_triples,
        max_time=max_trimming_time
    )

    logger.info(f"--- Accuracy of Trimming subset (subset size {len(trimming_indices)}) ---")
    if len(trimming_indices) > target_size:
        logger.info(f"Too many indices returned from trimming - sample to match target size: {target_size}")
        trimming_indices = np.random.choice(trimming_indices, target_size, replace=False)

    subset_df = synthetic_data.iloc[trimming_indices]
    acc = calculate_accuracy(train_df, subset_df)
    logger.info(f"Overall accuracy: {acc['overall_accuracy']}")

    unique_indices = np.unique(trimming_indices)
    initial_mask = np.zeros(len(synthetic_data), dtype=bool)
    initial_mask[unique_indices] = True
    logger.info(f"Trimming returned {len(trimming_indices)} indices with {len(unique_indices)} unique ones.")

    logger.info("--- Step 4: Refining the subset ---")
    print_memory_consumption()
    refinement_size = len(unique_indices)
    refinement_train = train_df
    if refinement_size > len(train_df):
        refinement_train = pd.concat(
            [refinement_train, refinement_train.sample(refinement_size - len(train_df), replace=True)])
    logger.info(f"Using train set with len {len(refinement_train)} rows to align to refinement target size {refinement_size}.")

    final_indices = choose_rows_by_refinement(
        train_df=refinement_train,
        pool_df=synthetic_data,
        target_size=refinement_size,
        iterations=refinement_iterations,
        initial_mask=initial_mask,
        top_k_pairs=refinement_top_pairs,
        top_k_triples=refinement_top_triples,
        max_time=max_refinement_time
    )
    gc.collect()

    if len(final_indices) > target_size:
        logger.info(f"Trimming final set from {len(final_indices)} to {target_size}")
        final_indices = np.random.choice(final_indices, size=target_size, replace=False)
    elif len(final_indices) < target_size:
        num_needed = target_size - len(final_indices)
        logger.info(f"Padding final set with {num_needed} random rows to reach {target_size}")

        current_mask = np.zeros(len(synthetic_data), dtype=bool)
        current_mask[final_indices] = True
        available_pool, = np.where(~current_mask)
        random_fill = np.random.choice(available_pool, size=num_needed, replace=False)
        final_indices = np.concatenate([final_indices, random_fill])

    return final_indices
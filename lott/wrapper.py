"""
Experiment wrapper for running LoTT on various datasets.

Handles data loading, optional embedding extraction, data splitting
(train/calibration/holdout), and repeated testing.
"""

import numpy as np
import torch
import time
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exp.dataloader import load_data, check_device
from lott.lott import LoTT, LoTTWithSelection


def run_lott(dataset, N_ref, N_query, rs, check, n_test=100, alpha=0.05,
             is_selection=True, model_arch=None, model=None, verbose=False):
    """
    Run LoTT two-sample test on the given dataset.

    Args:
        dataset: Dataset name ('blob', 'cifar10', 'higgs')
        N_ref: Number of reference samples (N in the paper)
        N_query: Number of query samples (M in the paper)
        rs: Random seed
        check: 1 = test power (P != Q), 0 = type I error (P = Q)
        n_test: Number of independent test repetitions
        alpha: Significance level
        is_selection: If True, use LoTTWithSelection; else use LoTT
        model_arch: Model architecture name for embedding extraction
        model: Pre-loaded model (optional, avoids reloading)
        verbose: Print selection details

    Returns:
        H: Array of rejection decisions (0 or 1) over n_test trials
    """
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model is not None:
        model.to(device)
        model.eval()

    H = np.zeros(n_test)
    test_time = 0

    for k in range(n_test):
        start_time = time.time()

        X, Y_test, _ = load_data(dataset, N_ref, N_query, rs * 1000 + k, check,
                                  need_labels=True, model_arch=model_arch or "Res18")
        X = X.to(device, dtype=torch.float32)
        Y_test = Y_test.to(device, dtype=torch.float32)

        if model is not None:
            with torch.no_grad():
                X = model(X)
                Y_test = model(Y_test)

        n = len(X)
        perm = torch.randperm(n, device=device)
        n_train, n_calib = int(n * 0.4), int(n * 0.1)
        X_train = X[perm[:n_train]]
        X_calib = X[perm[n_train:n_train + n_calib]]
        X_hold = X[perm[n_train + n_calib:]]

        if is_selection:
            lott = LoTTWithSelection(
                alpha=alpha, n_permutations=500,
                selection_method='precision_weight', verbose=verbose
            )
        else:
            lott = LoTT(alpha=alpha, n_permutations=500)

        lott.fit(X_train, X_calib, X_hold)
        results = lott.test(Y_test)
        H[k] = int(results['reject'])

        test_time += time.time() - start_time

    torch.cuda.empty_cache()
    return H

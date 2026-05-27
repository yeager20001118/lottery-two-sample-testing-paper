"""
BLOB experiment: Varying reference size N with fixed query size M=50.
Reproduces Table 1 (left, BLOB) in the paper.
"""

import numpy as np
import argparse
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lott.wrapper import run_lott

parser = argparse.ArgumentParser(description="BLOB: Varying N (reference size)")
parser.add_argument('--check', default=1, type=int, help='1=test power, 0=type I error')
parser.add_argument('--N_ref', default=[100, 150, 200, 300, 400, 500, 1000], nargs='+', type=int,
                    help='Reference sample sizes N to evaluate')
parser.add_argument('--N_query', default=50, type=int, help='Query sample size M')
parser.add_argument('--n_exp', default=10, type=int, help='Number of independent experiments')
parser.add_argument('--n_test', default=100, type=int, help='Number of test repetitions per experiment')
parser.add_argument('--alpha', default=0.05, type=float, help='Significance level')
parser.add_argument('--seed', default=819, type=int, help='Base random seed')
args = parser.parse_args()

exp_path = os.path.dirname(os.path.abspath(__file__))

for N in args.N_ref:
    print(f"\n{'='*60}")
    print(f"BLOB | N={N}, M={args.N_query}, check={args.check}")
    print(f"{'='*60}")

    results = np.zeros(args.n_exp)
    for kk in range(args.n_exp):
        rs = kk * 1000 + args.seed + N
        H = run_lott('blob', N, args.N_query, rs, args.check,
                     n_test=args.n_test, alpha=args.alpha, is_selection=True)
        results[kk] = np.mean(H)

    mean_power = np.mean(results)
    std_power = np.std(results) / np.sqrt(args.n_exp)
    print(f"LoTT: {mean_power:.3f} +/- {std_power:.3f}")

    tag = "test_power" if args.check else "typeI_error"
    out_dir = os.path.join(exp_path, "Results", tag, str(args.alpha))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"blob_N{N}_M{args.N_query}_seed{args.seed}"
    np.savetxt(os.path.join(out_dir, fname), [mean_power, std_power], fmt='%.4f')

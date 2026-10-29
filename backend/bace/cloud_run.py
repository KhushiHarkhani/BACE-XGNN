"""GPU job: D-MPNN out-of-fold + test predictions for every outer split.

Output (one file) is merged by the local stacking step:
  <split>__tr, <split>__te      index arrays
  <split>__oof                  inner 5-fold OOF probs for the train indices
  <split>__test                 probs for test indices from a model fit on all of train

Run: python -m bace.cloud_run --out gnn_oof.npz --seeds 3 [--only random0,strict]
"""
import argparse
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from . import data, features
from .gnn import GNNModel, make_tables


def run_split(name, tr, te, seeds, inner, epochs, threads):
    torch.set_num_threads(threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = data.load(); y = df["y"].values
    graphs, extra, pic = make_tables(df, features.get(df))
    mk = lambda: GNNModel(graphs, extra, pic, seeds=tuple(range(seeds)), epochs=epochs, device=dev)
    oof = np.zeros(len(tr))
    for ia, ib in StratifiedKFold(inner, shuffle=True, random_state=0).split(tr, y[tr]):
        oof[ib] = mk().fit(tr[ia][:, None], y[tr[ia]]).predict_proba(tr[ib][:, None])[:, 1]
    test = mk().fit(tr[:, None], y[tr]).predict_proba(te[:, None])[:, 1]
    return name, {f"{name}__tr": tr, f"{name}__te": te, f"{name}__oof": oof, f"{name}__test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gnn_oof.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1, help="splits run in parallel (CPU)")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)

    df = data.load()
    splits = data.eval_splits(df)
    if a.only:
        splits = {k: v for k, v in splits.items() if k in a.only.split(",")}
    res = {}
    t0 = time.time()
    jobs = (delayed(run_split)(n, tr, te, a.seeds, a.inner, a.epochs, a.threads) for n, (tr, te) in splits.items())
    for name, d in Parallel(n_jobs=a.workers, return_as="generator_unordered")(jobs):
        res.update(d)
        np.savez(a.out, **res)  # saved as each split lands, so an interruption loses nothing
        print(f"{name} done ({time.time()-t0:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()

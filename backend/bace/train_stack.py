"""Fit the deployable stack on all 1,513 molecules and save it to artifacts/stack.joblib.
The meta-learner and threshold are fit on out-of-fold predictions only.
Needs artifacts/gnn.pt (python -m bace.train_final). Run: python -m bace.train_stack"""
import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from . import data, features
from .ensemble import base_models, stack_matrix
from .gnn import GNNModel, make_tables
from .train_final import OUT


def _gnn_oof(df, y, blocks, inner=3, epochs=40):
    torch.set_num_threads(4)
    g, e, p = make_tables(df, blocks)
    idx = np.arange(len(df)); oof = np.zeros(len(df))
    for ia, ib in StratifiedKFold(inner, shuffle=True, random_state=0).split(idx, y):
        m = GNNModel(g, e, p, seeds=(0,), epochs=epochs).fit(idx[ia][:, None], y[ia])
        oof[ib] = m.predict_proba(idx[ib][:, None])[:, 1]
    return oof


def _base(name, bn, mk, blocks, y):
    x = stack_matrix(blocks, bn)
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        oof[b] = mk().fit(x[a], y[a]).predict_proba(x[b])[:, 1]
    return name, oof, mk().fit(x, y)


def main():
    df = data.load(); y = df["y"].values
    blocks = features.get(df)
    models = base_models(0)
    res = Parallel(n_jobs=4)(delayed(_base)(n, bn, mk, blocks, y) for n, (bn, mk) in models.items())
    gnn_oof = _gnn_oof(df, y, blocks)
    oof = np.column_stack([r[1] for r in res] + [gnn_oof])
    meta = LogisticRegression(max_iter=1000).fit(oof, y)
    p = meta.predict_proba(oof)[:, 1]
    ths = np.linspace(0.3, 0.7, 41)
    thr = float(ths[np.argmax([accuracy_score(y, p > t) for t in ths])])
    OUT.mkdir(exist_ok=True)
    joblib.dump(dict(names=[r[0] for r in res], feats={n: bn for n, (bn, _) in models.items()},
                     models={r[0]: r[2] for r in res}, meta=meta, threshold=thr), OUT / "stack.joblib")
    print("saved stack.joblib; threshold", thr, "OOF acc", accuracy_score(y, p > thr))


if __name__ == "__main__":
    main()

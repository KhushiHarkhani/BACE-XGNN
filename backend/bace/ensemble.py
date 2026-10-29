"""Stacked ensemble of non-GNN models. The meta-learner and threshold only ever see
out-of-fold predictions from the training portion, never the held-out fold.
Run: python -m bace.ensemble"""
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from joblib import Parallel, delayed

from . import data, features


def minmax_kernel(a, b):
    """Generalised Tanimoto (MinMax) kernel for count fingerprints, vectorised via L1 distance."""
    l1 = cdist(a, b, "cityblock")
    tot = a.sum(1)[:, None] + b.sum(1)[None, :]
    return ((tot - l1) / np.maximum(tot + l1, 1e-9)).astype(np.float32)


class SVMKernel:
    def __init__(self, C=3.0):
        self.C = C

    def fit(self, x, y):
        self.x, self.m = x, SVC(C=self.C, kernel="precomputed", probability=True, random_state=0)
        self.m.fit(minmax_kernel(x, x), y)
        return self

    def predict_proba(self, x):
        return self.m.predict_proba(minmax_kernel(x, self.x))


def base_models(seed):
    """name -> (feature-block names, model factory)"""
    return {
        "cat_dense": (["rdkit", "maccs"], lambda: CatBoostClassifier(
            iterations=250, depth=6, learning_rate=0.08, l2_leaf_reg=6, random_seed=seed, verbose=0, thread_count=2)),
        "lgbm_all": (["ecfp4", "atompair", "rdkit", "maccs"], lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.04, n_jobs=2, num_leaves=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.3, reg_lambda=5, min_child_samples=10, random_state=seed, verbose=-1)),
        "et_fp": (["ecfp4", "fcfp4", "maccs", "avalon"], lambda: ExtraTreesClassifier(
            n_estimators=300, max_features=0.1, min_samples_leaf=1, n_jobs=2, random_state=seed)),
        "svm_ecfp4": (["ecfp4"], lambda: SVMKernel(3.0)),
        "svm_atompair": (["atompair"], lambda: SVMKernel(3.0)),
        "svm_ecfp6": (["ecfp6"], lambda: SVMKernel(3.0)),
    }


def stack_matrix(blocks, names):
    return np.hstack([blocks[n] for n in names])


def base_predictions(blocks, y, tr, te, seed=0, inner=5):
    """Base-model OOF matrix on tr (inner CV) and test matrix from fits on all of tr."""
    models = base_models(seed)
    oof = np.zeros((len(tr), len(models)))
    te_m = np.zeros((len(te), len(models)))
    skf = StratifiedKFold(inner, shuffle=True, random_state=seed)
    for j, (name, (bn, mk)) in enumerate(models.items()):
        x = stack_matrix(blocks, bn)
        for a, b in skf.split(tr, y[tr]):
            oof[b, j] = mk().fit(x[tr[a]], y[tr[a]]).predict_proba(x[tr[b]])[:, 1]
        te_m[:, j] = mk().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return oof, te_m, list(models)


def meta_fit_predict(oof, ytr, te_m):
    """Meta-learner + threshold use only out-of-fold predictions from train."""
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(oof, ytr)
    p_oof = meta.predict_proba(oof)[:, 1]
    ths = np.linspace(0.3, 0.7, 41)
    thr = float(ths[np.argmax([accuracy_score(ytr, p_oof > t) for t in ths])])
    return meta.predict_proba(te_m)[:, 1], thr


def report(tag, ys, ps, thrs):
    y, p = np.concatenate(ys), np.concatenate(ps)
    accs = [accuracy_score(a, b > t) for a, b, t in zip(ys, ps, thrs)]
    pred = np.concatenate([b > t for b, t in zip(ps, thrs)])
    print(f"{tag:22s} acc={np.mean(accs):.4f}+-{np.std(accs):.4f} "
          f"auc={roc_auc_score(y, p):.4f} f1={f1_score(y, pred):.4f} mcc={matthews_corrcoef(y, pred):.4f}", flush=True)


def _one(name, tr, te, blocks, y):
    oof, te_m, names = base_predictions(blocks, y, tr, te)
    return name, dict(tr=tr, te=te, oof=oof, te_m=te_m, names=np.array(names))


def main():
    df = data.load()
    y = df["y"].values
    blocks = features.get(df)
    splits = data.eval_splits(df)
    out = dict(Parallel(n_jobs=5)(delayed(_one)(n, tr, te, blocks, y) for n, (tr, te) in splits.items()))
    flat = {f"{n}__{k}": v for n, d in out.items() for k, v in d.items()}
    np.savez(data.DATA.parent / "base_preds.npz", **flat)

    def group(prefix):
        ks = [k for k in out if k.startswith(prefix)]
        ys, ps, ts = [], [], []
        for k in ks:
            p, t = meta_fit_predict(out[k]["oof"], y[out[k]["tr"]], out[k]["te_m"])
            ys.append(y[out[k]["te"]]); ps.append(p); ts.append(t)
        return ys, ps, ts
    report("RANDOM 5-fold", *group("random"))
    report("STRICT scaffold", *group("strict"))
    report("RANDOMISED scaffold x3", *group("scaffold"))


if __name__ == "__main__":
    main()

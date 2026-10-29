"""Fingerprint + descriptor baseline, random-CV and scaffold-split. Run: python -m bace.baseline"""
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from . import data


def fit_predict(xtr, ytr, xte, seed=0):
    m = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=5,
                           random_seed=seed, verbose=0, thread_count=-1)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1], m.predict_proba(xtr)[:, 1]


def main():
    df = data.load()
    mols = list(df["mol_obj"])
    y = df["y"].values
    desc, _ = data.rdkit_desc(mols)
    x = np.hstack([data.ecfp(mols), desc])
    print(f"n={len(df)} features={x.shape[1]} pos={y.mean():.3f}")

    accs, aucs, tr_accs = [], [], []
    for rep in range(2):
        for tr, te in data.random_folds(y, 5, seed=rep):
            p, ptr = fit_predict(x[tr], y[tr], x[te], seed=rep)
            accs.append(accuracy_score(y[te], p > .5)); aucs.append(roc_auc_score(y[te], p))
            tr_accs.append(accuracy_score(y[tr], ptr > .5))
    print(f"RANDOM 5x2 CV  acc={np.mean(accs):.4f}+-{np.std(accs):.4f} auc={np.mean(aucs):.4f} train_acc={np.mean(tr_accs):.4f}")

    accs, aucs = [], []
    for seed in range(5):
        tr, va, te = data.scaffold_split(df, seed)
        p, _ = fit_predict(x[tr], y[tr], x[te], seed=seed)
        accs.append(accuracy_score(y[te], p > .5)); aucs.append(roc_auc_score(y[te], p))
    print(f"SCAFFOLD x5    acc={np.mean(accs):.4f}+-{np.std(accs):.4f} auc={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()

"""Explanation quality over 30 random molecules (same size as the Phase 1 report's check).
Faithfulness: drop in predicted-class probability from removing the top-5 atoms minus random-5 atoms.
Stability: mean pairwise Jaccard of the top-5 atoms across independently trained seed networks.
Run: python -m bace.evaluate_explain"""
import json

import numpy as np

from . import data, explain, gnn
from .train_final import OUT


def main(n=30):
    df = data.load()
    m = gnn.load(OUT / "gnn_graph.pt")  # the explainer network, not the descriptor-fused predictor
    idx = np.random.RandomState(0).choice(len(df), n, replace=False)
    f, s = [], []
    for i in idx:
        mol = df["mol_obj"][i]
        imp = explain.atom_importance(m, mol)
        f.append(explain.faithfulness(m, mol, imp)); s.append(explain.stability(m, mol))
    out = dict(n=n, faithfulness_mean=float(np.mean(f)), faithfulness_positive_frac=float(np.mean(np.array(f) > 0)),
               stability_seed_jaccard_mean=float(np.nanmean(s)))
    print(out)
    (OUT / "explain_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

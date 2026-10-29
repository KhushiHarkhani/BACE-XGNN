"""Train the deployed GNN on all 1,513 molecules. Metrics come from cross-validation (ensemble.py),
not from this model: it has seen every molecule. Run: python -m bace.train_final --seeds 5"""
import argparse
from pathlib import Path

from . import data, features, gnn

OUT = Path(__file__).resolve().parents[2] / "artifacts"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--graph-only", action="store_true", help="no descriptor branch: the explainer network")
    a = ap.parse_args()
    df = data.load(); y = df["y"].values
    g, e, p = gnn.make_tables(df, features.get(df))
    import numpy as np
    idx = np.arange(len(df))
    m = gnn.GNNModel(g, e, p, seeds=tuple(range(a.seeds)), graph_only=a.graph_only).fit(idx[:, None], y)
    OUT.mkdir(exist_ok=True)
    path = OUT / ("gnn_graph.pt" if a.graph_only else "gnn.pt")
    gnn.save(m, path)
    print("saved", path)


if __name__ == "__main__":
    main()

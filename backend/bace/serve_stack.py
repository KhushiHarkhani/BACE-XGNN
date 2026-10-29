"""Load the deployable stack and score a single molecule from its SMILES alone."""
import joblib
import numpy as np

from . import explain, features
from .gnn import mol_to_graph
from .train_final import OUT


class Stack:
    def __init__(self, gnn_model):
        d = joblib.load(OUT / "stack.joblib")
        self.d, self.gnn = d, gnn_model
        self.threshold = d["threshold"]

    def predict(self, mol) -> float:
        blocks = features.build_from_mols([mol])
        cols = []
        for name in self.d["names"]:
            x = np.hstack([blocks[b] for b in self.d["feats"][name]])
            cols.append(self.d["models"][name].predict_proba(x)[:, 1][0])
        cols.append(float(explain._prob(self.gnn, [mol_to_graph(mol)], explain.extra_vector(mol))[0]))
        return float(self.d["meta"].predict_proba(np.array(cols)[None])[:, 1][0])

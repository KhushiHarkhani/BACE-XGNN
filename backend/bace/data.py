"""Data loading, featurisation and splits for BACE-1.

Leakage guard. Never used as features:
  pIC50     Class matches pIC50 >= 7 for 98.9% of molecules (17 class-1 exceptions)
  canvasUID row ID, and the file is sorted by class, so it encodes the label (AUC 0.9998 alone)
  Model, CID, mol, Class
pIC50 is allowed only as an auxiliary *training* target in gnn.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold

RDLogger.DisableLog("rdApp.*")

DATA = Path(__file__).resolve().parents[2] / "data" / "bace.csv"
LEAK_COLS = {"mol", "CID", "Class", "Model", "pIC50", "canvasUID"}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["mol_obj"] = df["mol"].map(Chem.MolFromSmiles)
    df = df[df["mol_obj"].notna()].copy()
    df["smiles"] = df["mol_obj"].map(Chem.MolToSmiles)
    df = df.drop_duplicates("smiles").reset_index(drop=True)
    df["y"] = df["Class"].astype(int)
    return df


def ecfp(mols, radius=2, n_bits=2048, counts=True) -> np.ndarray:
    out = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        fp = AllChem.GetHashedMorganFingerprint(m, radius, nBits=n_bits) if counts else None
        for k, v in fp.GetNonzeroElements().items():
            out[i, k] = v
    return out


def rdkit_desc(mols) -> np.ndarray:
    names = [n for n, _ in Descriptors.descList]
    x = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1e6, 1e6).astype(np.float32), names


def scaffold_of(m) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def scaffold_split(df, seed=0, frac=(0.8, 0.1, 0.1)):
    """Group molecules by Murcko scaffold; whole groups go to one partition.
    Groups are shuffled (seeded) rather than size-sorted so repeated runs differ."""
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(df["mol_obj"]):
        groups.setdefault(scaffold_of(m), []).append(i)
    keys = list(groups)
    rng = np.random.RandomState(seed)
    big = [k for k in keys if len(groups[k]) > len(df) * frac[1] / 2]
    small = [k for k in keys if k not in big]
    rng.shuffle(small)
    order = big + small
    n = len(df)
    tr, va, te = [], [], []
    for k in order:
        g = groups[k]
        if len(tr) + len(g) <= frac[0] * n:
            tr += g
        elif len(va) + len(g) <= frac[1] * n:
            va += g
        else:
            te += g
    return np.array(tr), np.array(va), np.array(te)


def strict_scaffold_split(df, frac=(0.8, 0.1, 0.1)):
    """Deterministic (DeepChem/MoleculeNet-style): biggest scaffold groups to train,
    rarest to val/test, so test scaffolds are the least represented in training."""
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(df["mol_obj"]):
        groups.setdefault(scaffold_of(m), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(df)
    tr, va, te = [], [], []
    for g in ordered:
        if len(tr) + len(g) <= frac[0] * n:
            tr += g
        elif len(va) + len(g) <= frac[1] * n:
            va += g
        else:
            te += g
    return np.array(tr), np.array(va), np.array(te)


def eval_splits(df):
    """Every outer evaluation used in the report: name -> (train_idx, test_idx).
    Validation is folded into train; model selection happens inside train only."""
    y = df["y"].values
    out = {}
    for k, (tr, te) in enumerate(random_folds(y, 5, seed=0)):
        out[f"random{k}"] = (tr, te)
    tr, va, te = strict_scaffold_split(df)
    out["strict"] = (np.concatenate([tr, va]), te)
    for s in range(3):
        tr, va, te = scaffold_split(df, s)
        out[f"scaffold{s}"] = (np.concatenate([tr, va]), te)
    return out


def random_folds(y, n_splits=5, seed=0):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

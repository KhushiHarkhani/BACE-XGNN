"""Feature blocks. Each block is cached to data/features.npz so models can pick subsets."""
import numpy as np
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator as G

from . import data

CACHE = data.DATA.parent / "features.npz"


def _counts(gen, mols, n):
    out = np.zeros((len(mols), n), dtype=np.float32)
    for i, m in enumerate(mols):
        for k, v in gen.GetCountFingerprint(m).GetNonzeroElements().items():
            out[i, k] = v
    return out


def build_from_mols(mols):
    """Every block that can be computed from a molecule alone (what the API uses)."""
    desc, _ = data.rdkit_desc(mols)
    return {
        "ecfp4": _counts(G.GetMorganGenerator(radius=2, fpSize=2048), mols, 2048),
        "ecfp6": _counts(G.GetMorganGenerator(radius=3, fpSize=2048), mols, 2048),
        "fcfp4": _counts(G.GetMorganGenerator(radius=2, fpSize=1024,
                         atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()), mols, 1024),
        "atompair": _counts(G.GetAtomPairGenerator(fpSize=2048), mols, 2048),
        "torsion": _counts(G.GetTopologicalTorsionGenerator(fpSize=1024), mols, 1024),
        "maccs": np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols], dtype=np.float32),
        "avalon": np.array([list(pyAvalonTools.GetAvalonFP(m, 1024)) for m in mols], dtype=np.float32),
        "rdkit": desc,
    }


def build(df):
    blocks = build_from_mols(list(df["mol_obj"]))
    # Dataset-only descriptor columns. NOT computable for new molecules, so no deployed model uses them.
    csv_cols = [c for c in df.columns if c not in data.LEAK_COLS | {"mol_obj", "smiles", "y"}
                and df[c].dtype != object]
    blocks["csv"] = np.nan_to_num(df[csv_cols].to_numpy(dtype=np.float32))
    return blocks


def get(df):
    if CACHE.exists():
        z = np.load(CACHE)
        if z["ecfp4"].shape[0] == len(df):
            return {k: z[k] for k in z.files}
    b = build(df)
    np.savez_compressed(CACHE, **b)
    return b

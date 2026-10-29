"""Explanations for the D-MPNN: atom occlusion + SMARTS functional-group (symbolic) layer.

atom_importance: leave-one-atom-out. Atom i's features and its bonds' features are zeroed,
and importance = drop in P(inhibitor). Positive = pushes toward inhibitor.
The symbolic layer maps atom scores onto chemically named groups via SMARTS.
These describe the *model's* behaviour, not proven binding mechanisms.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch_geometric.data import Batch

from . import data
from .gnn import mol_to_graph

# name -> (SMARTS, why a chemist cares about it for BACE-1)
GROUPS = {
    "Amidine / guanidine": ("[NX3][CX3]=[NX2]", "classic BACE-1 warhead: binds the two catalytic aspartates"),
    "Protonatable amine": ("[NX3;H2,H1;!$(NC=O)]", "charged amine can sit on the catalytic aspartates"),
    "Hydroxyl": ("[OX2H]", "transition-state mimic, H-bonds the catalytic site"),
    "Amide": ("C(=O)[NX3]", "backbone-like H-bonding with the enzyme"),
    "Carboxylic acid": ("C(=O)[OX2H]", "polar/charged, usually unfavourable in the S1 pocket"),
    "Halogen (F/Cl/Br)": ("[F,Cl,Br]", "fills lipophilic pockets, tunes metabolic stability"),
    "Trifluoromethyl": ("C(F)(F)F", "lipophilic, metabolically stable"),
    "Phenyl ring": ("c1ccccc1", "hydrophobic contact in S1/S3 pockets"),
    "Heteroaromatic ring": ("[n,o,s;r5,r6]", "aromatic H-bond acceptor / pi stacking"),
    "Sulfonamide / sulfone": ("S(=O)(=O)", "polar acceptor group"),
    "Ether": ("[OD2]([#6])[#6]", "H-bond acceptor"),
    "Nitrile": ("C#N", "small polar acceptor"),
}
_PATS = {k: (Chem.MolFromSmarts(v[0]), v[1]) for k, v in GROUPS.items()}


def extra_vector(mol) -> np.ndarray:
    desc, _ = data.rdkit_desc([mol])
    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32)[None]
    return np.hstack([desc, maccs]).astype(np.float32)[0]


def _prob(model, graphs, extra, nets=None) -> np.ndarray:
    """P(inhibitor) averaged over the model's seed networks (or just `nets`), for graphs sharing one extra vector."""
    e = np.clip((extra - model.mu) / model.sd, -5, 5).astype(np.float32)
    b = Batch.from_data_list(graphs).to(model.dev)
    ex = torch.tensor(np.tile(e, (len(graphs), 1)), device=model.dev)
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(b, ex)[0]).cpu().numpy() for n in (nets or model.nets)], 0)


def predict_smiles(model, smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return float(_prob(model, [mol_to_graph(mol)], extra_vector(mol))[0])


def _masked(g, atoms):
    g2 = g.clone()
    atoms = list(atoms)
    g2.x[atoms] = 0
    hit = torch.isin(g2.edge_index[0], torch.tensor(atoms)) | torch.isin(g2.edge_index[1], torch.tensor(atoms))
    g2.edge_attr[hit] = 0
    return g2


def atom_importance(model, mol, nets=None) -> np.ndarray:
    g, extra = mol_to_graph(mol), extra_vector(mol)
    base = _prob(model, [g], extra, nets)[0]
    drops = _prob(model, [_masked(g, [i]) for i in range(mol.GetNumAtoms())], extra, nets)
    return (base - drops).astype(np.float32)


def group_scores(mol, imp: np.ndarray) -> list[dict]:
    """Symbolic layer: mean atom importance per matched functional group."""
    out = []
    for name, (patt, why) in _PATS.items():
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        atoms = sorted({a for m in matches for a in m})
        out.append(dict(group=name, count=len(matches), atoms=atoms, score=float(imp[atoms].mean()), note=why))
    return sorted(out, key=lambda d: abs(d["score"]), reverse=True)


def faithfulness(model, mol, imp, k=5) -> float:
    """Drop in P(predicted class) when the top-k atoms are removed minus when k random atoms are.
    Higher = the highlighted atoms matter more than chance."""
    g, extra = mol_to_graph(mol), extra_vector(mol)
    n = mol.GetNumAtoms(); k = min(k, n)
    base = _prob(model, [g], extra)[0]
    sign = 1.0 if base >= 0.5 else -1.0
    top = np.argsort(-sign * imp)[:k]
    rng = np.random.RandomState(0)
    rand = [rng.choice(n, k, replace=False) for _ in range(10)]
    p = _prob(model, [_masked(g, top)] + [_masked(g, r) for r in rand], extra)
    d_top = sign * (base - p[0])
    d_rand = np.mean([sign * (base - x) for x in p[1:]])
    return float(d_top - d_rand)


def stability(model, mol, k=5) -> float:
    """Mean pairwise Jaccard overlap of the top-k atoms between the separately trained seed networks.
    If independently trained networks highlight the same atoms, the explanation is not noise."""
    sets = []
    for net in model.nets:
        imp = atom_importance(model, mol, nets=[net])
        p = _prob(model, [mol_to_graph(mol)], extra_vector(mol), nets=[net])[0]
        sign = 1.0 if p >= 0.5 else -1.0
        sets.append(set(np.argsort(-sign * imp)[:k]))
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean([len(a & b) / len(a | b) for a, b in pairs])) if pairs else float("nan")

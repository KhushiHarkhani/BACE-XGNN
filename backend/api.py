"""FastAPI service. Run from backend/: uvicorn api:app --port 8000"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors

from bace import explain, gnn
from bace.serve_stack import Stack

ART = Path(__file__).resolve().parents[1] / "artifacts"
app = FastAPI(title="BACE-XGNN")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_model = None
_stack = None
_explainer = None


def model():
    global _model
    if _model is None:
        if not (ART / "gnn.pt").exists():
            raise HTTPException(503, "model not trained yet: run python -m bace.train_final")
        _model = gnn.load(ART / "gnn.pt")
    return _model


def explainer():
    """Graph-only network used for atom highlights (no descriptor shortcut, so hiding an atom really changes it)."""
    global _explainer
    if _explainer is None:
        f = ART / "gnn_graph.pt"
        _explainer = gnn.load(f) if f.exists() else model()
    return _explainer


def stack():
    global _stack
    if _stack is None:
        if not (ART / "stack.joblib").exists():
            raise HTTPException(503, "stack not trained yet: run python -m bace.train_stack")
        _stack = Stack(model())
    return _stack


class Req(BaseModel):
    smiles: str


def _graph(mol):
    """2D layout (RDKit depiction coordinates) normalised into a 0..1 box for the frontend to draw."""
    m = Chem.Mol(mol)
    AllChem.Compute2DCoords(m)
    pts = np.array([list(m.GetConformer().GetAtomPosition(i))[:2] for i in range(m.GetNumAtoms())])
    lo, hi = pts.min(0), pts.max(0)
    span = max((hi - lo).max(), 1e-6)
    pts = (pts - (lo + hi) / 2) / span  # centred, longest side = 1
    return dict(
        atoms=[dict(i=i, symbol=a.GetSymbol(), x=float(pts[i, 0]), y=float(-pts[i, 1])) for i, a in enumerate(m.GetAtoms())],
        bonds=[dict(a=b.GetBeginAtomIdx(), b=b.GetEndAtomIdx(), order=b.GetBondTypeAsDouble()) for b in m.GetBonds()],
    )


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": (ART / "gnn.pt").exists()}


@app.post("/predict")
def predict(r: Req):
    mol = Chem.MolFromSmiles(r.smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise HTTPException(422, "invalid SMILES")
    if mol.GetNumAtoms() > 120:
        raise HTTPException(422, "molecule too large (max 120 heavy atoms)")
    m = model()
    st = stack()
    p = st.predict(mol)
    ex = explainer()
    imp = explain.atom_importance(ex, mol)  # highlights come from the graph-only explainer network
    atoms = [dict(index=i, symbol=a.GetSymbol(), importance=float(imp[i])) for i, a in enumerate(mol.GetAtoms())]
    return dict(
        smiles=Chem.MolToSmiles(mol), probability=p, threshold=st.threshold,
        label="BACE-1 inhibitor" if p >= st.threshold else "Non-inhibitor",
        properties=dict(mol_weight=Descriptors.MolWt(mol), logp=Crippen.MolLogP(mol), hbd=rdMolDescriptors.CalcNumHBD(mol),
                        hba=rdMolDescriptors.CalcNumHBA(mol), heavy_atoms=mol.GetNumHeavyAtoms(),
                        rotatable=rdMolDescriptors.CalcNumRotatableBonds(mol)),
        atoms=sorted(atoms, key=lambda a: -abs(a["importance"])),
        groups=explain.group_scores(mol, imp),
        faithfulness=explain.faithfulness(ex, mol, imp),
        graph=_graph(mol),
        disclaimer="Computational estimate, not experimental validation.",
    )


@app.get("/graph")
def graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise HTTPException(422, "invalid SMILES")
    return _graph(mol)


@app.get("/metrics")
def metrics():
    import json
    f = ART / "metrics.json"
    if not f.exists():
        raise HTTPException(404, "metrics.json not generated yet")
    return json.loads(f.read_text(encoding="utf-8"))


@app.get("/examples")
def examples():
    """Real molecules from the dataset with their true labels (for the demo buttons)."""
    from bace import data
    df = data.load()
    rng = np.random.RandomState(7)
    pick = lambda c: [dict(smiles=s, truth=c) for s in df[df.y == c]["smiles"].sample(3, random_state=rng).tolist()]
    return pick(1) + pick(0)

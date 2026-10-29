"""Directed-MPNN (Chemprop-style) with descriptor fusion and an auxiliary pIC50 head.

pIC50 is used only as a *training* target (extra supervision). It is never an input,
so nothing leaks at prediction time. Run a quick benchmark: python -m bace.gnn
"""
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
       Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
BT = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _oh(v, choices):
    out = [float(v == c) for c in choices]
    out.append(float(v not in choices))
    return out


def atom_feat(a):
    return (_oh(a.GetAtomicNum(), ELEMENTS) + _oh(a.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            _oh(a.GetFormalCharge(), [-1, 0, 1]) + _oh(a.GetHybridization(), HYB) +
            _oh(a.GetTotalNumHs(), [0, 1, 2, 3]) + _oh(int(a.GetChiralTag()), [0, 1, 2]) +
            [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0])


def bond_feat(b):
    return (_oh(b.GetBondType(), BT) + [float(b.GetIsConjugated()), float(b.IsInRing())] +
            _oh(int(b.GetStereo()), [0, 1, 2, 3]))


def mol_to_graph(m):
    x = torch.tensor([atom_feat(a) for a in m.GetAtoms()], dtype=torch.float32)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_feat(b)
        src += [i, j]; dst += [j, i]; ef += [f, f]  # edge 2k and 2k+1 are reverses of each other
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    ea = torch.tensor(ef, dtype=torch.float32) if ef else torch.zeros((0, len(bond_feat_dummy())))
    return Data(x=x, edge_index=ei, edge_attr=ea)


def bond_feat_dummy():
    return bond_feat(Chem.MolFromSmiles("CC").GetBondWithIdx(0))


class DMPNN(nn.Module):
    def __init__(self, n_atom, n_bond, n_extra, hid=256, depth=4, drop=0.2):
        super().__init__()
        self.depth = depth
        self.w_i = nn.Linear(n_atom + n_bond, hid, bias=False)
        self.w_h = nn.Linear(hid, hid, bias=False)
        self.w_o = nn.Linear(n_atom + hid, hid)
        self.drop = nn.Dropout(drop)
        # n_extra == 0 -> graph-only network (no descriptor shortcut), used for faithful atom explanations
        self.extra = nn.Sequential(nn.Linear(n_extra, 128), nn.ReLU(), nn.Dropout(drop)) if n_extra else None
        self.ffn = nn.Sequential(nn.Linear(2 * hid + (128 if n_extra else 0), hid), nn.ReLU(), nn.Dropout(drop))
        self.cls = nn.Linear(hid, 1)
        self.reg = nn.Linear(hid, 1)

    def forward(self, b, extra):
        x, (src, dst), ea = b.x, b.edge_index, b.edge_attr
        rev = torch.arange(src.numel(), device=x.device) ^ 1
        h0 = self.w_i(torch.cat([x[src], ea], 1))
        h = torch.relu(h0)
        for _ in range(self.depth - 1):
            a = scatter(h, dst, 0, dim_size=x.size(0), reduce="sum")
            m = a[src] - h[rev]  # messages into src, excluding the one coming back from dst
            h = self.drop(torch.relu(h0 + self.w_h(m)))
        a = scatter(h, dst, 0, dim_size=x.size(0), reduce="sum")
        hv = torch.relu(self.w_o(torch.cat([x, a], 1)))
        g = torch.cat([scatter(hv, b.batch, 0, reduce="mean"), scatter(hv, b.batch, 0, reduce="sum") / 20.0], 1)
        z = self.ffn(torch.cat([g, self.extra(extra)], 1) if self.extra is not None else g)
        return self.cls(z).squeeze(-1), self.reg(z).squeeze(-1)


class GNNModel:
    """sklearn-like. fit/predict take an (n,1) array of molecule indices into the shared tables."""

    def __init__(self, graphs, extra, pic50, seeds=(0,), epochs=60, hid=256, depth=4, lr=1e-3, aux_w=0.5, device="cpu", graph_only=False):
        self.graph_only = graph_only
        self.dev = torch.device(device)
        self.g, self.extra, self.pic = graphs, extra, pic50
        self.seeds, self.epochs, self.hid, self.depth, self.lr, self.aux_w = seeds, epochs, hid, depth, lr, aux_w
        self.nets = []

    def _batch(self, idx):
        return Batch.from_data_list([self.g[i] for i in idx]), torch.tensor(self.extra[idx], dtype=torch.float32)

    def fit(self, idx, y):
        idx = idx.ravel()
        mu = self.extra[idx].mean(0); sd = self.extra[idx].std(0) + 1e-6
        self.mu, self.sd = mu, sd
        self.extra_n = np.clip((self.extra - mu) / sd, -5, 5).astype(np.float32)
        pm, ps = self.pic[idx].mean(), self.pic[idx].std()
        self.nets = []
        for s in self.seeds:
            torch.manual_seed(s); rng = np.random.RandomState(s)
            perm = rng.permutation(len(idx)); nv = max(40, len(idx) // 10)
            va, tr = idx[perm[:nv]], idx[perm[nv:]]
            yv = y[perm[:nv]]; ytr = y[perm[nv:]]
            net = DMPNN(self.g[0].x.size(1), self.g[0].edge_attr.size(1), 0 if self.graph_only else self.extra.shape[1], self.hid, self.depth).to(self.dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * int(np.ceil(len(tr) / 64)))
            best, best_state, bad = 1e9, None, 0
            for ep in range(self.epochs):
                net.train()
                order = rng.permutation(len(tr))
                for k in range(0, len(tr), 64):
                    sel = order[k:k + 64]
                    b, e = self._batch_n(tr[sel])
                    lc, lr_ = net(b, e)
                    t = torch.tensor(ytr[sel], dtype=torch.float32, device=self.dev)
                    tp = torch.tensor((self.pic[tr[sel]] - pm) / ps, dtype=torch.float32, device=self.dev)
                    loss = nn.functional.binary_cross_entropy_with_logits(lc, t) + self.aux_w * nn.functional.mse_loss(lr_, tp)
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
                net.eval()
                with torch.no_grad():
                    b, e = self._batch_n(va)
                    vl = nn.functional.binary_cross_entropy_with_logits(net(b, e)[0], torch.tensor(yv, dtype=torch.float32, device=self.dev)).item()
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 12:
                        break
            net.load_state_dict(best_state); net.eval()
            self.nets.append(net)
        return self

    def _batch_n(self, idx):
        return (Batch.from_data_list([self.g[i] for i in idx]).to(self.dev),
                torch.tensor(self.extra_n[idx], dtype=torch.float32, device=self.dev))

    def predict_proba(self, idx):
        idx = idx.ravel()
        ps = []
        with torch.no_grad():
            for net in self.nets:
                out = []
                for k in range(0, len(idx), 256):
                    b, e = self._batch_n(idx[k:k + 256])
                    out.append(torch.sigmoid(net(b, e)[0]).cpu().numpy())
                ps.append(np.concatenate(out))
        p = np.mean(ps, 0)
        return np.column_stack([1 - p, p])


def save(model, path):
    torch.save(dict(states=[n.state_dict() for n in model.nets], mu=model.mu, sd=model.sd,
                    dims=(model.g[0].x.size(1), model.g[0].edge_attr.size(1), 0 if model.graph_only else model.extra.shape[1], model.hid, model.depth)), path)


class _Stub:
    """Just enough of GNNModel for explain.py / inference on new molecules."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        na, nb, ne, hid, depth = ck["dims"]
        self.dev, self.mu, self.sd, self.nets = torch.device(device), ck["mu"], ck["sd"], []
        for st in ck["states"]:
            n = DMPNN(na, nb, ne, hid, depth); n.load_state_dict(st); n.eval(); self.nets.append(n.to(self.dev))


def load(path, device="cpu"):
    return _Stub(path, device)


def make_tables(df, blocks):
    graphs = [mol_to_graph(m) for m in df["mol_obj"]]
    extra = np.hstack([blocks["rdkit"], blocks["maccs"]]).astype(np.float32)
    return graphs, extra, df["pIC50"].values.astype(np.float32)


if __name__ == "__main__":
    import time
    from . import data, features
    torch.set_num_threads(4)
    df = data.load(); y = df["y"].values; blocks = features.get(df)
    graphs, extra, pic = make_tables(df, blocks)
    accs, aucs = [], []
    for k, (tr, te) in enumerate(data.random_folds(y, 5, seed=0)[:2]):
        t0 = time.time()
        m = GNNModel(graphs, extra, pic, seeds=(0,)).fit(tr[:, None], y[tr])
        p = m.predict_proba(te[:, None])[:, 1]
        accs.append(accuracy_score(y[te], p > .5)); aucs.append(roc_auc_score(y[te], p))
        print(f"fold {k}: acc={accs[-1]:.4f} auc={aucs[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"D-MPNN single seed: acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f}")

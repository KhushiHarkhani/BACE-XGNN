"use client";
import { useState } from "react";
import { Metrics } from "@/lib/api";

export default function Pipeline({ metrics }: { metrics: Metrics | null }) {
  const rand = metrics?.blocks[0]?.rows.at(-1);
  const gnn = metrics?.blocks[0]?.rows[0];
  const n = metrics?.dataset;
  const steps = [
    { t: "Dataset", s: "MoleculeNet BACE", d: `${n ? n.molecules.toLocaleString() : "1,513"} molecules: ${n?.inhibitors ?? 691} inhibitors, ${n?.non_inhibitors ?? 822} non-inhibitors. canvasUID, a row ID that leaks the label, is dropped.` },
    { t: "Graph", s: "atoms = nodes, bonds = edges", d: "Atoms are nodes, bonds are edges. Each atom carries element, charge, hybridisation and ring flags. Each bond carries type and conjugation." },
    { t: "Train", s: "D-MPNN + 6 classic models", d: `A D-MPNN learns from the graph, with pIC50 as an extra training target. Six fingerprint and descriptor models train alongside it.${gnn ? ` GNN alone: ${gnn.accuracy.toFixed(3)} accuracy.` : ""}` },
    { t: "Predict", s: "stacked probability", d: `A logistic model combines the seven predictions, fit on out-of-fold outputs.${rand ? ` Cross-validated: ${rand.accuracy.toFixed(3)} accuracy, ${rand.auc.toFixed(3)} AUC.` : ""}` },
    { t: "Explain", s: "atoms + chemical groups", d: "Each atom is hidden in turn and the drop in prediction is recorded. SMARTS patterns group the scores into named chemical groups." },
    { t: "Deploy", s: "FastAPI + Next.js", d: "FastAPI serves the models. Next.js serves this page." },
  ];
  const [i, setI] = useState(0);
  return (
    <div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-6">
        {steps.map((s, k) => (
          <button key={s.t} onClick={() => setI(k)} aria-pressed={i === k}
            className={`rounded-2xl border p-3 text-left transition-all ${i === k ? "border-transparent bg-[var(--ink)] text-white shadow-lg" : "border-[var(--line)] bg-white hover:-translate-y-0.5 hover:border-[var(--pink)]"}`}>
            <span className={`mono text-[0.7rem] ${i === k ? "text-[var(--lilac-2)]" : "text-[var(--muted)]"}`}>0{k + 1}</span>
            <span className="display block text-base font-bold">{s.t}</span>
            <span className={`block text-[0.72rem] leading-snug ${i === k ? "text-white/70" : "text-[var(--muted)]"}`}>{s.s}</span>
          </button>
        ))}
      </div>
      <div key={i} className="card mint mt-3 text-[0.95rem] leading-relaxed">
        <p className="display mb-1 text-lg font-bold">{steps[i].t}</p>
        <p className="text-[var(--ink-2)]">{steps[i].d}</p>
      </div>
    </div>
  );
}

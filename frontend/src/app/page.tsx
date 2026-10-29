"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import CountUp from "@/components/CountUp";
import MoleculeGraph from "@/components/MoleculeGraph";
import Pipeline from "@/components/Pipeline";
import Reveal from "@/components/Reveal";
import { Example, Graph, Metrics, getJSON } from "@/lib/api";

/** Shown until the API answers (or if it is down): a plain benzamidine-like fragment. */
const FALLBACK: Graph = {
  atoms: [
    { i: 0, symbol: "C", x: -0.3, y: 0.0 }, { i: 1, symbol: "C", x: -0.15, y: -0.26 }, { i: 2, symbol: "C", x: 0.15, y: -0.26 },
    { i: 3, symbol: "C", x: 0.3, y: 0.0 }, { i: 4, symbol: "C", x: 0.15, y: 0.26 }, { i: 5, symbol: "C", x: -0.15, y: 0.26 },
    { i: 6, symbol: "C", x: -0.6, y: 0.0 }, { i: 7, symbol: "N", x: -0.8, y: -0.2 }, { i: 8, symbol: "N", x: -0.8, y: 0.22 },
  ],
  bonds: [
    { a: 0, b: 1, order: 1.5 }, { a: 1, b: 2, order: 1.5 }, { a: 2, b: 3, order: 1.5 }, { a: 3, b: 4, order: 1.5 },
    { a: 4, b: 5, order: 1.5 }, { a: 5, b: 0, order: 1.5 }, { a: 0, b: 6, order: 1 }, { a: 6, b: 7, order: 2 }, { a: 6, b: 8, order: 1 },
  ],
};

export default function Home() {
  const [graph, setGraph] = useState<Graph>(FALLBACK);
  const [live, setLive] = useState(false);
  const [m, setM] = useState<Metrics | null>(null);

  useEffect(() => {
    getJSON<Example[]>("/examples")
      .then((ex) => getJSON<Graph>(`/graph?smiles=${encodeURIComponent(ex.find((e) => e.truth === 1)!.smiles)}`))
      .then((g) => { setGraph(g); setLive(true); })
      .catch(() => {});
    getJSON<Metrics>("/metrics").then(setM).catch(() => {});
  }, []);

  const final = m?.blocks[0]?.rows.at(-1);
  const stats = [
    { label: "Accuracy", sub: "5-fold CV", v: final?.accuracy ?? 0.838, d: 3 },
    { label: "ROC-AUC", sub: "5-fold CV", v: final?.auc ?? 0.907, d: 3 },
    { label: "Sensitivity", sub: "inhibitors found", v: final?.sens ?? 0.867, d: 3 },
    { label: "Molecules", sub: "MoleculeNet BACE", v: m?.dataset.molecules ?? 1513, d: 0 },
  ];

  return (
    <div className="space-y-20">
      <section className="grid items-center gap-10 pt-6 lg:grid-cols-[1.15fr_1fr]">
        <div>
          <p className="eyebrow mb-5">Deep Learning · Project Phase 2 · Group 59</p>
          <h1 className="display text-[clamp(2.4rem,6vw,4.4rem)] font-extrabold leading-[1.02]">
            <span className="lift"><span style={{ "--d": "0ms" } as React.CSSProperties}>Explainable graph</span></span>
            <span className="lift"><span style={{ "--d": "120ms" } as React.CSSProperties}>networks for</span></span>
            <span className="lift"><span className="accent" style={{ "--d": "240ms" } as React.CSSProperties}>BACE-1 drugs.</span></span>
          </h1>
          <p className="mt-6 max-w-lg text-[1.05rem] leading-relaxed text-[var(--ink-2)]">
            Paste a SMILES string. The model predicts whether the molecule inhibits BACE-1, an Alzheimer&apos;s drug target, and highlights the <strong className="border-b-2 border-[var(--pink)]">atoms and chemical groups</strong> behind the answer.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/predict" className="btn">Try a molecule →</Link>
            <Link href="/performance" className="btn ghost">See the results</Link>
          </div>
        </div>
        <div className="card relative overflow-hidden">
          <div className="eyebrow mb-2 flex items-center justify-between">
            <span>{live ? "A real BACE inhibitor" : "Molecular graph"}</span>
            <span className="chip"><span className="h-2 w-2 rounded-full bg-[var(--gold)]" />message passing</span>
          </div>
          <MoleculeGraph graph={graph} flow className="mx-auto w-full max-w-[420px]" />
          <p className="mt-1 text-center text-xs text-[var(--muted)]">Gold pulses: message passing along bonds.</p>
        </div>
      </section>

      <section>
        <Reveal>
          <h2 className="display mb-1 text-3xl font-bold">Held-out <span className="accent">results</span></h2>
          <p className="mb-6 max-w-2xl text-[var(--ink-2)]">Every number comes from molecules the model did not train on. </p>
        </Reveal>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {stats.map((s, k) => (
            <Reveal key={s.label} delay={k * 90}>
              <div className={`card h-full ${k % 2 ? "mint" : ""}`}>
                <div className="display text-4xl font-extrabold"><CountUp to={s.v} decimals={s.d} /></div>
                <div className="mt-1 font-semibold">{s.label}</div>
                <div className="text-xs text-[var(--muted)]">{s.sub}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <section>
        <Reveal>
          <h2 className="display mb-1 text-3xl font-bold">The <span className="accent">pipeline</span></h2>
          <div className="mb-6" />
          <Pipeline metrics={m} />
        </Reveal>
      </section>
    </div>
  );
}

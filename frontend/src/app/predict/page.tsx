"use client";
import { useEffect, useState } from "react";
import Gauge from "@/components/Gauge";
import MoleculeGraph from "@/components/MoleculeGraph";
import { Example, Prediction, getJSON, predict } from "@/lib/api";

const PROP_LABEL: Record<string, string> = { mol_weight: "Mol. weight", logp: "logP", hbd: "H-donors", hba: "H-acceptors", heavy_atoms: "Heavy atoms", rotatable: "Rotatable bonds" };

export default function Predict() {
  const [smiles, setSmiles] = useState("");
  const [res, setRes] = useState<Prediction | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [examples, setExamples] = useState<Example[]>([]);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => { getJSON<Example[]>("/examples").then(setExamples).catch(() => {}); }, []);

  async function run(s = smiles) {
    setBusy(true); setErr("");
    try { setRes(await predict(s)); } catch (e) { setErr((e as Error).message); setRes(null); }
    setBusy(false);
  }

  const imp = res ? Object.fromEntries(res.atoms.map((a) => [a.index, a.importance])) : {};
  const impArr = res ? res.graph.atoms.map((a) => imp[a.i] ?? 0) : undefined;
  const maxAtom = res ? Math.max(...res.atoms.map((a) => Math.abs(a.importance)), 1e-9) : 1;
  const maxGroup = res ? Math.max(...res.groups.map((g) => Math.abs(g.score)), 1e-9) : 1;
  const hit = res ? res.probability >= res.threshold : false;

  return (
    <div className="grid gap-6 pt-4 lg:grid-cols-[360px_1fr]">
      <aside className="space-y-4 self-start lg:sticky lg:top-6">
        <div className="card">
          <h1 className="display mb-3 text-2xl font-bold">Molecule <span className="accent">input</span></h1>
          <label htmlFor="smiles" className="eyebrow">SMILES string</label>
          <textarea id="smiles" value={smiles} onChange={(e) => setSmiles(e.target.value)} rows={4} spellCheck={false}
            placeholder="CN1C(=O)C2(CC(C)(C)Oc3ccc(...)cc32)N=C1N"
            className="mono mt-1 w-full resize-none rounded-xl border border-[var(--line)] bg-white p-3 text-[0.82rem] outline-none transition focus:border-[var(--pink)] focus:ring-4 focus:ring-[var(--pink)]/15" />
          <button className="btn mt-3 w-full" disabled={busy || !smiles.trim()} onClick={() => run()}>
            {busy ? "Analysing…" : "Predict & explain"}
          </button>
          {err && <p role="alert" className="mt-3 rounded-lg bg-[var(--bad)]/10 px-3 py-2 text-sm text-[var(--bad)]">{err}</p>}
        </div>
        {examples.length > 0 && (
          <div className="card mint">
            <p className="eyebrow mb-2">Real molecules from the dataset</p>
            <div className="space-y-1.5">
              {examples.map((e, i) => (
                <button key={i} onClick={() => { setSmiles(e.smiles); run(e.smiles); }} title={e.smiles}
                  className="flex w-full items-center gap-2 rounded-xl bg-white px-3 py-2 text-left transition hover:-translate-y-0.5 hover:shadow-md">
                  <span className={`chip shrink-0 ${e.truth ? "text-[var(--pink-deep)]" : "text-[var(--neg)]"}`}>{e.truth ? "inhibitor" : "non-inhibitor"}</span>
                  <span className="mono truncate text-[0.72rem] text-[var(--ink-2)]">{e.smiles}</span>
                </button>
              ))}
            </div>
            <p className="mt-2 text-[0.7rem] text-[var(--muted)]">Labels are the dataset&apos;s true answers. The deployed model trained on these molecules.</p>
          </div>
        )}
      </aside>

      <section className="space-y-5">
        {!res && !busy && (
          <div className="card paper grid min-h-[320px] place-items-center text-center text-[var(--muted)]">
            <div>
              <p className="display text-xl font-bold text-[var(--ink)]">No molecule yet</p>
              <p className="mt-1 text-sm">Paste a SMILES or pick an example.</p>
            </div>
          </div>
        )}
        {busy && (
          <div className="grid gap-5 md:grid-cols-2" aria-busy="true">
            <div className="skeleton h-56" /><div className="skeleton h-56" /><div className="skeleton h-40 md:col-span-2" />
          </div>
        )}
        {res && !busy && (
          <>
            <div className="grid gap-5 md:grid-cols-[1fr_1.2fr]">
              <div className="card text-center">
                <p className="eyebrow">Prediction</p>
                <Gauge value={res.probability} threshold={res.threshold} />
                <p className={`display mt-1 text-xl font-bold ${hit ? "text-[var(--pink-deep)]" : "text-[var(--neg)]"}`}>{res.label}</p>
                <p className="mt-1 text-xs text-[var(--muted)]">Tick marks the {Math.round(res.threshold * 100)}% cutoff, chosen on training data.</p>
              </div>
              <div className="card sand">
                <p className="eyebrow mb-2">Properties</p>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(res.properties).map(([k, v]) => (
                    <div key={k} className="rounded-xl bg-white/70 px-3 py-2">
                      <p className="text-[0.68rem] text-[var(--muted)]">{PROP_LABEL[k]}</p>
                      <p className="mono text-base font-semibold">{Number(v).toFixed(k === "mol_weight" || k === "logp" ? 2 : 0)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid gap-5 md:grid-cols-[1.25fr_1fr]">
              <div className="card paper flex flex-col">
                <div className="mb-1 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
                  <h2 className="display text-lg font-bold whitespace-nowrap">Why this answer</h2>
                  <div className="flex gap-3 text-[0.7rem] text-[var(--muted)]">
                    <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[var(--pos)]" />pushes to inhibitor</span>
                    <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[var(--neg)]" />pushes away</span>
                  </div>
                </div>
                <div className="flex flex-1 items-center py-2">
                  <MoleculeGraph graph={res.graph} importance={impArr} hover={hover} onHover={setHover} className="mx-auto w-full max-w-[460px]" />
                </div>
                <p className="text-center text-xs text-[var(--muted)]">Hover an atom or row. Rings mark the six strongest atoms.</p>
              </div>
              <div className="card">
                <h2 className="display mb-2 text-lg font-bold">Most influential atoms</h2>
                <ul className="space-y-1">
                  {res.atoms.slice(0, 8).map((a) => (
                    <li key={a.index} onMouseEnter={() => setHover(a.index)} onMouseLeave={() => setHover(null)}
                      className={`flex items-center gap-2 rounded-lg px-2 py-1 text-sm transition ${hover === a.index ? "bg-white shadow-sm" : ""}`}>
                      <span className="mono w-12 text-[var(--ink-2)]">{a.symbol}{a.index}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white">
                        <div className="h-2 rounded-full transition-[width] duration-700" style={{ width: `${(Math.abs(a.importance) / maxAtom) * 100}%`, background: a.importance >= 0 ? "var(--pos)" : "var(--neg)" }} />
                      </div>
                      <span className="mono w-14 text-right text-xs">{a.importance.toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 rounded-lg bg-white/60 p-2 text-[0.72rem] leading-snug text-[var(--ink-2)]">
                  Faithfulness {res.faithfulness.toFixed(3)}: extra drop in probability when the top atoms are hidden versus random atoms. Highlights come from a graph-only network, not the stack.
                </p>
              </div>
            </div>

            <div className="card mint">
              <h2 className="display mb-3 text-lg font-bold">Chemical groups</h2>
              {res.groups.length === 0 && <p className="text-sm text-[var(--muted)]">No known functional groups matched.</p>}
              <ul className="grid gap-2 sm:grid-cols-2">
                {res.groups.map((g) => (
                  <li key={g.group} onMouseEnter={() => setHover(g.atoms[0] ?? null)} onMouseLeave={() => setHover(null)}
                    className="rounded-xl bg-white p-3 transition hover:-translate-y-0.5 hover:shadow-md">
                    <div className="flex items-baseline justify-between gap-2">
                      <p className="font-semibold">{g.group} <span className="text-xs font-normal text-[var(--muted)]">×{g.count}</span></p>
                      <span className="mono text-xs" style={{ color: g.score >= 0 ? "var(--pos)" : "var(--neg)" }}>{g.score.toFixed(3)}</span>
                    </div>
                    <div className="my-1.5 h-1.5 overflow-hidden rounded-full bg-[var(--lilac)]">
                      <div className="h-1.5 rounded-full" style={{ width: `${(Math.abs(g.score) / maxGroup) * 100}%`, background: g.score >= 0 ? "var(--pos)" : "var(--neg)" }} />
                    </div>
                    <p className="text-[0.72rem] text-[var(--muted)]">{g.note}</p>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-[0.7rem] text-[var(--muted)]">{res.disclaimer}</p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

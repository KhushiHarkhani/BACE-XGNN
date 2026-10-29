"use client";
import { useEffect, useState } from "react";
import Reveal from "@/components/Reveal";
import { Metrics, Row, getJSON } from "@/lib/api";

const f = (x: number) => x.toFixed(3);
const TARGET = 0.9;

function Bar({ v, sd, color, mark }: { v: number; sd?: number; color: string; mark?: boolean }) {
  // axis runs 0.5..1.0 so differences between models are visible
  const pos = (x: number) => `${Math.max(0, Math.min(100, ((x - 0.5) / 0.5) * 100))}%`;
  return (
    <div className="relative h-3 rounded-full bg-white">
      <div className="h-3 rounded-full transition-[width] duration-1000" style={{ width: pos(v), background: color }} />
      {sd ? <div className="absolute top-1/2 h-px -translate-y-1/2 bg-[var(--ink)]/60" style={{ left: pos(v - sd), width: `calc(${pos(v + sd)} - ${pos(v - sd)})` }} /> : null}
      {mark && <div className="absolute -top-1 h-5 w-px bg-[var(--gold)]" style={{ left: pos(TARGET) }} title="0.90 target" />}
    </div>
  );
}

export default function Performance() {
  const [m, setM] = useState<Metrics | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { getJSON<Metrics>("/metrics").then(setM).catch((e) => setErr(e.message)); }, []);

  if (err) return <div className="card mt-6">Couldn&apos;t load results: {err}. Is the API running?</div>;
  if (!m) return <div className="skeleton mt-6 h-72" />;

  const final = (rows: Row[]) => rows[rows.length - 1];
  return (
    <div className="space-y-10 pt-4">
      <header>
        <h1 className="display text-4xl font-extrabold">Model <span className="accent">performance</span></h1>
        <p className="mt-2 max-w-2xl text-[var(--ink-2)]">Out-of-fold results. The gold line is the 0.90 accuracy target.</p>
      </header>

      <Reveal>
        <div className="grid gap-3 md:grid-cols-3">
          {m.blocks.map((b, k) => {
            const r = final(b.rows);
            return (
              <div key={b.title} className={`card ${["", "mint", "sand"][k]}`}>
                <p className="eyebrow">{["Random split", "Scaffold ×3", "Strict scaffold"][k]}</p>
                <p className="display mt-1 text-4xl font-extrabold">{f(r.accuracy)}</p>
                <p className="text-sm text-[var(--ink-2)]">accuracy · AUC {f(r.auc)} · MCC {f(r.mcc)}</p>
                <p className="mt-3 text-xs text-[var(--muted)]">{b.note}</p>
              </div>
            );
          })}
        </div>
      </Reveal>

      {m.ceiling && (
        <Reveal>
          <section className="card paper">
            <h2 className="display text-xl font-bold">Why accuracy stops near {final(m.blocks[0].rows).accuracy.toFixed(2)}</h2>
            <p className="mt-1 max-w-3xl text-sm text-[var(--ink-2)]">
              The label is a cutoff on a measured potency (pIC50 of 7, which matches Class for {(m.ceiling.rule_match * 100).toFixed(1)}% of molecules).
              Molecules near the cutoff are close calls. {(m.ceiling.near_share_of_molecules * 100).toFixed(0)}% of molecules sit within 0.5 of it and they make {(m.ceiling.near_share_of_errors * 100).toFixed(0)}% of the errors.
            </p>
            <div className="mt-5 grid gap-6 md:grid-cols-[1.1fr_1fr]">
              <div>
                <p className="eyebrow mb-3">Accuracy by distance from the cutoff (random 5-fold)</p>
                <div className="space-y-3">
                  {m.ceiling.bins.map((b) => (
                    <div key={b.label} className="grid grid-cols-[100px_1fr_auto] items-center gap-3 text-sm">
                      <span className="text-[var(--ink-2)]">{b.label}</span>
                      <Bar v={b.accuracy} color={b.accuracy < 0.8 ? "var(--pink)" : "var(--neg)"} />
                      <span className="mono text-xs">{f(b.accuracy)} <span className="text-[var(--muted)]">n={b.n}</span></span>
                    </div>
                  ))}
                </div>
              </div>
              <ul className="space-y-3 text-sm text-[var(--ink-2)]">
                <li><strong className="text-[var(--ink)]">Noisy boundary.</strong> Measured potencies carry experimental error, so a molecule at 6.9 and one at 7.1 get different labels while behaving almost the same.</li>
                <li><strong className="text-[var(--ink)]">Small data.</strong> 1,513 molecules is little for a deep model. The GNN alone reaches {f(m.blocks[0].rows[0].accuracy)}, below the stack.</li>
                <li><strong className="text-[var(--ink)]">Inside the noise.</strong> Fold-to-fold spread is ±{f(final(m.blocks[0].rows).acc_sd ?? 0)}, so a one-point tuning gain cannot be told apart from luck.</li>
                <li><strong className="text-[var(--ink)]">Not tried.</strong> Pretrained molecule embeddings and a wider hyperparameter search. We expect about a point from them. Going further means tuning on the test folds, which inflates the score.</li>
              </ul>
            </div>
          </section>
        </Reveal>
      )}

      {m.blocks.map((b) => (
        <Reveal key={b.title}>
          <section className="card paper">
            <h2 className="display text-xl font-bold">{b.title}</h2>
            <p className="mb-4 text-sm text-[var(--muted)]">{b.note}</p>
            <div className="space-y-4">
              {b.rows.map((r, idx) => (
                <div key={r.model} className="grid items-center gap-x-4 gap-y-1 md:grid-cols-[210px_1fr_1fr]">
                  <p className={`text-sm ${idx === b.rows.length - 1 ? "font-bold" : "font-medium"}`}>{r.model}</p>
                  <div>
                    <div className="mb-0.5 flex justify-between text-[0.7rem] text-[var(--muted)]"><span>Accuracy</span><span className="mono text-[var(--ink)]">{f(r.accuracy)}{r.acc_sd ? ` ±${f(r.acc_sd)}` : ""}</span></div>
                    <Bar v={r.accuracy} sd={r.acc_sd} color="var(--pink)" mark />
                  </div>
                  <div>
                    <div className="mb-0.5 flex justify-between text-[0.7rem] text-[var(--muted)]"><span>ROC-AUC</span><span className="mono text-[var(--ink)]">{f(r.auc)}</span></div>
                    <Bar v={r.auc} color="var(--neg)" mark />
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[520px] text-sm">
                <thead className="text-left text-xs text-[var(--muted)]"><tr><th className="py-1 font-medium">Model</th><th>F1</th><th>MCC</th><th>Sensitivity</th><th>Specificity</th></tr></thead>
                <tbody className="mono">
                  {b.rows.map((r) => (
                    <tr key={r.model} className="border-t border-[var(--line)]"><td className="py-1.5 font-sans">{r.model}</td><td>{f(r.f1)}</td><td>{f(r.mcc)}</td><td>{f(r.sens)}</td><td>{f(r.spec)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </Reveal>
      ))}

      <Reveal>
        <section className="card sand">
          <h2 className="display text-xl font-bold">Overfitting check</h2>
          <p className="mb-4 text-sm text-[var(--ink-2)]">Training-fold accuracy next to test-fold accuracy, final model.</p>
          <div className="grid gap-3 md:grid-cols-3">
            {m.blocks.map((b, k) => {
              const r = final(b.rows);
              const gap = (r.inner_acc ?? r.accuracy) - r.accuracy;
              return (
                <div key={b.title} className="rounded-2xl bg-white/70 p-4">
                  <p className="text-xs text-[var(--muted)]">{["Random", "Scaffold ×3", "Strict scaffold"][k]}</p>
                  <p className="mono mt-1 text-sm">train-fold {f(r.inner_acc ?? r.accuracy)} → test {f(r.accuracy)}</p>
                  <p className={`mt-1 text-sm font-semibold ${gap < 0.03 ? "text-[var(--good)]" : "text-[var(--bad)]"}`}>
                    {gap < 0.03 ? "No overfitting" : `${(gap * 100).toFixed(0)}-point gap`}
                  </p>
                  {gap >= 0.03 && <p className="mt-1 text-xs text-[var(--ink-2)]">Rare scaffolds are unfamiliar to the model. This is distribution shift.</p>}
                </div>
              );
            })}
          </div>
        </section>
      </Reveal>

      <Reveal>
        <section className="card">
          <h2 className="display mb-2 text-xl font-bold">Notes</h2>
          <ul className="list-disc space-y-1 pl-5 text-sm text-[var(--ink-2)]">{m.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
        </section>
      </Reveal>
    </div>
  );
}

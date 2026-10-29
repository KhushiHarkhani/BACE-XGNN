"use client";
import { useState } from "react";
import { ABOUT, CITATIONS, METHOD, RESEARCH } from "@/lib/content";

const TABS = [
  { id: "about", label: "About", items: ABOUT },
  { id: "method", label: "Method", items: METHOD },
  { id: "research", label: "Research", items: RESEARCH },
  { id: "cite", label: "Citations", items: null },
] as const;

export default function Sidebar({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("about");
  const cur = TABS.find((t) => t.id === tab)!;

  return (
    <div className="flex h-full w-full flex-col rounded-[28px] bg-[var(--lilac)] shadow-[0_30px_80px_-30px_rgb(10_40_50/0.55)] ring-1 ring-white/50">
      <div className="flex items-start justify-between gap-3 px-5 pt-5">
        <div>
          <p className="eyebrow">Project</p>
          <h2 className="display text-xl font-extrabold leading-tight">BACE-1 <span className="accent">explained</span></h2>
        </div>
        <button onClick={onClose} aria-label="Close sidebar"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-white/70 text-[var(--ink)] transition hover:bg-[var(--ink)] hover:text-white">
          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M5 5l10 10M15 5L5 15" /></svg>
        </button>
      </div>

      <div role="tablist" aria-label="Project info" className="mx-5 mt-4 grid grid-cols-4 gap-1 rounded-full bg-white/60 p-1 text-[0.72rem] font-semibold">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
            className={`rounded-full px-1 py-1.5 transition-colors ${tab === t.id ? "bg-[var(--ink)] text-white" : "text-[var(--ink-2)] hover:text-[var(--ink)]"}`}>
            {t.label}
          </button>
        ))}
      </div>

      <div key={tab} role="tabpanel" className="sidebar-scroll mt-4 flex-1 space-y-3 overflow-y-auto px-5 pb-5">
        {cur.items &&
          cur.items.map((it) => (
            <section key={it.h} className="rounded-2xl bg-white/75 p-4">
              <h3 className="display text-[0.95rem] font-bold">{it.h}</h3>
              <p className="mt-1 text-[0.85rem] leading-relaxed text-[var(--ink-2)]">{it.p}</p>
            </section>
          ))}
        {tab === "cite" && (
          <ol className="space-y-2">
            {CITATIONS.map((c, i) => (
              <li key={i} className="rounded-2xl bg-white/75 p-3.5 text-[0.8rem] leading-relaxed text-[var(--ink-2)]">
                <span className="mono mr-1.5 text-[0.7rem] text-[var(--pink-deep)]">[{i + 1}]</span>
                {c.text}
                {c.note && <span className="block text-[0.72rem] text-[var(--muted)]">{c.note}</span>}
                {(c.doi || c.url) && (
                  <a href={c.doi ? `https://doi.org/${c.doi}` : c.url} target="_blank" rel="noopener noreferrer"
                    className="mono mt-1 block break-all text-[0.7rem] text-[var(--neg)] underline-offset-2 hover:underline">
                    {c.doi ? `doi:${c.doi}` : c.url}
                  </a>
                )}
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}

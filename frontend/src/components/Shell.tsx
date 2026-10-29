"use client";
import { ReactNode, useEffect, useState } from "react";
import Nav from "./Nav";
import Sidebar from "./Sidebar";

const KEY = "bace-sidebar";

/** Page frame: rounded, toggleable project sidebar on the left, the white sheet on the right. */
export default function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  const set = (v: boolean) => {
    setOpen(v);
    try { localStorage.setItem(KEY, v ? "1" : "0"); } catch { /* storage can be blocked; the toggle still works */ }
  };

  useEffect(() => {
    let saved: boolean | null = null;
    try { const s = localStorage.getItem(KEY); if (s !== null) saved = s === "1"; } catch { /* ignore */ }
    const initial = saved ?? window.matchMedia("(min-width: 1024px)").matches;
    const id = requestAnimationFrame(() => setOpen(initial)); // next frame, so the sidebar slides in
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="mx-auto flex max-w-[1560px] justify-center gap-4">
      {open && <div onClick={() => set(false)} className="fixed inset-0 z-30 bg-[var(--ink)]/30 backdrop-blur-[2px] lg:hidden" />}
      <aside
        id="project-sidebar" aria-label="Project information" inert={!open}
        className={`z-40 overflow-hidden rounded-[28px] drop-shadow-xl
          lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)] lg:shrink-0 lg:self-start lg:transition-[width,opacity,margin] lg:duration-500 lg:ease-[cubic-bezier(.22,1,.36,1)]
          max-lg:fixed max-lg:inset-y-3 max-lg:left-3 max-lg:w-[min(340px,calc(100vw-1.5rem))] max-lg:transition-transform max-lg:duration-300
          ${open ? "lg:w-[340px] lg:opacity-100 max-lg:translate-x-0" : "lg:-mr-4 lg:w-0 lg:opacity-0 max-lg:-translate-x-[115%]"}`}
      >
        <div className="h-full w-[min(340px,calc(100vw-1.5rem))] lg:w-[340px]">
          <Sidebar onClose={() => set(false)} />
        </div>
      </aside>

      <div className="sheet flex min-h-[calc(100vh-3rem)] w-full min-w-0 max-w-6xl flex-col overflow-hidden">
        <Nav open={open} onToggle={() => set(!open)} />
        <main className="flex-1 px-5 pb-12 sm:px-10">{children}</main>
        <footer className="border-t border-[var(--line)] px-5 py-5 text-xs text-[var(--muted)] sm:px-10">
          Computational estimates, not experimental results. Highlights show model behaviour, not binding.
        </footer>
      </div>
    </div>
  );
}

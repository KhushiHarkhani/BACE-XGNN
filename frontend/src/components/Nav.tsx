"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/predict", label: "Predict & explain" },
  { href: "/performance", label: "Performance" },
];

export default function Nav({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const path = usePathname();
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 px-5 py-5 sm:px-10">
      <div className="flex items-center gap-3">
        <button onClick={onToggle} aria-expanded={open} aria-controls="project-sidebar" aria-label={open ? "Close project info" : "Open project info"}
          className={`flex items-center gap-2 rounded-full px-3.5 py-2 text-sm font-semibold transition-colors ${open ? "bg-[var(--ink)] text-white" : "bg-[var(--lilac)] text-[var(--ink)] hover:bg-[var(--lilac-2)]"}`}>
          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2.5" y="3" width="15" height="14" rx="3.5" /><path d="M7.5 3v14" />
          </svg>
          <span className="hidden sm:inline">Project info</span>
        </button>
        <Link href="/" className="display flex items-center gap-2 text-lg font-extrabold">
          <span className="grid h-8 w-8 place-items-center rounded-full bg-[var(--ink)] text-[0.7rem] text-white">XG</span>
          <span>BACE<span className="accent">-XGNN</span></span>
        </Link>
      </div>
      <nav className="flex gap-1 rounded-full bg-[var(--lilac)] p-1 text-sm font-semibold">
        {LINKS.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={`rounded-full px-4 py-1.5 transition-colors ${
              path === l.href ? "bg-[var(--ink)] text-white" : "text-[var(--ink-2)] hover:text-[var(--ink)]"
            }`}
          >
            {l.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}

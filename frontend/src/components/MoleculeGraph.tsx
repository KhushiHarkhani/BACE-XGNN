"use client";
import { Graph } from "@/lib/api";

const COLOR: Record<string, string> = { C: "#6f9097", N: "#c27ab8", O: "#d9a62e", S: "#a58e2a", F: "#2f8a6a", Cl: "#2f8a6a", Br: "#2f8a6a" };
const BOX = 80; // coordinates arrive normalised to a unit box; this is its size in drawing units

type Props = {
  graph: Graph;
  /** signed per-atom importance: positive pushes toward inhibitor */
  importance?: number[];
  hover?: number | null;
  onHover?: (i: number | null) => void;
  /** animate a pulse running along the bonds (the "message passing" picture) */
  flow?: boolean;
  className?: string;
};

/**
 * Ball-and-stick drawing. Everything (atom size, line weight, label size) is derived from the
 * mean bond length `u`, so a 10-atom and a 45-atom molecule both stay legible, and the viewBox
 * is fitted to the molecule so it fills its card.
 */
export default function MoleculeGraph({ graph, importance, hover = null, onHover, flow = false, className = "" }: Props) {
  const P = graph.atoms.map((a) => ({ x: a.x * BOX, y: a.y * BOX }));
  const lens = graph.bonds.map((b) => Math.hypot(P[b.a].x - P[b.b].x, P[b.a].y - P[b.b].y));
  const u = lens.length ? lens.reduce((s, v) => s + v, 0) / lens.length : 8;
  const pad = 1.8 * u;
  const [x0, x1] = [Math.min(...P.map((p) => p.x)) - pad, Math.max(...P.map((p) => p.x)) + pad];
  const [y0, y1] = [Math.min(...P.map((p) => p.y)) - pad, Math.max(...P.map((p) => p.y)) + pad];
  const maxImp = Math.max(...(importance ?? [0]).map(Math.abs), 1e-9);
  // ring only the 6 strongest atoms (same ones the side list shows); marking all of them just adds noise
  const top = new Set(
    (importance ?? []).map((v, i) => [Math.abs(v), i] as const).sort((a, b) => b[0] - a[0]).slice(0, 6).map(([, i]) => i),
  );

  return (
    <svg viewBox={`${x0} ${y0} ${x1 - x0} ${y1 - y0}`} className={`${className} max-h-[380px]`} role="img" aria-label="Molecular graph">
      {graph.bonds.map((b, k) => {
        const [A, B] = [P[b.a], P[b.b]];
        const len = Math.hypot(B.x - A.x, B.y - A.y) || 1;
        const [nx, ny] = [-(B.y - A.y) / len, (B.x - A.x) / len];
        const offs = b.order === 2 ? [-0.13, 0.13] : b.order === 3 ? [-0.24, 0, 0.24] : b.order === 1.5 ? [0, 0.22] : [0];
        return (
          <g key={k}>
            {offs.map((o, j) => (
              <line key={j} x1={A.x + nx * o * u} y1={A.y + ny * o * u} x2={B.x + nx * o * u} y2={B.y + ny * o * u}
                stroke="#9fb3b9" strokeWidth={0.12 * u} strokeLinecap="round"
                strokeDasharray={b.order === 1.5 && j === 1 ? `${0.25 * u} ${0.2 * u}` : undefined} />
            ))}
            {flow && (
              <line x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke="#a58e2a" strokeWidth={0.2 * u} strokeLinecap="round"
                strokeDasharray={`${0.45 * u} ${1.6 * u}`} className="flow"
                style={{ animationDelay: `${(k % 9) * 0.16}s`, ["--off" as string]: `${-2.05 * u}px` }} />
            )}
          </g>
        );
      })}
      {graph.atoms.map((a) => {
        const p = P[a.i];
        const imp = importance?.[a.i] ?? 0;
        const rel = Math.abs(imp) / maxImp;
        const sign = imp >= 0 ? "var(--pos)" : "var(--neg)";
        const r = (a.symbol === "C" ? 0.3 : 0.38) * u * (hover === a.i ? 1.22 : 1);
        return (
          <g key={a.i} style={{ cursor: onHover ? "pointer" : "default" }}
            onMouseEnter={() => onHover?.(a.i)} onMouseLeave={() => onHover?.(null)}>
            {importance && top.has(a.i) && (
              <>
                <circle cx={p.x} cy={p.y} r={r + (0.12 + 0.4 * rel) * u} fill={sign} className="halo" style={{ ["--d" as string]: `${(a.i % 5) * 0.3}s` }} />
                <circle cx={p.x} cy={p.y} r={r + 0.1 * u} fill="none" stroke={sign} strokeWidth={(0.07 + 0.1 * rel) * u} />
              </>
            )}
            <circle cx={p.x} cy={p.y} r={r} fill={COLOR[a.symbol] ?? "#7d939c"} stroke="#fff" strokeWidth={0.07 * u} style={{ transition: "r .15s" }} />
            {a.symbol !== "C" && (
              <text x={p.x} y={p.y + 0.16 * u} textAnchor="middle" fontSize={0.46 * u} fontWeight={700} fill="#fff" pointerEvents="none">{a.symbol}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

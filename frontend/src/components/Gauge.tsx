"use client";
import { useEffect, useState } from "react";

const R = 50;
const LEN = Math.PI * R;

/** Half-circle probability dial with a tick at the decision threshold (the PPT's 0.78 dial). */
export default function Gauge({ value, threshold }: { value: number; threshold: number }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(value));
    return () => cancelAnimationFrame(id);
  }, [value]);

  const ang = Math.PI * (1 - threshold);
  const tick = (r: number) => [60 + r * Math.cos(ang), 60 - r * Math.sin(ang)];
  const [x1, y1] = tick(R - 7);
  const [x2, y2] = tick(R + 7);
  const hit = value >= threshold;

  return (
    <div className="relative mx-auto w-full max-w-[260px]">
      <svg viewBox="0 0 120 72" className="w-full">
        <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="var(--lilac-2)" strokeWidth={9} strokeLinecap="round" />
        <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke={hit ? "var(--pink-deep)" : "var(--neg)"} strokeWidth={9} strokeLinecap="round"
          strokeDasharray={LEN} strokeDashoffset={LEN * (1 - shown)} style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(.22,1,.36,1), stroke .4s" }} />
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink)" strokeWidth={1.6} strokeLinecap="round" />
        <text x="10" y="70" fontSize="5" fill="var(--muted)" textAnchor="middle">0</text>
        <text x="110" y="70" fontSize="5" fill="var(--muted)" textAnchor="middle">1</text>
      </svg>
      <div className="absolute inset-x-0 bottom-3 text-center">
        <div className="display text-4xl font-extrabold leading-none">{(value * 100).toFixed(1)}<span className="text-xl">%</span></div>
      </div>
    </div>
  );
}

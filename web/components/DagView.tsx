// path: web/components/DagView.tsx
// Hand-laid SVG of the assumed causal graph: covariates on the left,
// treatment in the middle, outcome on the right. Backdoor-set nodes are
// highlighted; edges added/removed by a copilot edit are coloured.
"use client";

import type { Dag, DagEdit } from "@/lib/types";

const LABELS: Record<string, string> = {
  treatment: "coupon (T)",
  outcome: "spend (Y)",
  loyalty_score: "loyalty",
  recency_days: "recency",
  frequency_purchases: "frequency",
  monetary_value: "monetary value",
  avg_basket_size: "basket size",
  price_sensitivity: "price sensitivity",
  pre_trend: "pre-campaign trend",
  customer_segment: "segment",
  region: "region",
  preferred_category: "category",
};

export function DagView({ dag, width = 560, height = 360 }: { dag: Dag | DagEdit; width?: number; height?: number }) {
  const edges = dag.edges;
  const backdoor = new Set(dag.backdoor_variables);
  const added = new Set(("added" in dag ? dag.added : []).map((e) => e.join("->")));
  const covariates = Array.from(new Set(edges.flatMap((e) => e))).filter((n) => n !== "treatment" && n !== "outcome").sort();

  const colX = { cov: 110, treatment: width / 2, outcome: width - 90 };
  const pos: Record<string, { x: number; y: number }> = {
    treatment: { x: colX.treatment, y: height / 2 },
    outcome: { x: colX.outcome, y: height / 2 },
  };
  covariates.forEach((n, i) => {
    pos[n] = { x: colX.cov, y: 28 + (i * (height - 56)) / Math.max(covariates.length - 1, 1) };
  });

  const nodeW = (n: string) => Math.max(72, (LABELS[n] ?? n).length * 7 + 18);

  const path = (a: string, b: string) => {
    const p = pos[a],
      q = pos[b];
    if (!p || !q) return "";
    const sx = p.x + nodeW(a) / 2,
      tx = q.x - nodeW(b) / 2;
    const c1 = sx + (tx - sx) * 0.5;
    return `M ${sx} ${p.y} C ${c1} ${p.y}, ${c1} ${q.y}, ${tx - 6} ${q.y}`;
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Assumed causal graph">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#52525b" />
        </marker>
        <marker id="arrow-added" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#0284c7" />
        </marker>
      </defs>
      {edges.map(([a, b]) => {
        const isAdded = added.has(`${a}->${b}`);
        const isEffect = a === "treatment" && b === "outcome";
        return (
          <path
            key={`${a}->${b}`}
            d={path(a, b)}
            fill="none"
            stroke={isAdded ? "#0284c7" : isEffect ? "#059669" : "#a1a1aa"}
            strokeWidth={isEffect ? 3 : isAdded ? 2.5 : 1.5}
            markerEnd={isAdded ? "url(#arrow-added)" : "url(#arrow)"}
            opacity={isEffect || isAdded ? 1 : 0.8}
          />
        );
      })}
      {Object.entries(pos).map(([n, p]) => {
        const w = nodeW(n);
        const isCore = n === "treatment" || n === "outcome";
        const inBackdoor = backdoor.has(n);
        return (
          <g key={n} transform={`translate(${p.x - w / 2}, ${p.y - 14})`}>
            <rect
              width={w}
              height={28}
              rx={8}
              fill={isCore ? (n === "treatment" ? "#fde68a" : "#bbf7d0") : inBackdoor ? "#dbeafe" : "#f4f4f5"}
              stroke={isCore ? "#92400e" : inBackdoor ? "#2563eb" : "#d4d4d8"}
              strokeWidth={inBackdoor || isCore ? 1.5 : 1}
            />
            <text x={w / 2} y={18} textAnchor="middle" fontSize={11} fontFamily="ui-sans-serif, system-ui" fill="#18181b">
              {LABELS[n] ?? n}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function DagLegend() {
  return (
    <div className="mt-2 flex flex-wrap gap-3 text-xs text-zinc-500">
      <span><span className="mr-1 inline-block h-3 w-3 rounded-sm border border-blue-600 bg-blue-100 align-middle" />backdoor adjustment set</span>
      <span><span className="mr-1 inline-block h-0.5 w-4 bg-emerald-600 align-middle" />causal effect of interest</span>
      <span><span className="mr-1 inline-block h-0.5 w-4 bg-sky-600 align-middle" />edge added by the copilot</span>
    </div>
  );
}

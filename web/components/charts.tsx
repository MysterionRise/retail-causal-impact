// path: web/components/charts.tsx
// Recharts views over the engine payloads. Each chart is a plain function of
// its data so the same component serves the dashboard and the chat cards.
"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AteRow, Simulation, StrategyRow, Targeting } from "@/lib/types";
import { STRATEGY_COLORS, STRATEGY_LABELS } from "./ui";

const METHOD_LABELS: Record<string, string> = {
  naive: "Naive difference",
  ipw: "IPW",
  aipw_linear: "AIPW (linear)",
  aipw_rf: "AIPW (random forest)",
};

export function AteChart({ rows, trueAte, height = 220 }: { rows: AteRow[]; trueAte: number | null; height?: number }) {
  const data = rows.map((r) => ({
    method: METHOD_LABELS[r.method] ?? r.method,
    ate: r.ate,
    err: r.ci_lower !== null && r.ci_upper !== null ? [r.ate - r.ci_lower, r.ci_upper - r.ate] : [0, 0],
    naive: r.method === "naive",
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => `$${v}`} domain={[0, "auto"]} />
        <YAxis type="category" dataKey="method" width={140} />
        <Tooltip formatter={(v) => [`$${Number(v).toFixed(2)}`, "estimate"]} />
        {trueAte !== null && <ReferenceLine x={trueAte} stroke="#059669" strokeDasharray="4 4" label={{ value: "true", position: "top", fill: "#059669", fontSize: 11 }} />}
        <Bar dataKey="ate" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.naive ? "#f97316" : "#2563eb"} />
          ))}
          <ErrorBar dataKey="err" width={4} strokeWidth={1.5} stroke="#111827" direction="x" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function StrategyBars({ rows, height = 200, showTrue = true }: { rows: StrategyRow[]; height?: number; showTrue?: boolean }) {
  const data = [...rows]
    .sort((a, b) => b.dr_uplift_per_customer - a.dr_uplift_per_customer)
    .map((r) => ({ name: STRATEGY_LABELS[r.strategy] ?? r.strategy, key: r.strategy, dr: r.dr_uplift_per_customer, truth: r.true_uplift_per_customer ?? null }));
  const hasTruth = showTrue && data.some((d) => d.truth !== null);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }} barCategoryGap={10}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => `$${v}`} />
        <YAxis type="category" dataKey="name" width={170} />
        <Tooltip formatter={(v, name) => [`$${Number(v).toFixed(2)}`, name === "dr" ? "doubly-robust" : "true"]} />
        {hasTruth && <Legend formatter={(v) => (v === "dr" ? "Doubly-robust estimate" : "True (synthetic)")} />}
        <Bar dataKey="dr" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={STRATEGY_COLORS[d.key] ?? "#6b7280"} />
          ))}
        </Bar>
        {hasTruth && <Bar dataKey="truth" fill="#d4d4d8" radius={[0, 4, 4, 0]} />}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function UpliftCurves({ targeting, height = 240 }: { targeting: Targeting; height?: number }) {
  const names = Object.keys(targeting.curves);
  const fractions = targeting.curves[names[0]]?.fractions ?? [];
  const data = fractions.map((f, i) => {
    const row: Record<string, number> = { pct: Math.round(f * 100), random: targeting.random_baseline[i] };
    for (const n of names) row[n] = targeting.curves[n].gain[i];
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="pct" tickFormatter={(v) => `${v}%`} />
        <YAxis tickFormatter={(v) => `$${Math.round(v / 1000)}k`} />
        <Tooltip formatter={(v, name) => [`$${Math.round(Number(v)).toLocaleString()}`, STRATEGY_LABELS[String(name)] ?? name]} labelFormatter={(l) => `${l}% targeted`} />
        <Legend formatter={(v) => STRATEGY_LABELS[v] ?? v} />
        {names.map((n) => (
          <Line key={n} type="monotone" dataKey={n} stroke={STRATEGY_COLORS[n] ?? "#6b7280"} strokeWidth={2.5} dot={false} />
        ))}
        <Line type="linear" dataKey="random" stroke="#9ca3af" strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SimulationBars({ sim, height = 180 }: { sim: Simulation; height?: number }) {
  const data = [
    { name: "Naive", value: sim.naive_ate, color: "#f97316" },
    { name: "IPW", value: sim.ipw_ate, color: "#60a5fa" },
    { name: "AIPW", value: sim.aipw_ate, color: "#2563eb" },
  ];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 8, right: 16, top: 16, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" />
        <YAxis tickFormatter={(v) => `$${v}`} domain={[0, "auto"]} />
        <Tooltip formatter={(v) => [`$${Number(v).toFixed(2)}`, "estimate"]} />
        <ReferenceLine y={sim.true_ate} stroke="#059669" strokeDasharray="4 4" label={{ value: `true $${sim.true_ate.toFixed(2)}`, position: "insideTopRight", fill: "#059669", fontSize: 11 }} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

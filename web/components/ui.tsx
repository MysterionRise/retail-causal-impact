// path: web/components/ui.tsx
// Small presentational primitives shared by dashboard panels and chat cards.
"use client";

import type { ReactNode } from "react";

export function Panel({
  id,
  title,
  subtitle,
  children,
  focused,
  right,
}: {
  id?: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
  focused?: boolean;
  right?: ReactNode;
}) {
  return (
    <section
      id={id}
      className={`rounded-xl border bg-white p-5 shadow-sm transition-shadow dark:bg-zinc-900 ${
        focused ? "border-sky-500 ring-2 ring-sky-200 dark:ring-sky-900" : "border-zinc-200 dark:border-zinc-800"
      }`}
    >
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">{title}</h2>
          {subtitle && <p className="mt-0.5 text-sm text-zinc-500 dark:text-zinc-400">{subtitle}</p>}
        </div>
        {right}
      </header>
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone = "neutral" }: { label: string; value: string; hint?: ReactNode; tone?: "neutral" | "good" | "bad" | "accent" }) {
  const color = {
    neutral: "text-zinc-900 dark:text-zinc-100",
    good: "text-emerald-600 dark:text-emerald-400",
    bad: "text-rose-600 dark:text-rose-400",
    accent: "text-sky-600 dark:text-sky-400",
  }[tone];
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${color}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-zinc-500">{hint}</div>}
    </div>
  );
}

export function Card({ title, children, tone = "neutral" }: { title: string; children: ReactNode; tone?: "neutral" | "good" | "bad" }) {
  const border = { neutral: "border-zinc-200 dark:border-zinc-700", good: "border-emerald-300", bad: "border-rose-300" }[tone];
  return (
    <div className={`my-2 rounded-lg border ${border} bg-white p-3 text-sm shadow-sm dark:bg-zinc-900`}>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">{title}</div>
      {children}
    </div>
  );
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="animate-pulse space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="h-4 rounded bg-zinc-200 dark:bg-zinc-800" style={{ width: `${90 - i * 15}%` }} />
      ))}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return <div className="rounded border border-rose-300 bg-rose-50 p-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">{message}</div>;
}

export function Table({ rows, columns }: { rows: Record<string, unknown>[]; columns: { key: string; label: string; format?: (v: unknown, row: Record<string, unknown>) => ReactNode }[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-zinc-200 text-xs uppercase tracking-wide text-zinc-500 dark:border-zinc-800">
            {columns.map((c) => (
              <th key={c.key} className="py-1.5 pr-3 font-medium">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
              {columns.map((c) => (
                <td key={c.key} className="py-1.5 pr-3 tabular-nums">
                  {c.format ? c.format(r[c.key], r) : String(r[c.key] ?? "–")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const STRATEGY_LABELS: Record<string, string> = {
  uplift_model: "Causal uplift model",
  response_model: "Predictive response model",
  random: "Random",
  "oracle (true CATE)": "Oracle (true effect)",
};

export const STRATEGY_COLORS: Record<string, string> = {
  uplift_model: "#2563eb",
  response_model: "#f97316",
  random: "#9ca3af",
  "oracle (true CATE)": "#059669",
};

// path: web/lib/api.ts
// Thin typed client for the causal engine REST API.

import type {
  Calibration,
  Dag,
  DagEdit,
  Refutation,
  SegmentRow,
  Simulation,
  SubgroupEffect,
  Summary,
  Targeting,
  WhatIf,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_CAUSAL_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const qs = (params: Record<string, number | string | undefined>) =>
  "?" +
  Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");

export const api = {
  summary: () => request<Summary>("/api/summary"),
  subgroup: (filters: Record<string, unknown>) =>
    request<SubgroupEffect>("/api/subgroup-effect", { method: "POST", body: JSON.stringify({ filters }) }),
  whatif: (budget_fraction: number, cost_per_coupon: number) =>
    request<WhatIf>(`/api/whatif${qs({ budget_fraction, cost_per_coupon })}`),
  targeting: (budget_fraction: number) => request<Targeting>(`/api/targeting${qs({ budget_fraction })}`),
  dag: () => request<Dag>("/api/dag"),
  editDag: (body: { add_edges?: [string, string][]; remove_edges?: [string, string][]; base_edges?: [string, string][] }) =>
    request<DagEdit>("/api/dag/edit", { method: "POST", body: JSON.stringify(body) }),
  simulate: (confounding_strength: number, n_samples = 5000) =>
    request<Simulation>(`/api/simulate${qs({ confounding_strength, n_samples })}`),
  refutation: () => request<Refutation>("/api/refutation"),
  calibration: () => request<Calibration>("/api/calibration"),
  segments: () => request<{ segments: SegmentRow[] }>("/api/segments"),
};

export const money = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined ? "–" : `$${v.toFixed(digits)}`;
export const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;
export const num = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined ? "–" : v.toFixed(digits);

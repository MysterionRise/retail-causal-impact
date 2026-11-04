// path: web/lib/store.tsx
// Dashboard state shared between the panels and the copilot.
//
// The copilot reads it through useAgentContext (so the model knows what the
// user is looking at) and changes it through frontend tools (set_budget,
// set_confounding_strength, highlight_segment, focus_panel, apply_dag_edit).
"use client";

import { createContext, useContext, useMemo, useReducer, type ReactNode } from "react";
import type { Dag, DagEdit, Simulation } from "./types";

export type PanelId = "ate" | "targeting" | "refutation" | "dag" | "simulation" | "segments";

export interface DashboardState {
  budgetFraction: number;
  costPerCoupon: number;
  confoundingStrength: number;
  highlightedSegment: string | null;
  focusedPanel: PanelId | null;
  dag: Dag | DagEdit | null;
  simulation: Simulation | null;
}

type Action =
  | { type: "set_budget"; budgetFraction?: number; costPerCoupon?: number }
  | { type: "set_confounding"; value: number }
  | { type: "highlight_segment"; segment: string | null }
  | { type: "focus_panel"; panel: PanelId | null }
  | { type: "set_dag"; dag: Dag | DagEdit }
  | { type: "set_simulation"; simulation: Simulation };

const initialState: DashboardState = {
  budgetFraction: 0.2,
  costPerCoupon: 1,
  confoundingStrength: 2,
  highlightedSegment: null,
  focusedPanel: null,
  dag: null,
  simulation: null,
};

function reducer(state: DashboardState, action: Action): DashboardState {
  switch (action.type) {
    case "set_budget":
      return {
        ...state,
        budgetFraction: action.budgetFraction ?? state.budgetFraction,
        costPerCoupon: action.costPerCoupon ?? state.costPerCoupon,
      };
    case "set_confounding":
      return { ...state, confoundingStrength: action.value };
    case "highlight_segment":
      return { ...state, highlightedSegment: action.segment };
    case "focus_panel":
      return { ...state, focusedPanel: action.panel };
    case "set_dag":
      return { ...state, dag: action.dag };
    case "set_simulation":
      return { ...state, simulation: action.simulation, confoundingStrength: action.simulation.confounding_strength };
  }
}

const Ctx = createContext<{ state: DashboardState; dispatch: (a: Action) => void } | null>(null);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useDashboard() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useDashboard must be used inside DashboardProvider");
  return ctx;
}

export const PANEL_TITLES: Record<PanelId, string> = {
  ate: "Naive vs causal estimate",
  targeting: "Predictive ML vs causal targeting",
  refutation: "Refutation checks",
  dag: "Causal graph",
  simulation: "Confounding simulator",
  segments: "Segments",
};

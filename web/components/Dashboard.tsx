// path: web/components/Dashboard.tsx
// The dashboard: every panel is a function of the engine API plus the shared
// dashboard state (budget, confounding strength, highlighted segment, DAG).
"use client";

import { useEffect } from "react";
import { api, money, num, pct } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { PANEL_TITLES, useDashboard } from "@/lib/store";
import { AteChart, SimulationBars, StrategyBars, UpliftCurves } from "./charts";
import { DagLegend, DagView } from "./DagView";
import { ErrorNote, Panel, Skeleton, Stat, Table } from "./ui";
import type { Dag, DagEdit } from "@/lib/types";

const asEdit = (d: Dag | DagEdit | null): DagEdit | null => (d && "aipw_estimate_with_new_adjustment" in d ? (d as DagEdit) : null);

export function Dashboard() {
  const { state, dispatch } = useDashboard();
  const summary = useApi(() => api.summary(), []);
  const targeting = useApi(() => api.targeting(state.budgetFraction), [state.budgetFraction], 150);
  const whatif = useApi(() => api.whatif(state.budgetFraction, state.costPerCoupon), [state.budgetFraction, state.costPerCoupon], 150);
  const refutation = useApi(() => api.refutation(), []);
  const dagApi = useApi(() => api.dag(), []);
  const simulation = useApi(() => api.simulate(state.confoundingStrength), [state.confoundingStrength], 250);
  const segments = useApi(() => api.segments(), []);

  // Seed shared state from the API the first time; copilot edits replace it later.
  useEffect(() => {
    if (dagApi.data && !state.dag) dispatch({ type: "set_dag", dag: dagApi.data });
  }, [dagApi.data, state.dag, dispatch]);
  useEffect(() => {
    if (simulation.data) dispatch({ type: "set_simulation", simulation: simulation.data });
  }, [simulation.data, dispatch]);
  useEffect(() => {
    if (state.focusedPanel) document.getElementById(`panel-${state.focusedPanel}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [state.focusedPanel]);

  const s = summary.data;
  const dag = state.dag ?? dagApi.data;

  return (
    <div className="space-y-5">
      {/* KPI strip */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {s ? (
          <>
            <Stat label="Naive difference in spend" value={money(s.naive_ate)} hint="treated minus control, no adjustment" tone="bad" />
            <Stat label={`Causal estimate (${s.headline_method.replace("_", " ").toUpperCase()})`} value={money(s.headline_ate)} hint={s.true_ate !== null ? `true effect ${money(s.true_ate)}` : "doubly-robust"} tone="good" />
            <Stat label="Naive overstates by" value={s.naive_overstatement_pct !== null ? `${s.naive_overstatement_pct.toFixed(0)}%` : "–"} hint="loyal customers get coupons and spend more anyway" tone="accent" />
            <Stat label="Refutation checks" value={s.refutation_checks_passed !== null ? `${s.refutation_checks_passed} / ${s.refutation_checks_total}` : "–"} hint={s.robustness_value !== null ? `robustness value ${pct(s.robustness_value)}` : undefined} />
          </>
        ) : summary.error ? (
          <div className="col-span-4"><ErrorNote message={`API unreachable: ${summary.error}. Start it with make api.`} /></div>
        ) : (
          Array.from({ length: 4 }).map((_, i) => <div key={i} className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800"><Skeleton lines={2} /></div>)
        )}
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* ATE */}
        <Panel id="panel-ate" title={PANEL_TITLES.ate} subtitle="Four estimators of the same effect with 95% bootstrap intervals" focused={state.focusedPanel === "ate"}>
          {s ? <AteChart rows={s.ate_table} trueAte={s.true_ate} /> : <Skeleton lines={5} />}
          {s?.balance && (
            <p className="mt-2 text-xs text-zinc-500">
              Balance after weighting: mean SMD {num(s.balance.mean_smd_before, 3)} → {num(s.balance.mean_smd_after, 3)}, {pct(s.balance["fraction_below_0.1_after"])} of covariates under 0.1.
            </p>
          )}
        </Panel>

        {/* Targeting */}
        <Panel
          id="panel-targeting"
          title={PANEL_TITLES.targeting}
          subtitle={`Top ${Math.round(state.budgetFraction * 100)}% of held-out customers, scored by doubly-robust incremental spend`}
          focused={state.focusedPanel === "targeting"}
          right={
            <label className="flex items-center gap-2 text-xs text-zinc-500">
              budget
              <input type="range" min={5} max={60} step={5} value={Math.round(state.budgetFraction * 100)} onChange={(e) => dispatch({ type: "set_budget", budgetFraction: Number(e.target.value) / 100 })} />
              <span className="w-8 tabular-nums">{Math.round(state.budgetFraction * 100)}%</span>
            </label>
          }
        >
          {targeting.data ? (
            <>
              <StrategyBars rows={targeting.data.strategies} />
              <UpliftCurves targeting={targeting.data} height={200} />
            </>
          ) : targeting.error ? (
            <ErrorNote message={targeting.error} />
          ) : (
            <Skeleton lines={6} />
          )}
          {whatif.data && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <Stat label="coupon cost" value={money(state.costPerCoupon)} hint={<CostInput value={state.costPerCoupon} onChange={(v) => dispatch({ type: "set_budget", costPerCoupon: v })} />} />
              <Stat label="net benefit (uplift policy)" value={money(whatif.data.uplift_policy.net_benefit, 0)} hint={`ROI ${whatif.data.uplift_policy.roi !== null ? `${(whatif.data.uplift_policy.roi * 100).toFixed(0)}%` : "–"}`} tone="good" />
              <Stat label="break-even budget" value={pct(whatif.data.breakeven_budget_fraction)} hint={`predicted uplift falls below ${money(state.costPerCoupon)}`} />
            </div>
          )}
        </Panel>

        {/* DAG */}
        <Panel id="panel-dag" title={PANEL_TITLES.dag} subtitle="What we assume causes what. DoWhy derives the adjustment set from it." focused={state.focusedPanel === "dag"}>
          {dag ? (
            <>
              <DagView dag={dag} />
              <DagLegend />
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">
                Backdoor set: <span className="font-mono text-xs">{dag.backdoor_variables.join(", ") || "none"}</span>
                {" · "}regression-adjustment estimate {money(dag.dowhy_estimate)}
                {asEdit(dag)?.aipw_estimate_with_new_adjustment != null && <> · AIPW on this set {money(asEdit(dag)?.aipw_estimate_with_new_adjustment)}</>}
              </p>
            </>
          ) : dagApi.error ? (
            <ErrorNote message={dagApi.error} />
          ) : (
            <Skeleton lines={6} />
          )}
        </Panel>

        {/* Simulation */}
        <Panel
          id="panel-simulation"
          title={PANEL_TITLES.simulation}
          subtitle="Regenerate the campaign with stronger targeting bias and watch the naive estimate drift"
          focused={state.focusedPanel === "simulation"}
          right={
            <label className="flex items-center gap-2 text-xs text-zinc-500">
              confounding
              <input type="range" min={0} max={8} step={0.5} value={state.confoundingStrength} onChange={(e) => dispatch({ type: "set_confounding", value: Number(e.target.value) })} />
              <span className="w-8 tabular-nums">{state.confoundingStrength.toFixed(1)}</span>
            </label>
          }
        >
          {state.simulation ? (
            <>
              <SimulationBars sim={state.simulation} />
              <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                <Stat label="naive bias" value={money(state.simulation.naive_bias)} tone="bad" />
                <Stat label="AIPW bias" value={money(state.simulation.aipw_bias)} tone="good" />
                <Stat label="propensity AUC" value={num(state.simulation.propensity_auc, 3)} hint={`SMD ${num(state.simulation.mean_smd_before, 2)} → ${num(state.simulation.mean_smd_after, 3)}`} />
              </div>
            </>
          ) : simulation.error ? (
            <ErrorNote message={simulation.error} />
          ) : (
            <Skeleton lines={5} />
          )}
        </Panel>

        {/* Refutation */}
        <Panel id="panel-refutation" title={PANEL_TITLES.refutation} subtitle="Placebo tests should land on zero; stability tests should stay at the original estimate" focused={state.focusedPanel === "refutation"}>
          {refutation.data ? (
            <>
              <Table
                rows={refutation.data.summary as unknown as Record<string, unknown>[]}
                columns={[
                  { key: "passed", label: "", format: (v) => (v ? "✅" : "❌") },
                  { key: "test", label: "check" },
                  { key: "source", label: "source" },
                  { key: "expected", label: "expected", format: (v) => money(v as number) },
                  { key: "new_effect", label: "re-estimated", format: (v) => money(v as number | null) },
                  { key: "p_value", label: "p", format: (v) => (v === null || v === undefined ? "–" : num(v as number, 2)) },
                ]}
              />
              {refutation.data.robustness && (
                <p className="mt-2 text-xs text-zinc-500">
                  Robustness value {pct(refutation.data.robustness.robustness_value)}: an unobserved confounder would need to explain at least that share of the remaining variance of both coupon assignment and spend to reduce the estimate to zero.
                </p>
              )}
            </>
          ) : refutation.error ? (
            <ErrorNote message={refutation.error} />
          ) : (
            <Skeleton lines={6} />
          )}
        </Panel>

        {/* Segments */}
        <Panel id="panel-segments" title={PANEL_TITLES.segments} subtitle="Doubly-robust effect by customer segment" focused={state.focusedPanel === "segments"}>
          {segments.data ? (
            <Table
              rows={segments.data.segments.map((r) => ({ ...r, _hl: r.segment === state.highlightedSegment })) as unknown as Record<string, unknown>[]}
              columns={[
                { key: "segment", label: "segment", format: (v, row) => <span className={row._hl ? "rounded bg-amber-100 px-1 font-semibold dark:bg-amber-900" : ""}>{String(v)}</span> },
                { key: "n", label: "n" },
                { key: "treatment_rate", label: "coupon rate", format: (v) => pct(v as number) },
                { key: "effect_dr", label: "effect (DR)", format: (v) => money(v as number) },
                { key: "mean_predicted_cate", label: "predicted", format: (v) => money(v as number) },
                { key: "true_effect", label: "true", format: (v) => money(v as number | undefined) },
              ]}
            />
          ) : segments.error ? (
            <ErrorNote message={segments.error} />
          ) : (
            <Skeleton lines={5} />
          )}
        </Panel>
      </div>
    </div>
  );
}

function CostInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      min={0}
      step={0.5}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="mt-1 w-20 rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900"
      aria-label="cost per coupon"
    />
  );
}

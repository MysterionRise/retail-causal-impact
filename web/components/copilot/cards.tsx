// path: web/components/copilot/cards.tsx
// Generative-UI cards rendered inside the chat for each backend tool call.
// A card receives the tool's streamed args and, once complete, the JSON the
// engine returned. Cards that change what the dashboard shows push the result
// into the shared store exactly once per tool call.
"use client";

import { useEffect } from "react";
import { money, num, pct } from "@/lib/api";
import { useDashboard } from "@/lib/store";
import type { DagEdit, Refutation, Simulation, StrategyRow, SubgroupEffect, Summary, Targeting, WhatIf } from "@/lib/types";
import { SimulationBars, StrategyBars } from "../charts";
import { DagView } from "../DagView";
import { Card, Skeleton, STRATEGY_LABELS } from "../ui";

export type CardProps = {
  toolCallId: string;
  status: string;
  args: Record<string, unknown>;
  result?: string;
};

function parse<T>(result: string | undefined): T | { error: string } | null {
  if (!result) return null;
  try {
    return JSON.parse(result) as T;
  } catch {
    return { error: result };
  }
}

function isError(v: unknown): v is { error: string } {
  return !!v && typeof v === "object" && "error" in (v as Record<string, unknown>);
}

const applied = new Set<string>();

/** Run `fn` once per completed tool call (guards against re-renders). */
function useApplyOnce(toolCallId: string, ready: boolean, fn: () => void) {
  useEffect(() => {
    if (!ready || applied.has(toolCallId)) return;
    applied.add(toolCallId);
    fn();
  }, [toolCallId, ready, fn]);
}

function Pending({ title, args }: { title: string; args: Record<string, unknown> }) {
  return (
    <Card title={title}>
      <div className="mb-2 font-mono text-[11px] text-zinc-500">{JSON.stringify(args)}</div>
      <Skeleton lines={2} />
    </Card>
  );
}

function Failed({ title, error }: { title: string; error: string }) {
  return (
    <Card title={title} tone="bad">
      <div className="text-rose-700 dark:text-rose-300">{error}</div>
    </Card>
  );
}

export function SubgroupCard({ toolCallId, status, args, result }: CardProps) {
  const data = parse<SubgroupEffect>(result);
  if (status !== "complete" || !data) return <Pending title="Subgroup effect" args={args} />;
  if (isError(data)) return <Failed title="Subgroup effect" error={data.error} />;
  const lift = data.effect_dr - data.population_ate;
  return (
    <Card title={`Subgroup effect · ${data.filters.join(" · ")}`} tone="good">
      <div className="grid grid-cols-3 gap-2">
        <Metric label="incremental spend / customer" value={money(data.effect_dr)} sub={`95% CI ${money(data.ci_lower)} to ${money(data.ci_upper)}`} />
        <Metric label="vs population" value={`${lift >= 0 ? "+" : ""}${money(lift)}`} sub={`population ${money(data.population_ate)}`} />
        <Metric label="customers" value={data.n.toLocaleString()} sub={`${pct(data.share_of_population)} of base, ${data.n_treated.toLocaleString()} got coupons`} />
      </div>
      <div className="mt-2 text-xs text-zinc-500">
        Naive treated-vs-control gap {money(data.naive_difference)} · model predicted {money(data.mean_predicted_cate)}
        {data.true_effect !== null && <> · true effect {money(data.true_effect)} (synthetic)</>}
      </div>
      <input type="hidden" value={toolCallId} />
    </Card>
  );
}

export function WhatIfCard({ toolCallId, status, args, result }: CardProps) {
  const { dispatch } = useDashboard();
  const data = parse<WhatIf>(result);
  const ok = status === "complete" && !!data && !isError(data);
  useApplyOnce(toolCallId, ok, () => {
    if (ok) dispatch({ type: "set_budget", budgetFraction: data.budget_fraction, costPerCoupon: data.cost_per_coupon });
  });
  if (status !== "complete" || !data) return <Pending title="What-if budget" args={args} />;
  if (isError(data)) return <Failed title="What-if budget" error={data.error} />;
  const rows: StrategyRow[] = [
    { strategy: "uplift_model", budget_fraction: data.budget_fraction, n_targeted: data.uplift_policy.n_targeted, dr_uplift_per_customer: data.uplift_policy.dr_uplift_per_customer, dr_total_uplift: data.uplift_policy.dr_total_uplift },
    { strategy: "response_model", budget_fraction: data.budget_fraction, n_targeted: data.response_model_policy.n_targeted, dr_uplift_per_customer: data.response_model_policy.dr_uplift_per_customer, dr_total_uplift: data.response_model_policy.dr_total_uplift },
    { strategy: "random", budget_fraction: data.budget_fraction, n_targeted: data.random_policy.n_targeted, dr_uplift_per_customer: data.random_policy.dr_uplift_per_customer, dr_total_uplift: data.random_policy.dr_total_uplift },
  ];
  return (
    <Card title={`What-if · ${pct(data.budget_fraction)} budget at ${money(data.cost_per_coupon)} per coupon`} tone="good">
      <div className="grid grid-cols-3 gap-2">
        <Metric label="net benefit" value={money(data.uplift_policy.net_benefit, 0)} sub={`${data.uplift_policy.n_targeted.toLocaleString()} coupons, cost ${money(data.uplift_policy.cost, 0)}`} />
        <Metric label="ROI" value={data.uplift_policy.roi !== null ? `${(data.uplift_policy.roi * 100).toFixed(0)}%` : "–"} sub={`incremental spend ${money(data.uplift_policy.dr_total_uplift, 0)}`} />
        <Metric label="break-even budget" value={pct(data.breakeven_budget_fraction)} sub={`treat everyone: ${money(data.treat_everyone.net_benefit, 0)} net`} />
      </div>
      <StrategyBars rows={rows} height={130} showTrue={false} />
    </Card>
  );
}

export function TargetingCard({ status, args, result }: CardProps) {
  const data = parse<Targeting>(result);
  if (status !== "complete" || !data) return <Pending title="Targeting comparison" args={args} />;
  if (isError(data)) return <Failed title="Targeting comparison" error={data.error} />;
  const by = Object.fromEntries(data.strategies.map((s) => [s.strategy, s]));
  const u = by.uplift_model?.dr_uplift_per_customer,
    r = by.response_model?.dr_uplift_per_customer;
  return (
    <Card title={`Predictive ML vs causal targeting · top ${pct(data.budget_fraction)} of ${data.n_heldout.toLocaleString()} held-out customers`} tone="good">
      <StrategyBars rows={data.strategies} height={160} />
      {u !== undefined && r !== undefined && (
        <div className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
          {STRATEGY_LABELS.uplift_model} yields {money(u)} per targeted customer, {(u / Math.max(r, 0.01)).toFixed(1)}× the {STRATEGY_LABELS.response_model.toLowerCase()} ({money(r)}).
        </div>
      )}
    </Card>
  );
}

export function DagEditCard({ toolCallId, status, args, result }: CardProps) {
  const { dispatch } = useDashboard();
  const data = parse<DagEdit>(result);
  const ok = status === "complete" && !!data && !isError(data);
  useApplyOnce(toolCallId, ok, () => {
    if (ok) dispatch({ type: "set_dag", dag: data });
  });
  if (status !== "complete" || !data) return <Pending title="Causal graph edit" args={args} />;
  if (isError(data)) return <Failed title="Causal graph edit" error={data.error} />;
  const addedBackdoor = data.backdoor_variables.filter((v) => !data.baseline_backdoor_variables.includes(v));
  const removedBackdoor = data.baseline_backdoor_variables.filter((v) => !data.backdoor_variables.includes(v));
  return (
    <Card title="Causal graph edit" tone={data.identified ? "good" : "bad"}>
      <DagView dag={data} width={520} height={300} />
      <div className="mt-2 space-y-1 text-xs">
        <div>
          Added {data.added.length ? data.added.map((e) => e.join(" → ")).join(", ") : "nothing"}; removed {data.removed.length ? data.removed.map((e) => e.join(" → ")).join(", ") : "nothing"}.
        </div>
        <div>
          Backdoor set now <span className="font-mono">{data.backdoor_variables.join(", ") || "∅"}</span>
          {addedBackdoor.length > 0 && <> (+{addedBackdoor.join(", ")})</>}
          {removedBackdoor.length > 0 && <> (−{removedBackdoor.join(", ")})</>}.
        </div>
        {data.identified ? (
          <div>
            Regression adjustment {money(data.baseline_dowhy_estimate)} → {money(data.dowhy_estimate)}; AIPW on the new set {money(data.aipw_estimate_with_new_adjustment)} (pipeline headline {money(data.pipeline_headline_ate)}
            {data.true_ate !== null && <>, true {money(data.true_ate)}</>}).
          </div>
        ) : (
          <div className="text-rose-700">The effect is no longer identified from observed variables under this graph.</div>
        )}
      </div>
    </Card>
  );
}

export function SimulationCard({ toolCallId, status, args, result }: CardProps) {
  const { dispatch } = useDashboard();
  const data = parse<Simulation>(result);
  const ok = status === "complete" && !!data && !isError(data);
  useApplyOnce(toolCallId, ok, () => {
    if (ok) dispatch({ type: "set_simulation", simulation: data });
  });
  if (status !== "complete" || !data) return <Pending title="Confounding simulation" args={args} />;
  if (isError(data)) return <Failed title="Confounding simulation" error={data.error} />;
  return (
    <Card title={`Confounding strength ${data.confounding_strength} · ${data.n_samples.toLocaleString()} simulated customers`} tone="good">
      <SimulationBars sim={data} height={150} />
      <div className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
        Naive bias {money(data.naive_bias)}, AIPW bias {money(data.aipw_bias)}; propensity AUC {num(data.propensity_auc, 2)}; mean SMD {num(data.mean_smd_before, 2)} → {num(data.mean_smd_after, 3)} after weighting.
      </div>
    </Card>
  );
}

export function RefutationCard({ status, args, result }: CardProps) {
  const data = parse<Refutation>(result);
  if (status !== "complete" || !data) return <Pending title="Refutation checks" args={args} />;
  if (isError(data)) return <Failed title="Refutation checks" error={data.error} />;
  const passed = data.summary.filter((r) => r.passed).length;
  return (
    <Card title={`Refutation checks · ${passed}/${data.summary.length} passed`} tone={passed === data.summary.length ? "good" : "bad"}>
      <ul className="space-y-1 text-xs">
        {data.summary.map((r) => (
          <li key={r.test} className="flex items-baseline gap-2">
            <span>{r.passed ? "✅" : "❌"}</span>
            <span className="flex-1">
              {r.test} <span className="text-zinc-400">[{r.source}]</span>
            </span>
            <span className="tabular-nums text-zinc-600 dark:text-zinc-300">
              {money(r.new_effect)} <span className="text-zinc-400">(expect {money(r.expected)})</span>
            </span>
          </li>
        ))}
      </ul>
      {data.robustness && (
        <div className="mt-2 text-xs text-zinc-500">
          Robustness value {pct(data.robustness.robustness_value)}: an unobserved confounder would need to explain at least that share of the remaining variance of both treatment and outcome to zero out the estimate.
        </div>
      )}
    </Card>
  );
}

export function SummaryCard({ status, args, result }: CardProps) {
  const data = parse<Summary>(result);
  if (status !== "complete" || !data) return <Pending title="Analysis summary" args={args} />;
  if (isError(data)) return <Failed title="Analysis summary" error={data.error} />;
  return (
    <Card title="Analysis summary" tone="good">
      <div className="grid grid-cols-3 gap-2">
        <Metric label="naive difference" value={money(data.naive_ate)} sub="no adjustment" />
        <Metric label={`causal (${data.headline_method.replace("_", " ")})`} value={money(data.headline_ate)} sub={data.true_ate !== null ? `true ${money(data.true_ate)}` : "doubly-robust"} />
        <Metric label="naive overstates by" value={data.naive_overstatement_pct !== null ? `${data.naive_overstatement_pct.toFixed(0)}%` : "–"} sub={`${data.n_customers.toLocaleString()} customers, ${pct(data.treatment_rate)} got coupons`} />
      </div>
      <div className="mt-2 text-xs text-zinc-500">
        Refutation {data.refutation_checks_passed}/{data.refutation_checks_total} passed · robustness value {pct(data.robustness_value)} · CATE model {data.cate_method}
      </div>
    </Card>
  );
}

export function GenericCard({ name, status, args, result }: CardProps & { name: string }) {
  const data = parse<unknown>(result);
  if (status !== "complete" || data === null) return <Pending title={name} args={args} />;
  if (isError(data)) return <Failed title={name} error={data.error} />;
  return (
    <Card title={name}>
      <pre className="max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-zinc-600 dark:text-zinc-300">{JSON.stringify(data, null, 1)}</pre>
    </Card>
  );
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md bg-zinc-50 px-2 py-1.5 dark:bg-zinc-950">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-[11px] text-zinc-500">{sub}</div>}
    </div>
  );
}

// path: web/lib/types.ts
// Payloads returned by the causal engine API (services/api). Kept hand-written
// and small; the engine's `_clean` makes every number a JSON number or null.

export interface AteRow {
  method: string;
  ate: number;
  ci_lower: number | null;
  ci_upper: number | null;
}

export interface HeldoutMetric {
  method: string;
  mean_cate_test: number;
  std_cate_test: number;
  auuc_heldout: number;
  corr_true_cate_heldout?: number;
  [key: string]: number | string | undefined;
}

export interface Summary {
  n_customers: number;
  treatment_rate: number;
  mean_outcome: number;
  true_ate: number | null;
  naive_ate: number;
  headline_ate: number;
  headline_method: string;
  naive_overstatement_pct: number | null;
  ate_table: AteRow[];
  cate_method: string | null;
  balance: { mean_smd_before: number; mean_smd_after: number; "fraction_below_0.1_after": number } | null;
  overlap: { common_support_min: number; common_support_max: number; treated_poor_overlap: number; control_poor_overlap: number } | null;
  auuc_heldout: number | null;
  heldout_metrics: HeldoutMetric[] | null;
  refutation_checks_passed: number | null;
  refutation_checks_total: number | null;
  robustness_value: number | null;
  backdoor_variables: string[] | null;
  propensity_model: string | null;
}

export interface SubgroupEffect {
  filters: string[];
  n: number;
  n_treated: number;
  share_of_population: number;
  effect_dr: number;
  se: number;
  ci_lower: number;
  ci_upper: number;
  naive_difference: number | null;
  mean_predicted_cate: number;
  population_ate: number;
  true_effect: number | null;
  mean_outcome: number;
}

export interface PolicyValue {
  n_targeted: number;
  predicted_total_uplift: number;
  dr_total_uplift: number;
  dr_uplift_per_customer: number;
  cost: number;
  net_benefit: number;
  roi: number | null;
  true_total_uplift?: number;
}

export interface WhatIf {
  budget_fraction: number;
  cost_per_coupon: number;
  uplift_policy: PolicyValue;
  random_policy: PolicyValue;
  response_model_policy: PolicyValue;
  treat_everyone: PolicyValue;
  breakeven_budget_fraction: number;
  cate_threshold: number;
}

export interface StrategyRow {
  strategy: string;
  budget_fraction: number;
  n_targeted: number;
  dr_uplift_per_customer: number;
  dr_total_uplift: number;
  true_uplift_per_customer?: number;
  true_total_uplift?: number;
}

export interface Targeting {
  budget_fraction: number;
  n_heldout: number;
  strategies: StrategyRow[];
  curves: Record<string, { fractions: number[]; gain: number[] }>;
  random_baseline: number[];
}

export interface Dag {
  edges: [string, string][];
  backdoor_variables: string[];
  identified: boolean;
  dowhy_estimate: number | null;
  graph: string;
  nodes: string[];
  available_variables?: string[];
}

export interface DagEdit extends Dag {
  added: [string, string][];
  removed: [string, string][];
  baseline_backdoor_variables: string[];
  baseline_dowhy_estimate: number | null;
  aipw_estimate_with_new_adjustment: number | null;
  pipeline_headline_ate: number;
  true_ate: number | null;
}

export interface Simulation {
  confounding_strength: number;
  n_samples: number;
  true_ate: number;
  naive_ate: number;
  ipw_ate: number;
  aipw_ate: number;
  naive_bias: number;
  aipw_bias: number;
  propensity_auc: number;
  mean_smd_before: number;
  mean_smd_after: number;
}

export interface RefutationRow {
  test: string;
  source: string;
  expected: number;
  original_effect: number;
  new_effect: number | null;
  p_value: number | null;
  passed: boolean;
}

export interface Refutation {
  summary: RefutationRow[];
  robustness: { robustness_value: number; robustness_value_alpha: number; partial_r2_treatment: number; ols_estimate: number } | null;
  leave_one_out: { feature_dropped: string; ate_without_feature: number; ate_change: number; pct_change: number }[] | null;
  backdoor_variables: string[] | null;
  dowhy_estimate: number | null;
  headline_ate: number;
}

export interface SegmentRow {
  segment: string;
  n: number;
  treatment_rate: number;
  mean_predicted_cate: number;
  effect_dr: number;
  mean_outcome: number;
  true_effect?: number;
}

export interface Calibration {
  cate_method: string | null;
  deciles: { decile: number; n: number; predicted_cate: number; dr_estimate: number; dr_se: number; true_cate?: number }[];
  heldout_metrics: HeldoutMetric[];
}

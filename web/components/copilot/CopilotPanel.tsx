// path: web/components/copilot/CopilotPanel.tsx
// Wires the copilot to the dashboard:
//  - useAgentContext: the model sees what the user is looking at
//  - useFrontendTool: the model can change the dashboard (budget, slider,
//    highlighted segment, focused panel). These run in the browser; the
//    Python agent returns them to the client as AG-UI tool calls.
//  - useRenderTool: backend tool calls become generative-UI cards in chat.
"use client";

import { CopilotSidebar, useAgentContext, useConfigureSuggestions, useDefaultRenderTool, useFrontendTool, useRenderTool } from "@copilotkit/react-core/v2";
import { z } from "zod";
import { PANEL_TITLES, useDashboard, type PanelId } from "@/lib/store";
import { DagEditCard, GenericCard, RefutationCard, SimulationCard, SubgroupCard, SummaryCard, TargetingCard, WhatIfCard, type CardProps } from "./cards";

const PANEL_IDS = ["ate", "targeting", "refutation", "dag", "simulation", "segments"] as const;
const SEGMENTS = ["high_value", "mid_value", "low_value", "new_customer"] as const;

// Tool argument schemas (mirrors services/api/tools.py; only used for typing
// the streamed args, the backend validates for real).
const anyArgs = z.object({}).passthrough();
const filtersArgs = z.object({ filters: z.record(z.string(), z.unknown()).optional() });
const budgetArgs = z.object({ budget_fraction: z.number(), cost_per_coupon: z.number().optional() });
const targetingArgs = z.object({ budget_fraction: z.number().optional() });
const edgeList = z.array(z.tuple([z.string(), z.string()]));
const dagArgs = z.object({ add_edges: edgeList.optional(), remove_edges: edgeList.optional(), base_edges: edgeList.optional() });
const simArgs = z.object({ confounding_strength: z.number(), n_samples: z.number().optional() });

type RenderArgs = { name: string; toolCallId: string; parameters: unknown; status: string; result?: string };

const toCardProps = (p: RenderArgs): CardProps => ({
  toolCallId: p.toolCallId,
  status: String(p.status),
  args: (p.parameters && typeof p.parameters === "object" ? (p.parameters as Record<string, unknown>) : {}),
  result: p.result,
});

export function CopilotPanel() {
  const { state, dispatch } = useDashboard();

  // ---- context the model can see --------------------------------------
  useAgentContext({
    description: "What the user currently sees on the causal dashboard",
    value: {
      budget_fraction: state.budgetFraction,
      cost_per_coupon: state.costPerCoupon,
      confounding_strength_slider: state.confoundingStrength,
      highlighted_segment: state.highlightedSegment,
      focused_panel: state.focusedPanel,
      dag_edges: state.dag?.edges ?? null,
      dag_backdoor_set: state.dag?.backdoor_variables ?? null,
      panels: PANEL_IDS.map((id) => `${id}: ${PANEL_TITLES[id]}`),
    },
  });

  // ---- frontend tools: the model drives the dashboard -----------------
  useFrontendTool(
    {
      name: "set_budget",
      description: "Change the targeting budget share (and optionally the cost per coupon) shown on the dashboard. The targeting and what-if panels re-render immediately.",
      parameters: z.object({
        budget_fraction: z.number().min(0.05).max(0.6).describe("Share of customers to target, 0.05 to 0.6"),
        cost_per_coupon: z.number().min(0).optional().describe("Cost of one coupon in dollars"),
      }),
      handler: async ({ budget_fraction, cost_per_coupon }) => {
        dispatch({ type: "set_budget", budgetFraction: budget_fraction, costPerCoupon: cost_per_coupon });
        dispatch({ type: "focus_panel", panel: "targeting" });
        return `Dashboard budget set to ${Math.round(budget_fraction * 100)}%${cost_per_coupon !== undefined ? ` at $${cost_per_coupon} per coupon` : ""}.`;
      },
    },
    [dispatch],
  );

  useFrontendTool(
    {
      name: "set_confounding_strength",
      description: "Move the confounding-strength slider on the simulator panel (0 = randomised assignment, 2 = the default campaign, up to 8). The panel regenerates the campaign and shows naive vs IPW vs AIPW against the truth.",
      parameters: z.object({ confounding_strength: z.number().min(0).max(8) }),
      handler: async ({ confounding_strength }) => {
        dispatch({ type: "set_confounding", value: confounding_strength });
        dispatch({ type: "focus_panel", panel: "simulation" });
        return `Confounding slider set to ${confounding_strength}; the simulator panel is updating.`;
      },
    },
    [dispatch],
  );

  useFrontendTool(
    {
      name: "highlight_segment",
      description: "Highlight a customer segment in the segments panel (or clear the highlight).",
      parameters: z.object({ segment: z.enum(SEGMENTS).nullable() }),
      handler: async ({ segment }) => {
        dispatch({ type: "highlight_segment", segment });
        dispatch({ type: "focus_panel", panel: "segments" });
        return segment ? `Highlighted ${segment}.` : "Cleared the segment highlight.";
      },
    },
    [dispatch],
  );

  useFrontendTool(
    {
      name: "focus_panel",
      description: "Scroll the dashboard to one panel and outline it: ate (naive vs causal), targeting (ML vs causal), refutation, dag (causal graph), simulation (confounding slider), segments.",
      parameters: z.object({ panel: z.enum(PANEL_IDS) }),
      handler: async ({ panel }) => {
        dispatch({ type: "focus_panel", panel: panel as PanelId });
        return `Focused the ${PANEL_TITLES[panel as PanelId]} panel.`;
      },
    },
    [dispatch],
  );

  // ---- generative UI for backend tool calls ---------------------------
  useRenderTool({ name: "get_subgroup_effect", parameters: filtersArgs, render: (p) => <SubgroupCard {...toCardProps(p)} /> });
  useRenderTool({ name: "whatif_budget", parameters: budgetArgs, render: (p) => <WhatIfCard {...toCardProps(p)} /> });
  useRenderTool({ name: "compare_targeting", parameters: targetingArgs, render: (p) => <TargetingCard {...toCardProps(p)} /> });
  useRenderTool({ name: "edit_dag", parameters: dagArgs, render: (p) => <DagEditCard {...toCardProps(p)} /> });
  useRenderTool({ name: "simulate_confounding", parameters: simArgs, render: (p) => <SimulationCard {...toCardProps(p)} /> });
  useRenderTool({ name: "get_refutation", parameters: anyArgs, render: (p) => <RefutationCard {...toCardProps(p)} /> });
  useRenderTool({ name: "get_summary", parameters: anyArgs, render: (p) => <SummaryCard {...toCardProps(p)} /> });
  useDefaultRenderTool({ render: (p) => <GenericCard {...toCardProps(p)} name={p.name} /> });

  useConfigureSuggestions({
    suggestions: [
      { title: "What did the analysis find?", message: "What did the analysis find? Lead with the headline numbers." },
      { title: "Effect for new customers in the north", message: "What is the coupon effect for new customers in the north region, and how does it compare to the population?" },
      { title: "What if coupons cost $3?", message: "What if a coupon costs $3 and we target 15% of customers? Show net benefit and ROI." },
      { title: "Assume region drives targeting", message: "Assume region also drives who gets a coupon. Edit the graph, re-identify and tell me if the estimate changes." },
      { title: "Why trust this estimate?", message: "Why should I trust the causal estimate? Walk me through the refutation checks." },
      { title: "Crank up the confounding", message: "Set the confounding strength to 6 and explain what happens to the naive and causal estimates." },
    ],
  });

  return (
    <CopilotSidebar
      defaultOpen
      labels={{
        modalHeaderTitle: "Causal copilot",
        sidebarLabel: "Causal copilot",
        welcomeMessageText: "Ask about the coupon campaign: effects for any subgroup, what-if budgets, the causal graph, or why the estimate can be trusted. Every number comes from the causal engine.",
        chatInputPlaceholder: "Ask about effects, budgets, the graph, or the checks…",
      }}
    />
  );
}

# path: services/api/tools.py
"""
Tool registry shared by the copilot agent and the REST API.

Each tool is a thin, validated wrapper around an ``Engine`` method. The
registry produces the Anthropic ``tools`` definitions and dispatches calls,
so the agent, the HTTP routes and the tests all exercise the same code.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .engine import Engine

FILTER_SCHEMA = {
    "type": "object",
    "description": (
        "Subgroup filters on raw customer columns. Categorical columns take a value or a list "
        "of values: customer_segment (high_value, mid_value, low_value, new_customer), region "
        "(north, south, east, west), preferred_category (grocery, electronics, apparel, home, "
        "other). Numeric columns take {min, max}: loyalty_score (0-100), recency_days, "
        "frequency_purchases, monetary_value, avg_basket_size, price_sensitivity (0-10), pre_trend."
    ),
    "properties": {
        "customer_segment": {"type": ["string", "array"], "items": {"type": "string"}},
        "region": {"type": ["string", "array"], "items": {"type": "string"}},
        "preferred_category": {"type": ["string", "array"], "items": {"type": "string"}},
        "loyalty_score": {"$ref": "#/$defs/range"},
        "recency_days": {"$ref": "#/$defs/range"},
        "frequency_purchases": {"$ref": "#/$defs/range"},
        "monetary_value": {"$ref": "#/$defs/range"},
        "avg_basket_size": {"$ref": "#/$defs/range"},
        "price_sensitivity": {"$ref": "#/$defs/range"},
        "pre_trend": {"$ref": "#/$defs/range"},
    },
    "additionalProperties": False,
}

RANGE_DEF = {
    "range": {
        "type": "object",
        "properties": {"min": {"type": "number"}, "max": {"type": "number"}},
        "additionalProperties": False,
    }
}

EDGE_LIST = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
    "description": "List of [source, target] pairs, e.g. [[\"region\", \"treatment\"]]",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[Engine, dict], Any]
    # Tools that change what the dashboard shows publish their result as
    # shared state under this key (consumed by useCoAgent in the UI).
    state_key: str | None = None

    def to_anthropic(self) -> dict:
        schema = {"$defs": RANGE_DEF, **self.input_schema}
        return {"name": self.name, "description": self.description, "input_schema": schema}


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_summary",
        description=(
            "Headline results of the causal analysis: naive vs causal ATE with confidence "
            "intervals, true ATE (synthetic data), balance, overlap, held-out CATE metrics, "
            "refutation pass count and robustness value. Call this first when asked what the "
            "analysis found."
        ),
        input_schema=_obj({}),
        handler=lambda eng, a: eng.summary(),
    ),
    ToolSpec(
        name="get_subgroup_effect",
        description=(
            "Average coupon effect (incremental spend per customer) inside a subgroup, estimated "
            "with doubly-robust pseudo-outcomes so it is valid despite confounded assignment. "
            "Returns the estimate with a 95% CI, the naive difference for contrast, the mean "
            "predicted CATE, the population ATE and, on synthetic data, the true effect. "
            "Omit filters for the whole population."
        ),
        input_schema=_obj({"filters": FILTER_SCHEMA}),
        handler=lambda eng, a: eng.subgroup_effect(a.get("filters") or {}),
        state_key="subgroup",
    ),
    ToolSpec(
        name="whatif_budget",
        description=(
            "What-if analysis for a coupon budget: target the top share of customers by predicted "
            "uplift at a given cost per coupon and report cost, doubly-robust incremental spend, "
            "net benefit and ROI, compared with random targeting, the predictive response model "
            "and treating everyone. Also returns the break-even budget share."
        ),
        input_schema=_obj(
            {
                "budget_fraction": {"type": "number", "minimum": 0.01, "maximum": 1.0, "description": "Share of customers to target, e.g. 0.2"},
                "cost_per_coupon": {"type": "number", "minimum": 0, "description": "Cost of one coupon in dollars, default 1"},
            },
            ["budget_fraction"],
        ),
        handler=lambda eng, a: eng.whatif_budget(a["budget_fraction"], a.get("cost_per_coupon", 1.0)),
        state_key="whatif",
    ),
    ToolSpec(
        name="compare_targeting",
        description=(
            "Predictive ML (response model that predicts spend) versus causal uplift targeting on "
            "the held-out split at a budget share: incremental spend per targeted customer for "
            "each strategy plus random and oracle, and the cumulative-gain curves."
        ),
        input_schema=_obj({"budget_fraction": {"type": "number", "minimum": 0.01, "maximum": 1.0}}),
        handler=lambda eng, a: eng.compare_targeting(a.get("budget_fraction", 0.2)),
        state_key="targeting",
    ),
    ToolSpec(
        name="get_dag",
        description=(
            "The assumed causal graph (edges), the backdoor adjustment set DoWhy derives from it, "
            "and the list of variables available for edits."
        ),
        input_schema=_obj({}),
        handler=lambda eng, a: eng.dag(),
        state_key="dag",
    ),
    ToolSpec(
        name="edit_dag",
        description=(
            "Edit the assumed causal graph by adding and/or removing edges, then re-identify the "
            "adjustment set with DoWhy and re-estimate the effect (DoWhy regression adjustment and "
            "the pipeline's AIPW on the new adjustment set). Use when the user proposes a different "
            "causal assumption, e.g. 'assume region also drives who gets a coupon'. Variables: "
            "treatment, outcome, loyalty_score, recency_days, frequency_purchases, monetary_value, "
            "avg_basket_size, price_sensitivity, pre_trend, customer_segment, region, preferred_category."
        ),
        input_schema=_obj(
            {
                "add_edges": EDGE_LIST,
                "remove_edges": EDGE_LIST,
                "base_edges": {**EDGE_LIST, "description": "Optional: start from this edge list instead of the default graph (pass the edges from a previous edit to chain edits)."},
            }
        ),
        handler=lambda eng, a: eng.edit_dag(a.get("add_edges"), a.get("remove_edges"), a.get("base_edges")),
        state_key="dag",
    ),
    ToolSpec(
        name="simulate_confounding",
        description=(
            "Regenerate the synthetic campaign with a chosen confounding strength (0 = randomised, "
            "2 = default, up to 8) and compare naive, IPW and AIPW estimates with the truth. Shows "
            "how the naive estimate drifts while the causal estimators stay put."
        ),
        input_schema=_obj(
            {
                "confounding_strength": {"type": "number", "minimum": 0, "maximum": 8},
                "n_samples": {"type": "integer", "minimum": 500, "maximum": 20000},
            },
            ["confounding_strength"],
        ),
        handler=lambda eng, a: eng.simulate_confounding(a["confounding_strength"], a.get("n_samples", 5000)),
        state_key="simulation",
    ),
    ToolSpec(
        name="get_refutation",
        description=(
            "Refutation and sensitivity results for the headline estimate: DoWhy placebo / random "
            "common cause / data subset, in-house placebo and pre-treatment checks, leave-one-"
            "covariate-out and the Cinelli-Hazlett robustness value."
        ),
        input_schema=_obj({}),
        handler=lambda eng, a: eng.refutation(),
    ),
    ToolSpec(
        name="get_calibration",
        description="Held-out calibration of predicted uplift by decile and metrics for every CATE model.",
        input_schema=_obj({}),
        handler=lambda eng, a: eng.calibration(),
    ),
    ToolSpec(
        name="get_segments",
        description="Predicted and doubly-robust coupon effect by customer segment (and the true effect on synthetic data).",
        input_schema=_obj({}),
        handler=lambda eng, a: eng.segments(),
    ),
    ToolSpec(
        name="list_top_customers",
        description="The customers a budget would target first, ordered by predicted uplift (ids, segment, loyalty, predicted effect).",
        input_schema=_obj(
            {
                "budget_fraction": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            }
        ),
        handler=lambda eng, a: eng.top_customers(a.get("budget_fraction", 0.2), a.get("limit", 20)),
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def anthropic_tool_definitions() -> list[dict]:
    return [t.to_anthropic() for t in TOOLS]


def run_tool(engine: Engine, name: str, arguments: dict) -> Any:
    """Dispatch a tool call. Raises KeyError for unknown tools, ValueError for bad input."""
    spec = TOOLS_BY_NAME[name]
    return spec.handler(engine, arguments or {})


def run_tool_json(engine: Engine, name: str, arguments: dict) -> tuple[str, bool]:
    """Run a tool and return (json_text, is_error) for a tool_result block."""
    try:
        return json.dumps(run_tool(engine, name, arguments)), False
    except KeyError:
        return json.dumps({"error": f"Unknown tool {name}"}), True
    except (ValueError, TypeError) as e:
        return json.dumps({"error": str(e)}), True

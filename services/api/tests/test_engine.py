# path: services/api/tests/test_engine.py
"""Engine queries return well-formed, internally consistent numbers."""

import json

import numpy as np
import pytest

from services.api import tools


def test_summary_is_json_and_consistent(small_engine):
    s = small_engine.summary()
    json.dumps(s)  # JSON-serialisable (no numpy types, no NaN)
    assert s["n_customers"] == 1500
    assert 0.2 < s["treatment_rate"] < 0.4
    assert s["naive_ate"] > s["headline_ate"], "confounding inflates the naive estimate"
    assert s["true_ate"] is not None
    assert s["refutation_checks_total"] == 6
    assert 0 < s["robustness_value"] < 1


def test_subgroup_effect_population_matches_headline(small_engine):
    whole = small_engine.subgroup_effect({})
    assert whole["n"] == 1500
    # DR mean over everyone is itself an AIPW estimate of the ATE
    assert abs(whole["effect_dr"] - whole["population_ate"]) < 2.5
    assert whole["ci_lower"] < whole["effect_dr"] < whole["ci_upper"]
    assert abs(whole["effect_dr"] - whole["true_effect"]) < 3.0


def test_subgroup_effect_filters_and_errors(small_engine):
    new = small_engine.subgroup_effect({"customer_segment": "new_customer"})
    high = small_engine.subgroup_effect({"customer_segment": "high_value"})
    assert new["n"] + high["n"] < 1500
    assert new["true_effect"] > high["true_effect"], "new customers respond more in the generator"
    assert new["mean_predicted_cate"] > high["mean_predicted_cate"]

    ranged = small_engine.subgroup_effect({"loyalty_score": {"min": 70}, "region": ["north", "south"]})
    assert ranged["filters"] == ["loyalty_score in [70, inf]", "region in ['north', 'south']"]

    with pytest.raises(ValueError, match="Unknown values"):
        small_engine.subgroup_effect({"region": "mars"})
    with pytest.raises(ValueError, match="Unknown filter column"):
        small_engine.subgroup_effect({"shoe_size": {"min": 1}})
    with pytest.raises(ValueError, match="at least 30"):
        small_engine.subgroup_effect({"loyalty_score": {"max": 5}})


def test_whatif_budget_accounting(small_engine):
    w = small_engine.whatif_budget(0.2, cost_per_coupon=3.0)
    up = w["uplift_policy"]
    assert up["n_targeted"] == 300
    assert up["cost"] == 900.0
    assert np.isclose(up["net_benefit"], up["dr_total_uplift"] - up["cost"], atol=0.01)
    assert w["treat_everyone"]["n_targeted"] == 1500
    assert 0 <= w["breakeven_budget_fraction"] <= 1
    assert up["true_total_uplift"] > w["random_policy"]["true_total_uplift"]
    with pytest.raises(ValueError):
        small_engine.whatif_budget(1.5)


def test_compare_targeting_orders_strategies(small_engine):
    c = small_engine.compare_targeting(0.2)
    names = {r["strategy"] for r in c["strategies"]}
    assert names == {"uplift_model", "response_model", "random", "oracle (true CATE)"}
    by = {r["strategy"]: r for r in c["strategies"]}
    assert by["oracle (true CATE)"]["true_uplift_per_customer"] >= by["uplift_model"]["true_uplift_per_customer"]
    assert by["uplift_model"]["true_uplift_per_customer"] > by["response_model"]["true_uplift_per_customer"]
    assert set(c["curves"]) == {"uplift_model", "response_model"}
    assert len(c["curves"]["uplift_model"]["gain"]) == 21


def test_dag_and_edits(small_engine):
    d = small_engine.dag()
    assert "loyalty_score" in d["backdoor_variables"]
    assert ["treatment", "outcome"] in d["edges"]
    assert d["identified"]

    edited = small_engine.edit_dag(add_edges=[["region", "treatment"], ["region", "outcome"]])
    assert "region" in edited["backdoor_variables"]
    assert ["region", "treatment"] in edited["edges"]
    assert edited["aipw_estimate_with_new_adjustment"] is not None
    assert abs(edited["aipw_estimate_with_new_adjustment"] - edited["pipeline_headline_ate"]) < 3.0

    dropped = small_engine.edit_dag(remove_edges=[["loyalty_score", "treatment"]])
    assert "loyalty_score" not in dropped["backdoor_variables"]

    chained = small_engine.edit_dag(add_edges=[["pre_trend", "treatment"]], base_edges=edited["edges"])
    assert ["region", "treatment"] in chained["edges"] and ["pre_trend", "treatment"] in chained["edges"]

    with pytest.raises(ValueError):
        small_engine.edit_dag(add_edges=[["outcome", "treatment"]])
    with pytest.raises(ValueError):
        small_engine.edit_dag(add_edges=[["unicorn", "treatment"]])


def test_simulate_confounding_bias_grows(small_engine):
    weak = small_engine.simulate_confounding(0.0, n_samples=3000)
    strong = small_engine.simulate_confounding(4.0, n_samples=3000)
    assert abs(weak["naive_bias"]) < abs(strong["naive_bias"])
    assert abs(strong["aipw_bias"]) < abs(strong["naive_bias"])
    assert weak["propensity_auc"] < strong["propensity_auc"]
    with pytest.raises(ValueError):
        small_engine.simulate_confounding(99)


def test_refutation_calibration_segments(small_engine):
    r = small_engine.refutation()
    assert len(r["summary"]) == 6
    cal = small_engine.calibration()
    assert len(cal["deciles"]) == 10
    seg = small_engine.segments()["segments"]
    assert [s["segment"] for s in seg][0] in ("new_customer", "low_value")
    top = small_engine.top_customers(0.1, limit=5)
    assert top["n_targeted"] == 150 and len(top["customers"]) == 5


def test_tool_registry_dispatch(small_engine):
    defs = tools.anthropic_tool_definitions()
    assert {d["name"] for d in defs} >= {"get_summary", "get_subgroup_effect", "whatif_budget", "edit_dag", "simulate_confounding"}
    for d in defs:
        assert d["input_schema"]["type"] == "object"
    text, err = tools.run_tool_json(small_engine, "get_subgroup_effect", {"filters": {"customer_segment": "low_value"}})
    assert not err and json.loads(text)["n"] > 30
    text, err = tools.run_tool_json(small_engine, "get_subgroup_effect", {"filters": {"region": "mars"}})
    assert err and "Unknown values" in json.loads(text)["error"]
    text, err = tools.run_tool_json(small_engine, "no_such_tool", {})
    assert err

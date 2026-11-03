# path: services/api/tests/test_main.py
"""HTTP surface: REST routes and the AG-UI SSE endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

from services.api.agent import CausalCopilotAgent
from services.api.main import create_app
from services.api.tests.conftest import FakeAnthropic, text_turn, tool_turn


@pytest.fixture
def client(small_engine):
    fake = FakeAnthropic([tool_turn([("c1", "get_summary", {})]), text_turn("Naive overstates the effect.")])
    agent = CausalCopilotAgent(small_engine, client=fake)
    app = create_app(engine=small_engine, agent=agent)
    with TestClient(app) as c:
        yield c


def test_rest_routes(client):
    assert client.get("/health").json()["status"] == "ok"
    s = client.get("/api/summary").json()
    assert s["n_customers"] == 1500
    assert client.get("/api/filters").json()["customer_segment"] == ["high_value", "low_value", "mid_value", "new_customer"]
    sub = client.post("/api/subgroup-effect", json={"filters": {"customer_segment": "new_customer"}}).json()
    assert sub.get("n", 0) > 30, sub
    assert client.post("/api/subgroup-effect", json={"filters": {"region": "mars"}}).status_code == 422
    assert client.get("/api/whatif", params={"budget_fraction": 0.2, "cost_per_coupon": 2}).json()["uplift_policy"]["n_targeted"] == 300
    assert client.get("/api/whatif", params={"budget_fraction": 5}).status_code == 422
    assert len(client.get("/api/targeting", params={"budget_fraction": 0.3}).json()["strategies"]) == 4
    assert "loyalty_score" in client.get("/api/dag").json()["backdoor_variables"]
    edited = client.post("/api/dag/edit", json={"add_edges": [["region", "treatment"], ["region", "outcome"]]}).json()
    assert "region" in edited["backdoor_variables"]
    sim = client.get("/api/simulate", params={"confounding_strength": 3, "n_samples": 1000}).json()
    assert {"naive_ate", "aipw_ate", "true_ate"} <= set(sim)
    assert len(client.get("/api/refutation").json()["summary"]) == 6
    assert len(client.get("/api/calibration").json()["deciles"]) == 10
    assert client.get("/api/segments").json()["segments"]
    assert len(client.get("/api/top-customers", params={"limit": 3}).json()["customers"]) == 3
    assert any(t["name"] == "edit_dag" for t in client.get("/api/tools").json()["tools"])


def test_agent_endpoint_streams_ag_ui_events(client):
    body = {
        "threadId": "thread-1",
        "runId": "run-1",
        "state": {},
        "messages": [{"id": "u1", "role": "user", "content": "What did the analysis find?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    with client.stream("POST", "/agent", json=body, headers={"accept": "text/event-stream"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())

    events = [json.loads(line[len("data: "):]) for line in raw.split("\n") if line.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert "TOOL_CALL_START" in types and "TOOL_CALL_RESULT" in types and "TEXT_MESSAGE_CONTENT" in types
    start = next(e for e in events if e["type"] == "TOOL_CALL_START")
    assert start["toolCallName"] == "get_summary" and "parentMessageId" in start
    result = next(e for e in events if e["type"] == "TOOL_CALL_RESULT")
    assert json.loads(result["content"])["n_customers"] == 1500

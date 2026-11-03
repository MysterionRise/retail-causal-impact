# path: services/api/tests/test_agent.py
"""AG-UI agent loop: event sequences, tool execution, frontend tools, history conversion."""

import json

import pytest
from ag_ui.core import EventType, RunAgentInput, Tool

from services.api.agent import convert_messages, frontend_tool_definitions
from services.api.tests.conftest import text_turn, tool_turn


def _input(messages, tools=None, state=None):
    return RunAgentInput.model_validate(
        {
            "threadId": "t1",
            "runId": "r1",
            "state": state or {},
            "messages": messages,
            "tools": tools or [],
            "context": [],
            "forwardedProps": {},
        }
    )


async def _collect(agent, run_input):
    return [ev async for ev in agent.run(run_input)]


def _types(events):
    return [e.type for e in events]


@pytest.mark.asyncio
async def test_plain_text_turn_streams_tokens(make_agent):
    agent, client = make_agent([text_turn("Hello| there")])
    events = await _collect(agent, _input([{"id": "u1", "role": "user", "content": "hi"}]))

    assert _types(events) == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.RUN_FINISHED,
    ]
    assert "".join(e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT) == "Hello there"
    req = client.requests[0]
    assert req["model"] == agent.model
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert any(t["name"] == "get_subgroup_effect" for t in req["tools"])
    assert req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["betas"]


@pytest.mark.asyncio
async def test_backend_tool_is_executed_and_fed_back(make_agent):
    agent, client = make_agent(
        [
            tool_turn([("call_1", "get_subgroup_effect", {"filters": {"customer_segment": "new_customer"}})], lead_text="Let me check."),
            text_turn("New customers respond most."),
        ]
    )
    events = await _collect(agent, _input([{"id": "u1", "role": "user", "content": "effect for new customers?"}]))
    types = _types(events)

    assert types[:5] == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.TOOL_CALL_START,
    ]
    assert EventType.TOOL_CALL_ARGS in types and EventType.TOOL_CALL_END in types
    result = next(e for e in events if e.type == EventType.TOOL_CALL_RESULT)
    assert result.tool_call_id == "call_1"
    payload = json.loads(result.content)
    assert payload["n"] > 30 and "effect_dr" in payload
    snapshot = next(e for e in events if e.type == EventType.STATE_SNAPSHOT)
    assert snapshot.snapshot["subgroup"]["n"] == payload["n"]
    assert types[-1] == EventType.RUN_FINISHED

    # Second request carried the assistant tool_use and the tool_result back to Claude
    second = client.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["content"][-1]["type"] == "tool_use"
    assert second[-1]["role"] == "user" and second[-1]["content"][0]["tool_use_id"] == "call_1"
    assert "effect_dr" in second[-1]["content"][0]["content"]


@pytest.mark.asyncio
async def test_tool_error_is_returned_as_error_result(make_agent):
    agent, client = make_agent(
        [tool_turn([("c1", "get_subgroup_effect", {"filters": {"region": "mars"}})]), text_turn("That region does not exist.")]
    )
    events = await _collect(agent, _input([{"id": "u1", "role": "user", "content": "mars?"}]))
    result = next(e for e in events if e.type == EventType.TOOL_CALL_RESULT)
    assert "Unknown values" in json.loads(result.content)["error"]
    assert client.requests[1]["messages"][-1]["content"][0].get("is_error") is True
    assert not any(e.type == EventType.STATE_SNAPSHOT for e in events)


@pytest.mark.asyncio
async def test_frontend_tool_ends_the_run_without_executing(make_agent):
    agent, client = make_agent([tool_turn([("c1", "set_budget", {"budget_fraction": 0.3})])])
    ui_tool = Tool(name="set_budget", description="Set the dashboard budget", parameters={"type": "object", "properties": {"budget_fraction": {"type": "number"}}})
    events = await _collect(agent, _input([{"id": "u1", "role": "user", "content": "set budget to 30%"}], tools=[ui_tool.model_dump()]))
    types = _types(events)

    assert EventType.TOOL_CALL_START in types and EventType.TOOL_CALL_END in types
    assert EventType.TOOL_CALL_RESULT not in types
    assert types[-1] == EventType.RUN_FINISHED
    assert len(client.requests) == 1
    assert any(t["name"] == "set_budget" for t in client.requests[0]["tools"])


@pytest.mark.asyncio
async def test_refusal_and_max_rounds(make_agent):
    agent, _ = make_agent([text_turn("", stop_reason="refusal")])
    events = await _collect(agent, _input([{"id": "u1", "role": "user", "content": "x"}]))
    assert any("can't help" in e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT)

    looping = [tool_turn([(f"c{i}", "get_summary", {})]) for i in range(8)]
    agent, _ = make_agent(looping)
    events = await _collect(agent, _input([{"id": "u1", "role": "user", "content": "loop"}]))
    assert any("too many tool calls" in e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT)
    assert events[-1].type == EventType.RUN_FINISHED


@pytest.mark.asyncio
async def test_errors_surface_as_run_error(make_agent):
    agent, _ = make_agent([])
    events = await _collect(agent, _input([]))
    assert _types(events) == [EventType.RUN_STARTED, EventType.RUN_ERROR]


def test_convert_messages_merges_tool_results_and_system():
    history = [
        {"id": "s", "role": "system", "content": "Be brief."},
        {"id": "u1", "role": "user", "content": "q"},
        {
            "id": "a1",
            "role": "assistant",
            "content": "",
            "toolCalls": [
                {"id": "c1", "type": "function", "function": {"name": "get_summary", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "get_dag", "arguments": "{}"}},
            ],
        },
        {"id": "t1", "role": "tool", "content": "{\"a\":1}", "toolCallId": "c1"},
        {"id": "t2", "role": "tool", "content": "{\"b\":2}", "toolCallId": "c2"},
        {"id": "a2", "role": "assistant", "content": "done"},
        {"id": "u2", "role": "user", "content": [{"type": "text", "text": "thanks"}]},
    ]
    msgs, extra = convert_messages(RunAgentInput.model_validate({"threadId": "t", "runId": "r", "state": {}, "messages": history, "tools": [], "context": [], "forwardedProps": {}}).messages)
    assert extra == ["Be brief."]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["c1", "c2"]
    assert msgs[4]["content"] == "thanks"


def test_convert_messages_pads_missing_tool_results():
    history = [
        {"id": "u1", "role": "user", "content": "q"},
        {"id": "a1", "role": "assistant", "content": "", "toolCalls": [{"id": "c1", "type": "function", "function": {"name": "set_budget", "arguments": "{\"budget_fraction\":0.3}"}}]},
        {"id": "u2", "role": "user", "content": "and now?"},
    ]
    msgs, _ = convert_messages(RunAgentInput.model_validate({"threadId": "t", "runId": "r", "state": {}, "messages": history, "tools": [], "context": [], "forwardedProps": {}}).messages)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "user"]
    assert msgs[2]["content"][0]["tool_use_id"] == "c1"
    assert msgs[1]["content"][0]["input"] == {"budget_fraction": 0.3}


def test_frontend_tool_definitions_skip_backend_names():
    ui = [
        Tool(name="get_summary", description="dupe", parameters={}),
        Tool(name="highlight_segment", description="h", parameters={"type": "object", "properties": {"segment": {"type": "string"}}}),
    ]
    defs = frontend_tool_definitions(ui)
    assert [d["name"] for d in defs] == ["highlight_segment"]
    assert defs[0]["input_schema"]["properties"]["segment"]["type"] == "string"

# path: services/api/agent.py
"""
AG-UI agent over Claude for the causal copilot.

Implements the AG-UI protocol on top of a manual Claude streaming loop:

* ``RunAgentInput`` (messages, tools, state) comes from CopilotKit's ``HttpAgent``.
* Backend tools (``tools.TOOLS``) run server-side against the ``Engine``; their
  results are streamed as ``TOOL_CALL_RESULT`` events (so the UI can render
  them as generative-UI cards) and fed back to Claude.
* Frontend tools (declared by the UI and delivered in ``input.tools``) are
  passed to Claude as well; when Claude calls one the run ends after the
  ``TOOL_CALL_*`` events so the browser can execute it. CopilotKit then
  re-runs the agent with the tool result appended to the history.
* Tools that change the dashboard publish their result as shared state via
  ``STATE_SNAPSHOT``.

The model only narrates and routes. Every number comes out of a tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ag_ui.core import (
    AssistantMessage,
    BaseEvent,
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)

from . import tools as tool_registry
from .engine import Engine

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are the causal copilot for a retail coupon-campaign analysis.

The analysis estimates the causal effect of sending a coupon on customer spend from
observational data in which coupons were targeted at loyal, high-value customers
(confounding). It uses propensity scores, inverse-propensity weighting, doubly-robust
AIPW, three CATE models (DRLearner, Causal Forest, X-Learner) evaluated on a held-out
split, a DoWhy causal graph with refutation tests, and a policy layer that compares
targeting rules by doubly-robust incremental spend.

Rules:
- Every number you state must come from a tool result in this conversation. Never
  estimate, extrapolate or round beyond what a tool returned. If you do not have a
  number, call the tool that produces it.
- Prefer one well-chosen tool call over several. Call tools in parallel when the
  user asks several independent questions.
- Explain causal concepts plainly (confounding, overlap, doubly-robust, backdoor
  set, placebo test, robustness value) in one or two sentences when they matter.
- Distinguish doubly-robust estimates (valid on real data) from "true" values, which
  exist only because this dataset is synthetic.
- When the user asks to change what the dashboard shows (budget, confounding
  strength, segment focus, graph), call the matching frontend tool if one is
  available; otherwise call the backend tool and describe the result.
- Keep answers short: lead with the number and what it means for the decision,
  then at most a few sentences of reasoning. Use a compact table for comparisons.
- Dollar amounts are per customer unless a total is explicitly requested.
"""


def _system_blocks(engine: Engine) -> list[dict]:
    """Stable system prompt (cached) plus the run's headline numbers."""
    summary = engine.summary()
    facts = {
        k: summary[k]
        for k in (
            "n_customers",
            "treatment_rate",
            "true_ate",
            "naive_ate",
            "headline_ate",
            "headline_method",
            "naive_overstatement_pct",
            "cate_method",
            "refutation_checks_passed",
            "refutation_checks_total",
            "robustness_value",
            "backdoor_variables",
        )
        if k in summary
    }
    return [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        {
            "type": "text",
            "text": "Headline numbers for the loaded analysis (JSON): " + json.dumps(facts),
        },
    ]


# ---------------------------------------------------------------------------
# AG-UI history -> Anthropic messages
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)


def convert_messages(messages: Iterable[Any]) -> tuple[list[dict], list[str]]:
    """
    Convert AG-UI messages to Anthropic ``messages``.

    Returns (messages, extra_system_texts). Tool results that follow one
    assistant turn are merged into a single user message, and an assistant
    tool call with no recorded result gets a placeholder result so the API
    accepts the history.
    """
    out: list[dict] = []
    extra_system: list[str] = []
    pending_tool_ids: list[str] = []

    def flush_pending() -> None:
        if not pending_tool_ids:
            return
        blocks = [
            {"type": "tool_result", "tool_use_id": tid, "content": "(no result was recorded for this call)"}
            for tid in pending_tool_ids
        ]
        pending_tool_ids.clear()
        out.append({"role": "user", "content": blocks})

    for msg in messages:
        role = msg.role
        if role in ("system", "developer"):
            text = _text_of(msg.content)
            if text:
                extra_system.append(text)
            continue
        if role == "user":
            flush_pending()
            text = _text_of(msg.content)
            if text.strip():
                out.append({"role": "user", "content": text})
            continue
        if role == "assistant":
            flush_pending()
            blocks: list[dict] = []
            text = _text_of(msg.content)
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for call in msg.tool_calls or []:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": call.id, "name": call.function.name, "input": args})
                pending_tool_ids.append(call.id)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            if msg.tool_call_id in pending_tool_ids:
                pending_tool_ids.remove(msg.tool_call_id)
            block = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": _text_of(msg.content) or "{}"}
            if getattr(msg, "error", None):
                block["is_error"] = True
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        # activity / reasoning messages carry no model-facing content

    flush_pending()
    # The API requires the first message to be from the user
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out, extra_system


def frontend_tool_definitions(input_tools: Iterable[Any]) -> list[dict]:
    """AG-UI tool declarations from the UI -> Anthropic tool definitions."""
    defs = []
    for t in input_tools or []:
        if t.name in tool_registry.TOOLS_BY_NAME:
            continue  # backend tool of the same name wins
        params = t.parameters if isinstance(t.parameters, dict) else {"type": "object", "properties": {}}
        defs.append({"name": t.name, "description": t.description or t.name, "input_schema": params})
    return defs


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class CausalCopilotAgent:
    """AG-UI compatible agent: ``run(input)`` yields AG-UI events."""

    def __init__(
        self,
        engine: Engine,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 8000,
        use_server_fallbacks: bool | None = None,
    ):
        self.engine = engine
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        if use_server_fallbacks is None:
            use_server_fallbacks = os.environ.get("CAUSAL_COPILOT_FALLBACKS", "1") != "0"
        self.use_server_fallbacks = use_server_fallbacks
        self._client = client
        self._system_cache: list[dict] | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _system(self) -> list[dict]:
        if self._system_cache is None:
            self._system_cache = _system_blocks(self.engine)
        return self._system_cache

    def _request_kwargs(self, system: list[dict], tools: list[dict], messages: list[dict]) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "tools": tools,
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if self.use_server_fallbacks:
            # Opus 5.5: route safety-classifier refusals to Anthropic's recommended
            # fallback model server-side. Set CAUSAL_COPILOT_FALLBACKS=0 to disable.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _open_stream(self, kwargs: dict) -> Any:
        if self.use_server_fallbacks:
            return self.client.beta.messages.stream(**kwargs)
        return self.client.messages.stream(**kwargs)

    async def run(self, run_input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        thread_id, run_id = run_input.thread_id, run_input.run_id
        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)

        state: dict = dict(run_input.state or {}) if isinstance(run_input.state, dict) else {}
        try:
            messages, extra_system = convert_messages(run_input.messages)
            if not messages:
                raise ValueError("The conversation has no user message to answer")
            system = list(self._system())
            for text in extra_system:
                system.append({"type": "text", "text": text})
            frontend_defs = frontend_tool_definitions(run_input.tools)
            frontend_names = {d["name"] for d in frontend_defs}
            tools = tool_registry.anthropic_tool_definitions() + frontend_defs

            for _round in range(MAX_TOOL_ROUNDS):
                stream_events, final = await asyncio.to_thread(self._stream_turn, system, tools, messages)
                for ev in stream_events:
                    yield ev

                if final.stop_reason == "refusal":
                    async for ev in self._say(
                        "I can't help with that request. Try asking about the campaign analysis."
                    ):
                        yield ev
                    break

                tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
                if final.stop_reason == "max_tokens" and tool_uses:
                    async for ev in self._say("The answer was cut off; please ask again more narrowly."):
                        yield ev
                    break
                if not tool_uses:
                    break

                # Execute backend tools; frontend tools are left to the browser.
                assistant_blocks = [b.model_dump() if hasattr(b, "model_dump") else b for b in final.content]
                messages.append({"role": "assistant", "content": assistant_blocks})
                results_blocks: list[dict] = []
                frontend_called = False
                state_changed = False
                for block in tool_uses:
                    if block.name in frontend_names:
                        frontend_called = True
                        continue
                    args = block.input if isinstance(block.input, dict) else {}
                    text, is_error = await asyncio.to_thread(
                        tool_registry.run_tool_json, self.engine, block.name, args
                    )
                    spec = tool_registry.TOOLS_BY_NAME.get(block.name)
                    if spec is not None and spec.state_key and not is_error:
                        state[spec.state_key] = json.loads(text)
                        state_changed = True
                    yield ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        message_id=str(uuid.uuid4()),
                        tool_call_id=block.id,
                        content=text,
                        role="tool",
                    )
                    result_block = {"type": "tool_result", "tool_use_id": block.id, "content": text}
                    if is_error:
                        result_block["is_error"] = True
                    results_blocks.append(result_block)

                if state_changed:
                    yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)

                if frontend_called:
                    # The browser executes the frontend tool(s) and CopilotKit
                    # re-runs the agent with the results appended.
                    break
                messages.append({"role": "user", "content": results_blocks})
            else:
                async for ev in self._say("I stopped after too many tool calls; please narrow the question."):
                    yield ev

            yield RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id)
        except Exception as e:  # surfaced to the UI as a run error
            logger.exception("copilot run failed")
            yield RunErrorEvent(type=EventType.RUN_ERROR, message=f"{type(e).__name__}: {e}")

    async def _say(self, text: str) -> AsyncIterator[BaseEvent]:
        mid = str(uuid.uuid4())
        yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant")
        yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=text)
        yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid)

    def _stream_turn(self, system: list[dict], tools: list[dict], messages: list[dict]) -> tuple[list[BaseEvent], Any]:
        """
        One Claude turn, streamed. Runs in a worker thread and returns the
        AG-UI events it produced plus the final message.

        Text and tool-call blocks are translated as they arrive so the UI
        sees tokens and tool arguments live.
        """
        events: list[BaseEvent] = []
        assistant_message_id = str(uuid.uuid4())
        open_text: str | None = None
        open_tool: str | None = None
        kwargs = self._request_kwargs(system, tools, messages)

        with self._open_stream(kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        open_text = assistant_message_id if open_text is None else str(uuid.uuid4())
                        events.append(TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=open_text, role="assistant"))
                    elif block.type == "tool_use":
                        open_tool = block.id
                        events.append(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=block.id,
                                tool_call_name=block.name,
                                parent_message_id=assistant_message_id,
                            )
                        )
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta" and open_text is not None:
                        events.append(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=open_text, delta=delta.text))
                    elif delta.type == "input_json_delta" and open_tool is not None and delta.partial_json:
                        events.append(ToolCallArgsEvent(type=EventType.TOOL_CALL_ARGS, tool_call_id=open_tool, delta=delta.partial_json))
                elif etype == "content_block_stop":
                    if open_text is not None and open_tool is None:
                        events.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=open_text))
                        open_text = None
                    elif open_tool is not None:
                        events.append(ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=open_tool))
                        open_tool = None
            final = stream.get_final_message()
        return events, final


# Re-exported for tests / type hints
__all__ = [
    "CausalCopilotAgent",
    "convert_messages",
    "frontend_tool_definitions",
    "AssistantMessage",
    "ToolMessage",
    "UserMessage",
]

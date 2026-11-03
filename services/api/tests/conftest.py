# path: services/api/tests/conftest.py
"""
Shared fixtures: a small but real engine built by running the pipeline's
estimation stage on 1,500 synthetic customers with tiny models, and a fake
Anthropic client that replays scripted streaming turns.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coupon_causal import data, utils  # noqa: E402
from scripts.run_pipeline import run_estimation_stage  # noqa: E402
from services.api.agent import CausalCopilotAgent  # noqa: E402
from services.api.engine import Engine  # noqa: E402

SMALL_OVERRIDES = {
    "synthetic": {"n_samples": 1500},
    "propensity": {"models": {"gbm": {"n_estimators": 20, "max_depth": 2}}},
    "ate": {"bootstrap_iterations": 20, "rf_outcome_model": {"n_estimators": 10, "max_depth": 4}},
    "cate": {
        "cross_fitting_folds": 2,
        "dr_learner": {
            "model_regression": {"n_estimators": 20, "max_depth": 3},
            "model_propensity": {"n_estimators": 20, "max_depth": 2},
            "model_final": {"n_estimators": 30, "max_depth": 4},
        },
        "causal_forest": {"n_trees": 40, "model_y": {"n_estimators": 20, "max_depth": 3}, "model_t": {"n_estimators": 20, "max_depth": 2}},
        "x_learner": {"n_estimators": 20, "max_depth": 4},
    },
    "refutation": {"n_simulations": 2, "n_placebo_permutations": 3, "placebo_outcome_bootstrap": 10},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


@pytest.fixture(scope="session")
def small_engine(tmp_path_factory) -> Engine:
    warnings.filterwarnings("ignore")
    logging.getLogger("coupon_causal").setLevel(logging.ERROR)
    logging.getLogger("dowhy").setLevel(logging.ERROR)
    config = _merge(utils.load_config(str(ROOT / "config" / "default.yaml")), SMALL_OVERRIDES)
    workdir = tmp_path_factory.mktemp("artifacts")
    config["paths"]["models"] = str(workdir / "models")
    df, _ = data.generate_synthetic_coupon_data(n_samples=config["synthetic"]["n_samples"], random_state=7)
    results = run_estimation_stage(config, df)
    return Engine.from_results(config, df, results)


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


class FakeBlock(SimpleNamespace):
    def model_dump(self):
        d = dict(self.__dict__)
        return d


class FakeStream:
    """Context manager that replays scripted raw stream events."""

    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


def text_turn(text: str, stop_reason: str = "end_turn") -> FakeStream:
    events = [
        SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        *[SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text=chunk)) for chunk in text.split("|")],
        SimpleNamespace(type="content_block_stop", index=0),
    ]
    final = SimpleNamespace(
        stop_reason=stop_reason,
        content=[FakeBlock(type="text", text=text.replace("|", ""))],
    )
    return FakeStream(events, final)


def tool_turn(calls: list[tuple[str, str, dict]], lead_text: str | None = None) -> FakeStream:
    """A turn that (optionally) says something, then calls tools (id, name, input)."""
    import json

    events = []
    content = []
    idx = 0
    if lead_text:
        events += [
            SimpleNamespace(type="content_block_start", index=idx, content_block=SimpleNamespace(type="text")),
            SimpleNamespace(type="content_block_delta", index=idx, delta=SimpleNamespace(type="text_delta", text=lead_text)),
            SimpleNamespace(type="content_block_stop", index=idx),
        ]
        content.append(FakeBlock(type="text", text=lead_text))
        idx += 1
    for call_id, name, args in calls:
        events += [
            SimpleNamespace(type="content_block_start", index=idx, content_block=SimpleNamespace(type="tool_use", id=call_id, name=name)),
            SimpleNamespace(type="content_block_delta", index=idx, delta=SimpleNamespace(type="input_json_delta", partial_json=json.dumps(args))),
            SimpleNamespace(type="content_block_stop", index=idx),
        ]
        content.append(FakeBlock(type="tool_use", id=call_id, name=name, input=args))
        idx += 1
    final = SimpleNamespace(stop_reason="tool_use", content=content)
    return FakeStream(events, final)


class FakeMessages:
    def __init__(self, turns, log):
        self._turns = list(turns)
        self._log = log

    def stream(self, **kwargs):
        self._log.append(kwargs)
        if not self._turns:
            raise AssertionError("fake client ran out of scripted turns")
        return self._turns.pop(0)


class FakeAnthropic:
    """Scripted replacement for anthropic.Anthropic() (beta and non-beta paths)."""

    def __init__(self, turns):
        self.requests: list[dict] = []
        self.messages = FakeMessages(turns, self.requests)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def make_agent(small_engine):
    def _make(turns, **kwargs):
        client = FakeAnthropic(turns)
        agent = CausalCopilotAgent(small_engine, client=client, **kwargs)
        return agent, client

    return _make

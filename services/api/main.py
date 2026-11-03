# path: services/api/main.py
"""
FastAPI service: REST access to the causal engine plus the AG-UI copilot endpoint.

Run locally with::

    make all            # produces the artifacts the engine loads
    make api            # uvicorn on :8000

Environment:
    CAUSAL_ROOT                 repo root holding config/, data/, models/ (default: this checkout)
    CAUSAL_CONFIG               config file (default: config/default.yaml)
    ANTHROPIC_API_KEY           required for /agent
    ANTHROPIC_MODEL             default claude-opus-5-5
    CAUSAL_COPILOT_FALLBACKS    "0" disables the server-side refusal fallback beta
    CORS_ORIGINS                comma-separated allowed origins (default: http://localhost:3000)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import tools as tool_registry
from .agent import CausalCopilotAgent
from .engine import ROOT, ArtifactsMissingError, Engine

logger = logging.getLogger("causal_api")


class SubgroupRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)


class DagEdit(BaseModel):
    add_edges: list[list[str]] = Field(default_factory=list)
    remove_edges: list[list[str]] = Field(default_factory=list)
    base_edges: list[list[str]] | None = None


def create_app(engine: Engine | None = None, agent: CausalCopilotAgent | None = None) -> FastAPI:
    """Build the app. Tests pass a prebuilt engine/agent; production loads from disk."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            root = Path(os.environ.get("CAUSAL_ROOT", ROOT))
            config_path = os.environ.get("CAUSAL_CONFIG")
            try:
                app.state.engine = Engine.load(root, config_path)
            except ArtifactsMissingError as e:
                logger.error(str(e))
                raise
        app.state.agent = agent or CausalCopilotAgent(app.state.engine)
        logger.info("causal engine ready: %d customers", len(app.state.engine.df))
        yield

    app = FastAPI(title="Retail Causal Impact API", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def eng(request: Request) -> Engine:
        return request.app.state.engine

    def guard(fn, *args, **kwargs) -> Any:
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except ArtifactsMissingError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------
    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "n_customers": len(eng(request).df)}

    @app.get("/api/summary")
    def summary(request: Request) -> dict:
        return eng(request).summary()

    @app.get("/api/filters")
    def filters(request: Request) -> dict:
        return eng(request).filter_options()

    @app.post("/api/subgroup-effect")
    def subgroup_effect(body: SubgroupRequest, request: Request) -> dict:
        return guard(eng(request).subgroup_effect, body.filters)

    @app.get("/api/whatif")
    def whatif(request: Request, budget_fraction: float = 0.2, cost_per_coupon: float = 1.0) -> dict:
        return guard(eng(request).whatif_budget, budget_fraction, cost_per_coupon)

    @app.get("/api/targeting")
    def targeting(request: Request, budget_fraction: float = 0.2) -> dict:
        return guard(eng(request).compare_targeting, budget_fraction)

    @app.get("/api/dag")
    def dag(request: Request) -> dict:
        return eng(request).dag()

    @app.post("/api/dag/edit")
    def dag_edit(body: DagEdit, request: Request) -> dict:
        return guard(eng(request).edit_dag, body.add_edges, body.remove_edges, body.base_edges)

    @app.get("/api/simulate")
    def simulate(request: Request, confounding_strength: float = 2.0, n_samples: int = 5000) -> dict:
        return guard(eng(request).simulate_confounding, confounding_strength, n_samples)

    @app.get("/api/refutation")
    def refutation(request: Request) -> dict:
        return guard(eng(request).refutation)

    @app.get("/api/calibration")
    def calibration(request: Request) -> dict:
        return eng(request).calibration()

    @app.get("/api/segments")
    def segments(request: Request) -> dict:
        return eng(request).segments()

    @app.get("/api/top-customers")
    def top_customers(request: Request, budget_fraction: float = 0.2, limit: int = 20) -> dict:
        return guard(eng(request).top_customers, budget_fraction, limit)

    @app.get("/api/tools")
    def list_tools() -> dict:
        return {"tools": tool_registry.anthropic_tool_definitions()}

    # ------------------------------------------------------------------
    # AG-UI agent endpoint (CopilotKit HttpAgent posts RunAgentInput here)
    # ------------------------------------------------------------------
    @app.post("/agent")
    async def agent_endpoint(run_input: RunAgentInput, request: Request) -> StreamingResponse:
        encoder = EventEncoder(accept=request.headers.get("accept"))
        copilot: CausalCopilotAgent = request.app.state.agent

        async def event_stream():
            async for event in copilot.run(run_input):
                yield encoder.encode(event)

        return StreamingResponse(event_stream(), media_type=encoder.get_content_type())

    return app


app = create_app()

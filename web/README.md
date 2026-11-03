# Causal copilot front end

Next.js 16 + CopilotKit (v2 API) dashboard for the retail coupon causal-impact analysis.
The copilot is an AG-UI agent served by the Python API in `../services/api`; this app
hosts the CopilotKit runtime that proxies to it and renders the dashboard.

```
app/api/copilotkit/[[...path]]/route.ts   CopilotKit runtime -> HttpAgent(CAUSAL_AGENT_URL)
app/providers.tsx                          <CopilotKit runtimeUrl="/api/copilotkit"> + dashboard store
components/Dashboard.tsx                   panels fed by the REST API (NEXT_PUBLIC_CAUSAL_API_URL)
components/copilot/CopilotPanel.tsx        frontend tools, generative-UI renderers, suggestions, sidebar
components/copilot/cards.tsx               cards rendered in chat for each backend tool call
components/charts.tsx, DagView.tsx         Recharts views and the hand-laid causal graph
lib/api.ts, lib/types.ts, lib/store.tsx    API client, payload types, shared dashboard state
```

## Run

```bash
# from the repository root, with the API running on :8000 (make api)
cd web && npm install && npm run dev        # http://localhost:3000
```

Environment (see `../.env.example`):

| Variable | Used by | Default |
|---|---|---|
| `CAUSAL_AGENT_URL` | runtime route (server) | `http://localhost:8000/agent` |
| `NEXT_PUBLIC_CAUSAL_API_URL` | browser fetches | `http://localhost:8000` |
| `COPILOTKIT_TELEMETRY_DISABLED` | runtime | unset |

## Checks

```bash
npx tsc --noEmit -p tsconfig.json
npx eslint .
npm run build
```

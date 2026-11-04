// path: web/app/api/copilotkit/[[...path]]/route.ts
// CopilotKit runtime (v2) for the Next.js app router.
//
// The runtime proxies the browser to the Python AG-UI agent served by
// services/api (POST /agent). It is registered as "default", the id the
// CopilotKit hooks and chat components use when none is given.

import { HttpAgent } from "@ag-ui/client";
import { CopilotRuntime, createCopilotEndpoint } from "@copilotkit/runtime/v2";
import { handle } from "hono/vercel";

const AGENT_URL = process.env.CAUSAL_AGENT_URL ?? "http://localhost:8000/agent";

const runtime = new CopilotRuntime({
  // Single-user demo: every browser shares the thread list. Pass identifyUser
  // instead when deploying for several people.
  threads: { access: "open" },
  agents: {
    default: new HttpAgent({
      url: AGENT_URL,
      description: "Causal copilot over the retail coupon-campaign analysis engine",
    }),
  },
});

const app = createCopilotEndpoint({ runtime, basePath: "/api/copilotkit" });
const handler = handle(app);

export const GET = handler;
export const POST = handler;
export const OPTIONS = handler;
export const DELETE = handler;
export const PATCH = handler;

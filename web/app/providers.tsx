// path: web/app/providers.tsx
"use client";

import { CopilotKit } from "@copilotkit/react-core/v2";
import type { ReactNode } from "react";
import { DashboardProvider } from "@/lib/store";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <DashboardProvider>{children}</DashboardProvider>
    </CopilotKit>
  );
}

// path: web/app/page.tsx
import { Dashboard } from "@/components/Dashboard";
import { CopilotPanel } from "@/components/copilot/CopilotPanel";

export default function Home() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-6 md:px-6">
      <header className="mb-5">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">Retail coupon campaign · causal impact</h1>
        <p className="mt-1 max-w-3xl text-sm text-zinc-600 dark:text-zinc-400">
          Coupons went mostly to loyal, high-value customers who spend more anyway. A plain treated-vs-control comparison credits the coupon
          for spend that would have happened regardless. The panels below remove that bias, stress-test the estimate, and answer the question
          that matters for the budget: whose spend does a coupon actually change? Ask the copilot anything; it only reports numbers it gets
          from the causal engine.
        </p>
      </header>
      <Dashboard />
      <CopilotPanel />
    </main>
  );
}

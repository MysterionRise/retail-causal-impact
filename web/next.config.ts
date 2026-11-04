import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output for the Docker image (web/Dockerfile)
  output: "standalone",
  // The dashboard is fully client-driven (live API data, CopilotKit threads),
  // so static prerendering of "/" is neither possible nor useful.
  cacheComponents: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;

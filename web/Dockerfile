# path: web/Dockerfile
# Next.js + CopilotKit front end. Build context is ./web.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Public URL of the API as seen from the browser (baked at build time)
ARG NEXT_PUBLIC_CAUSAL_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_CAUSAL_API_URL=$NEXT_PUBLIC_CAUSAL_API_URL
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]

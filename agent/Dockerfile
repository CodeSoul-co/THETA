FROM node:22-bookworm-slim AS builder

RUN npm install --global pnpm@10.15.0
WORKDIR /app/agent
COPY agent/package.json agent/pnpm-lock.yaml agent/tsconfig.json ./
RUN pnpm install --frozen-lockfile
COPY agent/src ./src
COPY agent/cli ./cli
COPY agent/web ./web
RUN pnpm build && pnpm prune --prod

FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    THETA_PROJECT_ROOT=/app \
    THETA_AGENT_HOME=/data \
    THETA_AGENT_API_PORT=4318

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3 python3-pip ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

COPY agent/requirements.service.txt /tmp/requirements.service.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.service.txt \
    && rm /tmp/requirements.service.txt

WORKDIR /app/agent
COPY --from=builder /app/agent/dist ./dist
COPY --from=builder /app/agent/node_modules ./node_modules
COPY --from=builder /app/agent/package.json ./package.json
COPY agent/workers ./workers
COPY agent/knowledge ./knowledge
COPY agent/skills ./skills
COPY src/models /app/src/models
COPY trainning/worker /app/trainning/worker

RUN mkdir -p /data /tmp/theta-agent && chown -R node:node /app /data /tmp/theta-agent
USER node
EXPOSE 4318
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:4318/healthz',{headers:{host:process.env.THETA_AGENT_HEALTH_HOST||'127.0.0.1:4318'}}).then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["node", "--disable-warning=ExperimentalWarning", "dist/web/server.js"]

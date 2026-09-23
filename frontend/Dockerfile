FROM node:22-alpine AS builder

WORKDIR /app

ARG NEXT_PUBLIC_LOCAL_NO_AUTH=false
ARG NEXT_PUBLIC_LOCAL_USER_ID=00000000-0000-4000-8000-000000000001
ENV NEXT_PUBLIC_LOCAL_NO_AUTH=${NEXT_PUBLIC_LOCAL_NO_AUTH}
ENV NEXT_PUBLIC_LOCAL_USER_ID=${NEXT_PUBLIC_LOCAL_USER_ID}

RUN corepack enable && corepack prepare pnpm@11.19.0 --activate
ENV NEXT_TELEMETRY_DISABLED=1

# 复制依赖文件
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml ./

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制源代码
COPY . .

# 构建应用
RUN pnpm build

FROM node:22-alpine AS runner

ARG THETA_RELEASE_REVISION=development
ARG THETA_RELEASE_SOURCE=https://github.com/CodeSoul-co/-THETA-

LABEL org.opencontainers.image.source="${THETA_RELEASE_SOURCE}" \
      org.opencontainers.image.revision="${THETA_RELEASE_REVISION}" \
      org.opencontainers.image.version="${THETA_RELEASE_REVISION}"

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# 创建非 root 用户
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]

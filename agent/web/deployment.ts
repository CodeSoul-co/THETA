import { timingSafeEqual } from 'node:crypto';
import type { IncomingMessage } from 'node:http';

export type AgentServiceMode = 'local' | 'deployed';

export interface AgentPrincipal {
  id: string;
  username?: string;
  role?: string;
}

export interface AgentAuthentication {
  principal: AgentPrincipal;
  csrfToken?: string;
}

export interface AgentServerOptions {
  /** Per-launch desktop secret, supplied only by the local application host. */
  localToken?: string;
  /** Trusted server-side local workspace; never supplied by an HTTP client. */
  manualHome?: string;
  mode?: AgentServiceMode;
  allowedOrigins?: string[];
  trustedHosts?: string[];
  authenticate?: (request: IncomingMessage) => Promise<AgentAuthentication>;
  requireCsrf?: boolean;
}

export class AgentAccessError extends Error {
  constructor(readonly status: number, readonly code: string, message: string) {
    super(message);
  }
}

const values = (raw: string | undefined): string[] =>
  (raw ?? '').split(',').map((value) => value.trim()).filter(Boolean);

const normalizedHost = (value: string): string => value.trim().toLowerCase().replace(/\.$/u, '');

export const loadAgentServerOptions = (
  env: Readonly<Record<string, string | undefined>> = process.env,
  fetchImpl: typeof fetch = fetch,
): AgentServerOptions => {
  const mode = env.THETA_AGENT_SERVICE_MODE === 'deployed' ? 'deployed' : 'local';
  if (mode === 'local') return { mode };
  const businessApiUrl = env.THETA_BUSINESS_API_URL?.trim();
  const allowedOrigins = values(env.THETA_AGENT_ALLOWED_ORIGINS);
  const trustedHosts = values(env.THETA_AGENT_TRUSTED_HOSTS).map(normalizedHost);
  if (!businessApiUrl) throw new Error('deployed 模式必须配置 THETA_BUSINESS_API_URL。');
  if (!allowedOrigins.length) throw new Error('deployed 模式必须配置 THETA_AGENT_ALLOWED_ORIGINS。');
  if (!trustedHosts.length) throw new Error('deployed 模式必须配置 THETA_AGENT_TRUSTED_HOSTS。');
  return {
    mode,
    allowedOrigins,
    trustedHosts,
    requireCsrf: true,
    authenticate: businessApiAuthenticator(businessApiUrl, fetchImpl),
  };
};

const forwardedCookie = (request: IncomingMessage): string | undefined => {
  if (request.headers.cookie) return request.headers.cookie;
  const authorization = request.headers.authorization;
  const match = /^Bearer\s+(.+)$/iu.exec(authorization ?? '');
  return match ? `__Host-theta_session=${match[1]}` : undefined;
};

/** Verify the browser session against the existing Business API. Credentials are
 * forwarded only to /auth/me and are never persisted by the Agent service. */
export const businessApiAuthenticator = (
  baseUrl: string,
  fetchImpl: typeof fetch = fetch,
): ((request: IncomingMessage) => Promise<AgentAuthentication>) => {
  const endpoint = new URL('/api/v1/auth/me', baseUrl).toString();
  return async (request) => {
    const cookie = forwardedCookie(request);
    if (!cookie) throw new AgentAccessError(401, 'unauthorized', '请先登录。');
    let response: Response;
    try {
      response = await fetchImpl(endpoint, {
        method: 'GET',
        redirect: 'error',
        signal: AbortSignal.timeout(15_000),
        headers: { accept: 'application/json', cookie },
      });
    } catch {
      throw new AgentAccessError(503, 'auth_unavailable', '认证服务暂时不可用。');
    }
    if (response.status === 401) throw new AgentAccessError(401, 'unauthorized', '登录状态已失效。');
    if (!response.ok) throw new AgentAccessError(503, 'auth_unavailable', '认证服务暂时不可用。');
    const body = await response.json().catch(() => null) as {
      user?: { id?: unknown; username?: unknown; role?: unknown };
      csrf_token?: unknown;
    } | null;
    const id = typeof body?.user?.id === 'string' ? body.user.id : '';
    if (!id) throw new AgentAccessError(503, 'auth_invalid_response', '认证服务返回了无效用户信息。');
    return {
      principal: {
        id,
        ...(typeof body?.user?.username === 'string' ? { username: body.user.username } : {}),
        ...(typeof body?.user?.role === 'string' ? { role: body.user.role } : {}),
      },
      ...(typeof body?.csrf_token === 'string' ? { csrfToken: body.csrf_token } : {}),
    };
  };
};

export const assertRequestOrigin = (request: IncomingMessage, options: AgentServerOptions): void => {
  const mode = options.mode ?? 'local';
  const host = normalizedHost(request.headers.host ?? '');
  const origin = request.headers.origin;
  if (mode === 'local') {
    if (options.localToken) {
      const supplied = request.headers['x-theta-desktop-token'];
      if (typeof supplied !== 'string' || !csrfEqual(options.localToken, supplied)) {
        throw new AgentAccessError(403, 'desktop_token_required', '仅允许当前桌面应用访问。');
      }
    }
    if (!/^(127\.0\.0\.1|localhost):\d+$/u.test(host)) {
      throw new AgentAccessError(403, 'host_rejected', '本地 Agent API 仅允许 localhost。');
    }
    if (origin && !/^http:\/\/(127\.0\.0\.1|localhost):(?:4320|4318)$/u.test(origin)) {
      throw new AgentAccessError(403, 'origin_rejected', '来源不允许。');
    }
    return;
  }
  const hostWithoutPort = host.replace(/:\d+$/u, '');
  if (!options.trustedHosts?.some((candidate) => candidate === host || candidate === hostWithoutPort)) {
    throw new AgentAccessError(403, 'host_rejected', '请求 Host 不在 Agent 服务允许列表中。');
  }
  if (origin && !options.allowedOrigins?.includes(origin)) {
    throw new AgentAccessError(403, 'origin_rejected', '请求 Origin 不在 Agent 服务允许列表中。');
  }
};

const csrfEqual = (expected: string, actual: string): boolean => {
  const left = Buffer.from(expected);
  const right = Buffer.from(actual);
  return left.length === right.length && timingSafeEqual(left, right);
};

export const authenticateRequest = async (
  request: IncomingMessage,
  options: AgentServerOptions,
): Promise<AgentPrincipal> => {
  if ((options.mode ?? 'local') === 'local') return { id: 'local-user', username: 'local' };
  if (!options.authenticate) throw new AgentAccessError(503, 'auth_not_configured', 'Agent 认证尚未配置。');
  const authenticated = await options.authenticate(request);
  const write = !['GET', 'HEAD', 'OPTIONS'].includes(request.method ?? 'GET');
  if (write && options.requireCsrf !== false) {
    const actual = String(request.headers['x-csrf-token'] ?? '');
    if (!authenticated.csrfToken || !actual || !csrfEqual(authenticated.csrfToken, actual)) {
      throw new AgentAccessError(403, 'csrf_failed', 'CSRF Token 无效，请刷新登录状态后重试。');
    }
  }
  return authenticated.principal;
};

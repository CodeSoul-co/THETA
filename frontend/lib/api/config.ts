/** 工作台数据始终通过同源代理访问本机服务。模型 API 由用户在设置中单独配置。 */
import { toast } from 'sonner'
import { ApiError } from './api-error'

export const API_BASE = '/api/backend';
export const AGENT_BASE = API_BASE;

/** 将 FastAPI 的 detail（可能是字符串或对象数组）转为可读错误信息 */
function formatErrorDetail(detail: unknown, fallback: string): string {
  if (detail == null) return String(fallback);
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d: any) => {
      if (typeof d === 'string') return d;
      if (d?.msg) return d.msg;
      return JSON.stringify(d);
    });
    return parts.filter(Boolean).join('; ') || String(fallback);
  }
  if (typeof detail === 'object' && detail !== null && 'msg' in detail) {
    return (detail as { msg: string }).msg;
  }
  return String(fallback);
}

/**
 * 通用 fetch 封装。桌面认证由应用和同源代理完成。
 */
export async function apiFetch<T>(
  base: string,
  endpoint: string,
  options?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const { timeoutMs = 30_000, ...fetchOptions } = options ?? {};
  const url = `${base}${endpoint}`;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(url, {
      ...fetchOptions,
      headers,
      credentials: fetchOptions.credentials ?? 'same-origin',
      signal: controller.signal,
    });
  } catch (err: any) {
    clearTimeout(timer);
    const msg = err?.name === 'AbortError'
      ? '请求超时，请检查后端服务是否已启动。'
      : '无法连接到后端服务，请检查网络和后端服务状态。';
    if (typeof window !== 'undefined') toast.error(msg);
    throw new Error(msg);
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    if (response.status >= 500 && typeof window !== 'undefined') toast.error('本地服务异常，请稍后重试');
    const body = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    const detail = body.detail ?? body.error?.message;
    const msg = formatErrorDetail(detail, `HTTP ${response.status}`);
    throw new ApiError(response.status === 401 ? '本地服务验证失败，请重新启动 THETA' : msg, response.status);
  }

  // 204 No Content 无响应体，直接返回
  if (response.status === 204) return null as T;
  return response.json();
}

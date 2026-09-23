
import { OPEN_SOURCE_EDITION } from '@/lib/edition'
import { allowLocalSession, isLocalRequestOrigin, isManualWorkspacePath } from '@/lib/local-development'
import type { NextRequest } from 'next/server'
import { NextResponse } from 'next/server'

interface RouteContext {
  params: Promise<{ path: string[] }>
}

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 300
const SESSION_COOKIE = 'theta_session'

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function HEAD(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204 })
}

async function proxy(request: NextRequest, context: RouteContext) {
  const { path } = await context.params
  const localBackend = process.env.THETA_MANUAL_LOCAL_API_URL?.trim() || 'http://127.0.0.1:4321'
  const localDevelopment = allowLocalSession(request.url, localBackend, process.env.NODE_ENV, process.env.THETA_LOCAL_AUTH_ENABLED)
  if (OPEN_SOURCE_EDITION && (!localDevelopment || !isManualWorkspacePath(path))) {
    return NextResponse.json({ detail: 'Only local manual-workspace APIs are available in the open-source edition.' }, { status: 404 })
  }
  if (localDevelopment) {
    if (!isLocalRequestOrigin(request.nextUrl.protocol, request.headers.get('host'), request.headers.get('origin'), request.headers.get('sec-fetch-site'))) {
      return NextResponse.json({ detail: '本地开发接口不接受跨站请求。' }, { status: 403 })
    }
  }

  const isAuthRequest = path[0] === 'api' && path[1] === 'auth'
  const isAgentRequest = path[0] === 'api' && ['auth', 'admin'].includes(path[1] ?? '')
  const baseUrl = localDevelopment ? localBackend : isAgentRequest
    ? process.env.THETA_AGENT_API_URL?.trim()
      || (process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:4318' : undefined)
    : process.env.THETA_LEGACY_API_URL?.trim()
      || 'https://theta-backend-nu.vercel.app'
  if (!baseUrl) {
    return NextResponse.json(
      {
        detail: isAgentRequest
          ? 'THETA_AGENT_API_URL 未配置，无法使用 MySQL 用户服务。'
          : 'THETA_LEGACY_API_URL 未配置。',
        code: 'THETA_BACKEND_NOT_CONFIGURED',
      },
      { status: 503, headers: { 'Cache-Control': 'no-store' } },
    )
  }
  const target = new URL(
    `/${path.map(encodeURIComponent).join('/')}`,
    ensureTrailingSlash(baseUrl),
  )
  target.search = request.nextUrl.search

  const headers = new Headers()
  for (const name of [
    'accept',
    'authorization',
    'content-type',
    'range',
    'user-agent',
    'x-forwarded-for',
    'x-forwarded-proto',
  ]) {
    const value = request.headers.get(name)
    if (value && !(OPEN_SOURCE_EDITION && name === 'authorization')) headers.set(name, value)
  }
  const sessionToken = localDevelopment ? undefined : request.cookies.get(SESSION_COOKIE)?.value
  if (sessionToken) {
    // 已登录会话使用 HttpOnly cookie 中的真实后端 JWT。
    headers.set('authorization', `Bearer ${sessionToken}`)
  }

  try {
    const upstream = await fetch(target, streamingRequestInit(request, headers))
    if (!localDevelopment && isAuthRequest && path[2] === 'login' && upstream.ok) {
      const payload = await upstream.json() as Record<string, unknown>
      const token = typeof payload.access_token === 'string' ? payload.access_token : ''
      if (!token) {
        return NextResponse.json({ detail: '登录服务未返回有效会话。' }, { status: 502 })
      }
      const expiresIn = typeof payload.expires_in === 'number' ? payload.expires_in : 86_400
      const result = NextResponse.json({ ...payload, access_token: 'http-only-cookie' }, {
        status: upstream.status,
        headers: responseHeaders(upstream),
      })
      setSessionCookie(result, token, expiresIn, request)
      return result
    }
    const result = new NextResponse(upstream.body, {
      status: upstream.status,
      headers: responseHeaders(upstream),
    })
    if (isAuthRequest && path[2] === 'logout') clearSessionCookie(result, request)
    return result
  } catch {
    return NextResponse.json(
      {
        detail: '无法连接已部署的 THETA 后端。',
        code: 'THETA_BACKEND_UNAVAILABLE',
      },
      { status: 502, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}

type StreamingRequestInit = RequestInit & { duplex?: 'half' }

function streamingRequestInit(request: NextRequest, headers: Headers): StreamingRequestInit {
  const hasBody = !['GET', 'HEAD'].includes(request.method) && request.body !== null
  return {
    method: request.method,
    cache: 'no-store',
    headers,
    body: hasBody ? request.body : undefined,
    signal: request.signal,
    redirect: 'manual',
    ...(hasBody ? { duplex: 'half' as const } : {}),
  }
}

function responseHeaders(response: Response) {
  const headers = new Headers({ 'Cache-Control': 'no-store' })
  for (const name of [
    'accept-ranges',
    'content-disposition',
    'content-range',
    'content-type',
    'location',
    'content-security-policy',
    'x-content-type-options',
    'set-cookie',
  ]) {
    const value = response.headers.get(name)
    if (value) headers.set(name, value)
  }
  return headers
}

function ensureTrailingSlash(value: string) {
  return value.endsWith('/') ? value : `${value}/`
}

function setSessionCookie(
  response: NextResponse,
  token: string,
  maxAge: number,
  request: NextRequest,
) {
  response.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: isHttpsRequest(request),
    sameSite: 'lax',
    path: '/',
    maxAge,
  })
}

function clearSessionCookie(response: NextResponse, request: NextRequest) {
  response.cookies.set(SESSION_COOKIE, '', {
    httpOnly: true,
    secure: isHttpsRequest(request),
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
  })
}

function isHttpsRequest(request: NextRequest) {
  return request.nextUrl.protocol === 'https:'
    || request.headers.get('x-forwarded-proto')?.split(',')[0]?.trim() === 'https'
}

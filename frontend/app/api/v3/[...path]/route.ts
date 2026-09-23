
import { allowLocalSession, isLocalRequestOrigin } from '@/lib/local-development'
import type { NextRequest } from 'next/server'
import { NextResponse } from 'next/server'

interface RouteContext {
  params: Promise<{ path: string[] }>
}

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 300

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context)
}

export async function POST(request: NextRequest, context: RouteContext) {
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
  const configuredBaseUrl = process.env.THETA_AGENT_API_URL?.trim() || 'http://127.0.0.1:4318'
  if (!allowLocalSession(request.url, configuredBaseUrl, process.env.NODE_ENV, process.env.THETA_LOCAL_AUTH_ENABLED, process.env.THETA_DESKTOP_TOKEN)) {
    return unavailable('无法使用本地 THETA Agent，请从本机工作台或 THETA 应用启动。')
  }
  if (!isLocalRequestOrigin(request.nextUrl.protocol, request.headers.get('host'), request.headers.get('origin'), request.headers.get('sec-fetch-site'))) {
    return new NextResponse('本地接口不接受跨站请求。', { status: 403 })
  }

  const { path } = await context.params
  const target = new URL(
    `/api/v3/${path.map(encodeURIComponent).join('/')}`,
    ensureTrailingSlash(configuredBaseUrl),
  )
  target.search = request.nextUrl.search

  const headers = new Headers()
  if (process.env.THETA_DESKTOP_TOKEN) headers.set('x-theta-desktop-token', process.env.THETA_DESKTOP_TOKEN)
  for (const name of ['accept', 'content-type', 'last-event-id', 'user-agent', 'x-forwarded-for', 'x-forwarded-proto']) {
    const value = request.headers.get(name)
    if (value) headers.set(name, value)
  }
  try {
    const response = await fetch(target, streamingRequestInit(request, headers))
    return new NextResponse(response.body, {
      status: response.status,
      headers: responseHeaders(response),
    })
  } catch {
    return unavailable('无法连接本地 THETA Agent，请重新启动工作台或应用。')
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

function unavailable(message: string) {
  return NextResponse.json(
    {
      ok: false,
      error: {
        code: 'THETA_AGENT_API_UNAVAILABLE',
        message,
      },
    },
    { status: 502, headers: { 'Cache-Control': 'no-store' } },
  )
}

function responseHeaders(response: Response) {
  const headers = new Headers({ 'Cache-Control': 'no-store' })
  for (const name of ['content-type', 'x-request-id', 'content-security-policy', 'x-content-type-options', 'content-disposition']) {
    const value = response.headers.get(name)
    if (value) headers.set(name, value)
  }
  if (response.headers.get('content-type')?.startsWith('text/event-stream')) {
    headers.set('Cache-Control', 'no-cache, no-transform')
    headers.set('X-Accel-Buffering', 'no')
  }
  return headers
}

function ensureTrailingSlash(value: string) {
  return value.endsWith('/') ? value : `${value}/`
}


import { allowLocalSession, isLocalRequestOrigin, isManualWorkspacePath } from '@/lib/local-development'
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
  const localDevelopment = allowLocalSession(request.url, localBackend, process.env.NODE_ENV, process.env.THETA_LOCAL_AUTH_ENABLED, process.env.THETA_DESKTOP_TOKEN)
  if (!localDevelopment) {
    return NextResponse.json({ detail: '无法使用本地 THETA 服务，请从本机工作台或 THETA 应用启动。' }, { status: 503 })
  }
  if (!isLocalRequestOrigin(request.nextUrl.protocol, request.headers.get('host'), request.headers.get('origin'), request.headers.get('sec-fetch-site'))) {
    return NextResponse.json({ detail: '本地接口不接受跨站请求。' }, { status: 403 })
  }
  if (!isManualWorkspacePath(path)) {
    return NextResponse.json({ detail: '本地工作台不支持此接口。' }, { status: 404 })
  }
  const target = new URL(
    `/${path.map(encodeURIComponent).join('/')}`,
    ensureTrailingSlash(localBackend),
  )
  target.search = request.nextUrl.search

  const headers = new Headers()
  if (localDevelopment && process.env.THETA_DESKTOP_TOKEN) headers.set('x-theta-desktop-token', process.env.THETA_DESKTOP_TOKEN)
  for (const name of [
    'accept',
    'content-type',
    'x-theta-file-size',
    'range',
    'user-agent',
    'x-forwarded-for',
    'x-forwarded-proto',
  ]) {
    const value = request.headers.get(name)
    if (value) headers.set(name, value)
  }
  try {
    const upstream = await fetch(target, streamingRequestInit(request, headers))
    const result = new NextResponse(upstream.body, {
      status: upstream.status,
      headers: responseHeaders(upstream),
    })
    return result
  } catch {
    return NextResponse.json(
      {
        detail: '无法连接本地 THETA 服务，请重新启动工作台或应用。',
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
  ]) {
    const value = response.headers.get(name)
    if (value) headers.set(name, value)
  }
  return headers
}

function ensureTrailingSlash(value: string) {
  return value.endsWith('/') ? value : `${value}/`
}

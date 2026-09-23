import { timingSafeEqual } from 'node:crypto'
import { NextRequest, NextResponse } from 'next/server'
import { isLocalRequestOrigin } from './lib/local-development'

// In desktop mode even static resources belong to the application's private session.
// The secret is injected by Electron, never embedded in a URL or client bundle.
export function proxy(request: NextRequest) {
  const token = process.env.THETA_DESKTOP_TOKEN
  if (!token) return NextResponse.next()
  const supplied = Buffer.from(request.headers.get('x-theta-desktop-token') ?? '')
  const expected = Buffer.from(token)
  if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)
    || !isLocalRequestOrigin(request.nextUrl.protocol, request.headers.get('host'), request.headers.get('origin'), request.headers.get('sec-fetch-site'))) {
    return new NextResponse('THETA desktop session required.', { status: 403 })
  }
  return NextResponse.next()
}

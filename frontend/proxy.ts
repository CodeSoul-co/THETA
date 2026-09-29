import { NextRequest, NextResponse } from 'next/server'
import { desktopAccessError } from './lib/server/desktop-access'

export function proxy(request: NextRequest) { return desktopAccessError(request) ?? NextResponse.next() }

// API handlers authenticate before forwarding the original stream without middleware cloning.
export const config = { matcher: ['/((?!api/).*)'] }

// Read-only connectivity/contract probe. Does not authenticate, upload, or train.
const base = new URL(process.env.THETA_API_BASE_URL || 'https://theta.code-soul.com');
if (base.username || base.password || base.search || base.hash || base.pathname !== '/') {
  throw new Error('THETA_API_BASE_URL must be an origin without credentials, path, query, or fragment.');
}
if (base.protocol !== 'https:' && !(base.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname))) {
  throw new Error('HTTPS is required except for localhost probes.');
}
const checks = [
  { path: '/healthz', status: 200, field: 'status', value: 'ok' },
  { path: '/readyz', status: 200, field: 'status', value: 'ready' },
  { path: '/api/v1/auth/me', status: 401 },
  { path: '/api/v1/projects', status: 401 },
];
const results = await Promise.all(checks.map(async check => {
  try {
    const response = await fetch(new URL(check.path, base), {
      signal: AbortSignal.timeout(15000), redirect: 'error',
      headers: { accept: 'application/json' },
    });
    // Never print raw response bodies, headers, cookies, or tokens.
    const body = await response.json().catch(() => null);
    const matchesBody = check.field ? body?.[check.field] === check.value
      : typeof body?.error?.code === 'string';
    return { path: check.path, expectedStatus: check.status, status: response.status,
      ok: response.status === check.status && matchesBody };
  } catch (error) {
    return { path: check.path, expectedStatus: check.status, status: null, ok: false,
      error: String(error.cause?.code ?? error.code ?? error.name) };
  }
}));
console.log(JSON.stringify({ checkedAt: new Date().toISOString(), baseURL: base.origin, results,
  scope: 'Unauthenticated public API checks only; this does not verify login, storage, workers, training parity, or results.' }, null, 2));
if (results.some(result => !result.ok)) process.exitCode = 1;

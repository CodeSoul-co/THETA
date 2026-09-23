import { apiFetch } from './config'
import type { ConsultationHistory } from '../consultation-history'

const route = (scope: string, action = '') => `/api/v3/consultations${action}?scope=${encodeURIComponent(scope)}`
async function request(scope: string, action = '', method = 'GET', body?: unknown): Promise<ConsultationHistory> {
  const response = await apiFetch<{ data: ConsultationHistory }>('', route(scope, action), {
    method, ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  })
  return response.data
}
export const ConsultationAPI = {
  list: (scope: string) => request(scope),
  import: (scope: string, history: ConsultationHistory) => request(scope, '/import', 'POST', history),
  create: (scope: string) => request(scope, '', 'POST', { id: crypto.randomUUID() }),
  patch: (scope: string, id: string, input: { pinned?: boolean; deleted?: boolean; select?: boolean }) => request(scope, `/${encodeURIComponent(id)}`, 'PATCH', input),
  delete: (scope: string, id: string) => request(scope, `/${encodeURIComponent(id)}`, 'DELETE'),
}

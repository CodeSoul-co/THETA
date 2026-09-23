# THETA Agent Frontend API

对话代码预处理、引用持久化和用户消息计数：见 [预处理与会话契约](conversation-preprocessing.md) 及 [OpenAPI schema](openapi.agent.yaml)。

This is the stable browser-facing contract for the deployable Agent service. It
uses the same session, Origin and CSRF rules as the THETA Business API and does
not change the Worker API. The machine-readable contract is
[`openapi.agent.yaml`](openapi.agent.yaml).

## Deployment boundary

```text
browser --HTTPS--> Nginx --HTTP--> Agent API --HTTP--> Business API --> Worker
```

The public prefix is `/api/v1/agent`. `/api/v3` remains a localhost-only
compatibility route and is disabled in deployed mode. `GET /healthz` is a
liveness check. `GET /readyz` additionally checks that inference and deployed
authentication are configured.

All JSON responses use the existing envelope:

```json
{"ok": true, "data": {}}
```

```json
{"ok": false, "error": {"code": "validation_failed", "message": "..."}}
```

## Local manual-result continuation

`POST /api/v3/projects/import-manual-results` accepts `{projectId: number, selectedJobId?: string}`.
Legacy dataset-only projects may instead supply `datasetName`; the server resolves
that exact registered dataset group (or its saved project), never an arbitrary path.
The ID is the saved manual project ID, not a filesystem path or dataset name.
It returns `{ok:true,data:{project,runId,copiedJobs,reused}}` and creates a persisted
conversation containing independent copies of the project's datasets, completed
compute jobs, model-specific plans, figures, plotting tables and model artifacts.
The selected completed job becomes the initial result selection. Running/failed
jobs, training queues, credentials and approvals are not imported. No inference,
embedding call or retraining is triggered by copying. Follow-up Agent actions use
the normal result-read, interpretation and training approval rules.

The source project is unchanged. Copying the same project/result snapshot again
returns the existing active destination, including after service restart. New
completed results produce a new snapshot. Missing projects return 404; unfinished
selected jobs, invalid paths or result-hash mismatches return 409. The browser
shows copying/error states and only navigates after the copy succeeds.

This bridge is localhost-only and reads the server-configured manual workspace.
Deployed/remote-compute mode rejects it with `import_unavailable` (409); a future
Business API import must verify ownership and transfer artifacts through its
authenticated transport, never read the local unauthenticated manual store.

## Authentication

- Send the existing THETA browser session Cookie. A trusted same-origin proxy
  may translate the Cookie to `Authorization: Bearer <session>`.
- Send `X-CSRF-Token` on `POST`, `PATCH` and `DELETE`.
- The Agent validates each deployed request against
  `GET /api/v1/auth/me` on the existing Business API.
- Projects, conversations, datasets and artifacts are checked against the
  authenticated user. IDs from another user return `404`.
- Nginx should keep the Agent and Business API on the same public origin, so no
  permissive CORS configuration is needed.

## Frontend flow

1. Read `GET /api/v1/agent/capabilities` to render the configured chat model,
   analysis modes, supported data formats, model IDs, tools and limits.
2. Create or select an Agent project.
3. Upload one dataset with `POST /api/v1/agent/datasets`, or reuse an existing
   dataset reference.
4. Create a conversation with `POST /api/v1/agent/runs`.
5. Open `GET /api/v1/agent/runs/{runId}/activities/stream`.
6. Submit a message asynchronously. Always generate a new `requestId` for a new
   user action; reuse the same ID only when retrying the identical request.
7. Render `conversation`, `activities` and `checkpoint` values received in the
   SSE `sync` event. A checkpoint decision must include the displayed
   `checkpointId` and `expectedContentHash`.
8. Read `GET /runs/{runId}/results`. A user may explicitly save a result
   selection, or Workbench may resolve `automatic/latest_completed`; both modes
   are persisted as the same immutable list of job IDs before interpretation.
   Both modes reuse the result UI. Local manual projects retain independent
   storage. Clicking “咨询结果助手” on a manual result explicitly copies completed
   results into an Agent project via the local bridge described below;
   merely switching workbench modes does not copy anything.
   A completed result opens from the opaque `artifacts.reportUrl`
   returned for that project run and is never retrained just for navigation.
9. Send the interpretation question with that selection. The selection does not
   grant result access or interpretation: the existing `results.read` and
   `results.synthesize` confirmation cards still apply.
10. Open artifact URLs returned in Agent messages. Public messages never contain
   host filesystem paths.

## Endpoint reference

| Method | Path | Returned data |
| --- | --- | --- |
| `GET` | `/healthz` | Process liveness |
| `GET` | `/readyz` | Inference and authentication readiness |
| `GET` | `/api/v1/agent/capabilities` | Inference, compute, formats, all model parameter contracts, tools and limits |
| `GET` | `/api/v1/agent/inference` | Frontend-safe inference provider/model catalog and current selection |
| `GET` | `/api/v1/agent/inference/settings` | Current non-secret inference settings; deployed mode is read-only |
| `POST` | `/api/v1/agent/advisory` | Read-only consultation or interpretation of up to four charts; optional `consultation` saves messages and references in the sidebar history database, without creating a training run |
| `GET/POST` | `/api/v1/agent/consultations?scope=…` | List/create project-scoped sidebar consultations for the authenticated user |
| `PATCH/DELETE` | `/api/v1/agent/consultations/{id}?scope=…` | Pin, select, restore, or soft-delete a sidebar consultation |
| `POST` | `/api/v1/agent/consultations/import?scope=…` | Idempotently migrate previous browser history without overwriting existing/deleted records |
| `GET`, `POST` | `/api/v1/agent/projects` | Project list or created project |
| `PATCH`, `DELETE` | `/api/v1/agent/projects/{projectId}` | Updated or archived project |
| `GET`, `POST` | `/api/v1/agent/datasets?projectId={projectId}` | Project-scoped dataset list or uploaded dataset metadata |
| `GET`, `POST` | `/api/v1/agent/runs` | Conversation list or initial conversation snapshot |
| `GET`, `PATCH`, `DELETE` | `/api/v1/agent/runs/{runId}` | Snapshot, changed identity fields, or archive receipt |
| `GET` | `/api/v1/agent/runs/{runId}/conversation` | User-visible messages and confirmation-card history |
| `GET` | `/api/v1/agent/runs/{runId}/activities` | Recent safe tool/provider execution activity |
| `GET` | `/api/v1/agent/runs/{runId}/activities/stream` | SSE `sync` snapshots for the complete conversation view |
| `GET` | `/api/v1/agent/runs/{runId}/checkpoint` | Current confirmation card or `null` |
| `GET` | `/api/v1/agent/runs/{runId}/results` | Selectable training results, hydration state and current exact selection, plus per-result artifact metadata (report URL, figure/table counts, file list, bundle URL) |
| `GET` | `/api/v1/agent/runs/{runId}/results/{jobId}/archive` | ZIP of one result's figures, tables and report document; model binaries (`.npy`, `.joblib`, `pt`, `bin`) are excluded |
| `GET` | `/api/v1/agent/runs/{runId}/results/{jobId}/figure-bundle?path={figure}` | ZIP of one figure (all delivered formats), the CSV table it was drawn from, `adjustment-spec.json` for re-rendered figures, and `plot-code.md` with the implementation notes |
| `POST` | `/api/v1/agent/runs/{runId}/result-selection` | Resolve and persist a manual or automatic selection without reading result contents |
| `POST` | `/api/v1/agent/runs/{runId}/messages` | Completed snapshot or `202` asynchronous task receipt |
| `POST` | `/api/v1/agent/runs/{runId}/checkpoint-decision` | Confirmation decision result or asynchronous task receipt |
| `POST` | `/api/v1/agent/runs/{runId}/stop` | Cancellation signal receipt |
| `GET` | `/api/v1/agent/runs/{runId}/tasks/{taskId}` | Durable task status and progress |
| `GET` | `/api/v1/agent/runs/{runId}/artifacts/{artifactId}/download` | Authorized result/report bytes |

The precise request and response types, optional fields, enums, HTTP statuses,
binary content types and SSE payload schema are defined in
[`openapi.agent.yaml`](openapi.agent.yaml). `/api/v1/agent/conversations` is an
implemented alias for `runs`, but new frontend code should use the documented
`runs` paths.

The advisory request accepts an optional `purpose`: `consultation` (default) or
`chart-analysis`, which switches the instruction set to a research-result
paragraph (one conclusion, two or three statements carrying their own
denominator or denominator proxy, and one explicit limitation) instead of free
consultation.

Manual mode must use `POST /api/v1/agent/advisory`, not a run message endpoint.
The service sends no tools to the model and does not create a training run.
With `consultation: {scope, id, requestId}`, the question, answer and chart references
are saved server-side to the independent sidebar consultation history. Without it,
the endpoint remains stateless (for standalone chart analysis). See
[右侧咨询历史存储与接口](consultation-history.md) for migration, pin/delete and recovery.
Training, parameter changes
and task cancellation remain project-card operations. Selecting a chart sends the
CSV/JSON data used to draw it, plus dataset/model/chart identifiers. No image bytes
are sent, so this flow does not require a vision-capable model. A request may contain
up to four charts and up to four data sources per chart; truncated sources must carry
their original character count so the answer can disclose the evidence limit.

The deployed inference catalog is read-only and is selected by service
configuration. For local Workbench development only, `PATCH
/api/v3/inference/settings` accepts `{"llm":{"providerId":"...","model":"..."}}`.
The provider must already have credentials configured, and the model must match
that provider's configured model. The local response uses the same provider ID as
the catalog, so the custom selector can preserve its selected state. The public
`/api/v1/agent/inference/settings` endpoint does not permit browser-side mutation.

## CLI information exposed to the frontend

`GET /api/v1/agent/capabilities` is the frontend equivalent of the safe parts
of CLI `/model`, `/models`, `/help` and runtime status. It exposes provider/model
names, supported model IDs, each model's description, runtime profile, input
requirements, embedding options and complete native parameter dictionary. A
parameter entry includes its type, default, choices when finite, execution
target and repository-relative implementation source. The response also includes
tool descriptions, dataset formats, approval types, limits and transport
features. It deliberately excludes credentials, provider URLs, prompts, tool
arguments, host filesystem paths and private reasoning.

The model parameter names are the actual plan keys, including namespaces such
as `prepare.*`, `model.*`, `trainer.*`, `config.*`, `main.*`, `pipeline.*` and
`embedding.*`. The frontend may use these fields for an inspector or form, but
the Agent still validates the final plan against the same native contract before
submission.

Execution activity uses the existing `ExecutionEvent` v1 semantics: tool or
provider name, safe detail, status, timestamps and duration. The endpoint never
returns chain-of-thought.

## Display guarantees for browser clients

- Host filesystem paths never reach the browser: message bodies, confirmation
  cards and the native report HTML have absolute paths and internal task paths
  (`job-…`, `report-…`) replaced with a neutral label.
- Figures referenced from an assistant message are rewritten to **absolute**
  same-origin artifact URLs, because the Workbench markdown renderer only draws
  images that carry an absolute `http(s)` destination. Plain links stay
  relative.
- The conversation payload is cached per run and invalidated by message, report
  or card changes, so polling clients (SSE `sync`) never rebuild it per tick.

## Asynchronous messages and SSE

Message submission:

```http
POST /api/v1/agent/runs/{runId}/messages
Content-Type: application/json
X-CSRF-Token: <token>

{
  "content": "分析这个数据集",
  "attachments": [{"kind": "dataset", "id": "dataset-..."}],
  "async": true,
  "requestId": "browser-generated-uuid"
}
```

`modelPreference` remains an optional compatibility field and accepts `lda`,
`btm`, `hdp`, `dtm`, `stm`, `bertopic`, `ctm`, `theta`, `etm`, `nvdm`, `gsm`,
or `prodlda`. It is a topic-analysis planning hint, not an LLM selector. The
normal conversation composer does not send it: the Agent LLM is selected through
the custom provider/model control backed by `GET /inference` and
`POST /inference/settings`. Topic models are proposed inside the research plan
and confirmed through the normal approval boundary. If an integration sends
`modelPreference`, it influences plan formation only and never approves
training.

## Result selection and interpretation

Result selection is a host-side boundary. It never authorizes result reading,
model inference, retraining or external embedding calls.

Manual selection:

```http
POST /api/v1/agent/runs/{runId}/result-selection

{"mode":"manual","jobIds":["job-..."]}
```

Workbench automatic selection:

```http
POST /api/v1/agent/runs/{runId}/result-selection

{"mode":"automatic","strategy":"latest_completed"}
```

The response always contains `resolvedJobIds`. Automatic resolution is saved at
that moment; later training completion does not mutate the selection. A message
may include a `ResultSelectionRequest` to atomically resolve/select before the
turn. If the Workbench first calls the selection endpoint, it should pass the
returned `resolvedJobIds` as a manual request in the later message so the exact
selection cannot drift. The Agent receives the resolved IDs as trusted host context and must
bind `results_read` / `results_synthesize` to those IDs. The user still confirms
the normal result-reading and deep-interpretation cards.

`GET /results` exposes task identity, model, live phase/progress and hydration
readiness. After result access has been approved it also returns
`artifacts.reportUrl` (native HTML report), `artifacts.archiveUrl` (one ZIP for
that result), file/figure/table/matrix counts and `artifacts.files[]` with
`{name, kind, url}` for every delivered figure, table, matrix and report
document. The conversation drawer and the manual workbench render the same
catalog for the same `runId`: a model's figures are listed per result, can be
previewed inline, quoted into the conversation and downloaded as a figure
bundle. Results re-rendered by the Agent's `figure_adjust` tool appear under
`adjustments/` in the same list, so one render shows up in the reply, the
drawer and the manual result page while the original figure is untouched. It never returns raw source rows or result contents. Once approved,
the Agent downloads the signed archive, verifies the Worker-provided SHA-256,
extracts only regular files into its private persistent volume and generates a
local `theta.result-report.v2`. Original-row interpretation is allowed only when
the report proves row alignment; otherwise the response states that limitation.

The three Worker containers used by the local 12-model acceptance run were
temporary, identical Redis consumers used to shorten the test. They are not
three Agents and do not create three user contexts. A deployment may run one
Worker (jobs queue serially) or several compatible replicas (jobs are claimed
from the same queue); the Agent and these APIs are unchanged. CPU and GPU
Workers should use their respective streams and runtime images.

The response is `202` with a durable task receipt. Polling a task is a view and
does not resubmit work. Nginx must disable buffering for the SSE route and use a
long read timeout.

## Browser example

All calls should be same-origin and include the existing session cookie. The
CSRF token below is the token obtained from the existing Business API login or
`GET /api/v1/auth/me` response.

```ts
const prefix = '/api/v1/agent';

async function agentJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${prefix}${path}`, {
    credentials: 'include',
    ...init,
    headers: { 'Content-Type': 'application/json', ...init.headers },
  });
  const envelope = await response.json();
  if (!response.ok || !envelope.ok) throw new Error(envelope.error?.message ?? 'Agent request failed');
  return envelope.data as T;
}

const project = await agentJson<{id: string}>('/projects', {
  method: 'POST',
  headers: { 'X-CSRF-Token': csrfToken },
  body: JSON.stringify({ name: '舆情研究' }),
});

const run = await agentJson<{runId: string}>('/runs', {
  method: 'POST',
  headers: { 'X-CSRF-Token': csrfToken },
  body: JSON.stringify({ projectId: project.id, researchGoal: '分析主题变化', analysisMode: 'topic' }),
});

const events = new EventSource(`${prefix}/runs/${encodeURIComponent(run.runId)}/activities/stream`, {
  withCredentials: true,
});
events.addEventListener('sync', event => {
  const state = JSON.parse((event as MessageEvent).data);
  renderConversation(state.conversation, state.snapshot, state.activity, state.checkpoint);
});

const selection = await agentJson(`/runs/${encodeURIComponent(run.runId)}/result-selection`, {
  method: 'POST',
  headers: { 'X-CSRF-Token': csrfToken },
  body: JSON.stringify({ mode: 'automatic', strategy: 'latest_completed' }),
});

await agentJson(`/runs/${encodeURIComponent(run.runId)}/messages`, {
  method: 'POST',
  headers: { 'X-CSRF-Token': csrfToken },
  body: JSON.stringify({ content: '解释我选中的结果', resultSelection: { mode: 'manual', jobIds: selection.selection.resolvedJobIds },
    async: true, requestId: crypto.randomUUID() }),
});
```

For uploads, do not set `Content-Type` manually because the browser must create
the multipart boundary:

```ts
const form = new FormData();
form.append('file', file);
const response = await fetch(`${prefix}/datasets`, {
  method: 'POST', credentials: 'include',
  headers: { 'X-CSRF-Token': csrfToken }, body: form,
});
```

Uploading or dragging a dataset only creates an attachment draft. The frontend
opens the research drawer so the user can manage the dataset, but it must not
submit a message automatically. The user may add instructions and explicitly
send the attachment with the next message.

The same upload flow is available directly inside the research drawer before a
dataset, run, or conversation exists. It reuses the dataset upload endpoint; no
separate API is required. The managed dataset remains attached to the project
after a message is sent, while one-time result selections are cleared. Later
Agent messages therefore carry the managed dataset reference automatically.

When a project is reopened, the frontend refreshes
`GET /datasets?projectId=...` and restores those opaque dataset references into
the research drawer before the next message. Upload success creates or reuses
the same Agent project used by conversation and manual mode. The drawer and
composer render suffix-aware SVG file icons; this presentation does not change
the upload API.

Dataset ownership is user-wide, but attachment is project-scoped. Supplying
`projectId` on upload binds the managed dataset to that project; supplying it on
list returns only the project's datasets. Creating a run with `datasetRef`, or
later sending that dataset as an attachment, also persists the same binding.
Legacy sessions are backfilled only when their run already references the
dataset, so a newly created project never inherits unrelated user uploads.

## Unified project and result navigation

Conversation mode, manual mode and project management are views over the same
Agent projects and runs. A successful upload creates or reuses this project;
there is no upload-only project and no result-only project. The manual project
center lists the `projectId` values returned by the Agent project API; it must not
synthesize an extra “Agent task” card or copy a run into the legacy project
namespace. Opening a completed project reads its existing
`GET /runs/{runId}/results` catalog and embeds or links the same-origin opaque
`artifacts.reportUrl`. Once result reading is authorized, `artifacts.files`
contains `{name, kind, url}` entries for the same report's figures, tables,
matrices and downloads. The frontend can therefore render native result tabs
and chart cards without scraping the full HTML report; all URLs remain opaque,
project-authorized artifact routes. Returning to the project center shows the
same project card, with its existing result state.

Mode navigation causes no message submission, training request, parameter
change, new run or Worker call. Manual-mode questions continue to use
`POST /advisory` and cannot mutate training.

## Confirmation cards

Approving, rejecting and revising are separate host actions:

```http
POST /api/v1/agent/runs/{runId}/checkpoint-decision

{
  "action": "approve",
  "checkpointId": "...",
  "expectedContentHash": "...",
  "async": true,
  "requestId": "browser-generated-uuid"
}
```

An expired or replaced card returns `409`. A revision must include `feedback`.
Agent text alone cannot bypass the host-side content-hash and effect approval
checks.

## Data and artifacts

The local Agent upload limit is 200 MiB and supports CSV, TSV, TXT, Markdown,
JSON/JSONL/NDJSON, XLS/XLSX, Parquet, PDF and DOCX. Hosted training still goes
through the existing Business API and Worker adapter.

Public artifact links use opaque IDs:

```text
GET /api/v1/agent/runs/{runId}/artifacts/{artifactId}/download
```

The old `files?path=` route is available only in localhost development mode.

## Container configuration

Required in deployed mode:

- `THETA_AGENT_SERVICE_MODE=deployed`
- `THETA_BUSINESS_API_URL`
- `THETA_AGENT_ALLOWED_ORIGINS`
- `THETA_AGENT_TRUSTED_HOSTS`
- one complete inference provider configuration
- `THETA_BUSINESS_BINDINGS` for hosted model/runtime mappings
- persistent `THETA_AGENT_HOME`, normally `/data`

Build and run locally:

```bash
docker compose -f agent/deploy/docker-compose.yml build
docker compose -f agent/deploy/docker-compose.yml up -d
curl http://localhost:8088/healthz
curl http://localhost:8088/api/v1/agent/capabilities
```

For production, include `agent/deploy/nginx-agent-location.conf` in the existing
TLS virtual host instead of starting a second public Nginx.

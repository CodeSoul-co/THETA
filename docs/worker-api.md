# Remote training worker

[中文](worker-api.zh.md) | English

**Cloud compute is coming soon. This release provides the configuration entry and compatibility contract.**

THETA keeps the Agent, conversation history, project ownership, data preview, and report writing on the user's computer. Training defaults to the bundled local worker. Settings → Compute can select a remote service for new training tasks in both conversation and manual modes. This is a client integration contract; this repository does not deploy a hosted commercial service.

Set the API base URL, API key, currency (CNY or USD), and maximum cost per experiment. Zero permits free tasks only. Clicking Done saves the configuration. Keys are encrypted with a private local key and are excluded from installers. Batch cost can reach the per-experiment limit multiplied by the number of model/topic-count combinations. The training dialog explicitly confirms uploading the selected dataset and any independently uploaded train/validation/test files. Local LLM and embedding keys are never forwarded. Model weights and any embedding service credentials required by remote training must be provisioned by the remote service.

Project IDs are namespaced per local workspace before transfer, so identically named manual projects on different computers remain distinct.

## HTTP contract

All JSON responses below are direct objects, without an `ok/data` wrapper. All control requests use `Authorization: Bearer <key>`. Production URLs require HTTPS; loopback HTTP supports development. Redirects are rejected. Errors use an HTTP error code and `{"error":{"message":"reason"}}`. Clients must never supply a local executable, shared filesystem path, or shell command to the remote service.

| Method and path | Request / response |
| --- | --- |
| `GET /v1/capabilities` | `{ "protocol": "theta-worker-v1", "models": ["lda", "bertopic"] }` |
| `POST /v1/preview` | `{dataset, plan}` → `{readiness:{ready,issues},rowCount,quote:{id,maxCost,currency,expiresAt}}`. Preview must not upload data, start training, or incur a model call charge. |
| `POST /v1/datasets` | Multipart `file`, `projectId`, `sha256` → `{datasetId,projectId,sha256}`. Verify the actual uploaded bytes before acknowledging. |
| `POST /v1/jobs` | `{clientJobId,idempotencyKey,projectId,dataset:{datasetId,sha256},plan,quoteId,maxCost,currency}` → `{id,projectId}`. Return promptly after a durable enqueue. |
| `GET /v1/projects/:projectId/jobs/by-client-id/:clientJobId` | Resolve an uncertain submission; return the same job identity, or an explicit not-found response. |
| `GET /v1/projects/:projectId/jobs/:jobId` | `{id,projectId,status,phase,percent,message,error,phaseHistory,telemetry,diagnostics}` |
| `POST /v1/projects/:projectId/jobs/:jobId/cancel` | Empty object; cancel the owned task idempotently. |
| `GET /v1/projects/:projectId/jobs/:jobId/results` | Completed delivery manifest described below. |

Dataset metadata in preview contains `datasetRef`, `fileName`, `sha256`, and `sizeBytes`; it never contains `managedPath`. Plans use the repository's `TrainingPlan` structure: model, parameters, selected columns, device, timeout, and `dataSplit`. Ratio splitting carries ratios, method, and seed. Independent uploads preserve each role and selected columns under `dataSplit.sources`; submission replaces each source's local descriptor with its uploaded `{datasetId,sha256}`. Model paths and secret-bearing plan parameters are rejected. The service must implement the requested split semantics, timeout, embedding request limit, and quote bound rather than silently ignore them.

The service must authorize the authenticated account's project, dataset, and job on **every** request. `projectId` is a client project identifier, not proof of ownership. Scope idempotency to account + project + key, persist it before charging or enqueueing, return the same job for identical retries, and reject conflicting payloads. Never charge beyond the accepted quote; expired quotes require a new confirmation. The client keeps the endpoint and credential binding for existing jobs even after the user switches new tasks back to local compute.

Use `queued`, `running`, `completed`, `failed`, or `cancelled`. `telemetry` follows `JobTelemetry` in `agent/src/domain/research.ts`, including epoch/batch/embedding events. Errors should describe the model, stage, cause, and corrective action. An unsuccessful model must not invalidate separate completed model jobs.

## Complete portable delivery

The manifest has this shape:

```json
{
  "schema": "theta-results-v1",
  "status": "ready",
  "projectId": "client-project-id",
  "jobId": "remote-job-id",
  "files": [
    {"name":"index.html","url":"https://worker.example/files/report","sha256":"64 lowercase hex characters","sizeBytes":1024,"kind":"report"}
  ],
  "evidence": {"evidence":[],"tables":[],"figures":[],"matrices":[]},
  "sourceData": {},
  "quality": {},
  "summary": {},
  "diagnostics": {}
}
```

The example file list is abbreviated. Include **all** training outputs, evaluation metrics, topic words, matrices, split outputs, visualization files and plotting tables, and logs. Preserve the native THETA relative layout: training outputs under `training/`, regenerated charts and plotting tables under `native/`, and `index.html` at the root. Include exactly one `split_results.json` with `groups.all`, `groups.train`, `groups.validation`, and `groups.test`; split charts use `native/.../splits/<role>/...`. Reuse the repository's native `theta.result-report.v2` evidence/quality/source-data fields. Every evidence `path` must name a file in `files` by its relative name, never a server filesystem path. HTML must reference relative delivered assets. Kinds are `figure`, `table`, `matrix`, `report`, or `artifact`.

Each file URL must be HTTPS (loopback HTTP for tests); signed query strings are supported. Credentials are sent only to the API's exact origin, never to a different object-storage origin. Cross-origin URLs therefore need their own short-lived authorization. No redirects are followed. Paths must be portable on Mac and Windows: no absolute paths, parent traversal, reserved device names, case-colliding duplicates, or executable hooks.

A server's `completed` status does not immediately become completed in THETA. The client displays “receiving and verifying results” until every declared file passes size/SHA-256 checks and the local capability API registers the delivery. Only then are results made available to both modes. Reports and result downloads continue to work locally after changing the configured service. Download failures do not trigger a new training task or a second charge; restore the connection and reopen the app to retry the same delivery. Locally modified results fail verification instead of being treated as original evidence.

## Configuration and validation

Desktop stores `compute-settings.json` beside its normal settings. Source Web uses the Agent home for both services. Custom launchers should set the **same** `THETA_COMPUTE_SETTINGS_HOME` for Agent and manual services while keeping their project/compute homes separate. The existing `THETA_WORKER_API_URL` is the internal local capability API; do not use it to configure a commercial training service.

Contract tests cover real HTTP upload/control/download requests, project scoping, split inputs, quote limits, original approval binding, both training modes, restoration after configuration changes, checksum failure, and local default behavior. Python tests cover registering downloaded results and generating local report evidence. These tests use an isolated fixture service; deployment-specific billing, account authorization, GPU scheduling, and real provider interoperability require acceptance testing against that provider. Native plot editing that requires server-only model/runtime provenance is not emulated locally.

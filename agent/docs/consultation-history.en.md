# Consultation history

**English** | [中文](consultation-history.md)

The right-side project assistant is separate from training conversations. The Agent stores consultations and the selected thread in `THETA_AGENT_HOME/research.sqlite`. Source installations default to `.theta_agent/research.sqlite`.

Records include questions, replies, chart references and data, title, pin state, deletion time, and update time. Unsent drafts remain in the browser and are scoped to the consultation ID.

## Local API

Use the `/api/v3` prefix. The `scope` value identifies the project or result namespace; it does not grant training permission or additional data access.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/consultations?scope=...` | Return active ID and threads, including soft-deleted records; pinned threads sort first |
| POST | `/consultations?scope=...` | Create and select a thread; repeated IDs are idempotent |
| PATCH | `/consultations/{id}?scope=...` | Pin, select, delete, or restore a thread |
| DELETE | `/consultations/{id}?scope=...` | Soft-delete without erasing text and select another available thread |
| POST | `/consultations/import?scope=...` | Import missing IDs in one transaction without replacing existing records |
| POST | `/advisory` | Use `consultation:{scope,id,requestId}` to save the question and response with thread context |

Repeating a request ID with identical content does not call the model again. Changed content returns 409; an in-progress request returns 202. A new question is rejected while the previous answer is unfinished. Requests without a consultation retain stateless advisory behavior.

Limits are 16,000 question characters, four charts with up to four CSV/JSON sources each, and imports of at most 1,000 consultations, 2,000 messages per thread, or 32 MiB. Excess input is rejected rather than silently truncated.

## Restore and delete

Existing browser history can be imported without overwriting server records or deletion markers. Questions are saved before model invocation. Navigating away does not prevent the service from saving its response while the service remains running.

Service restarts mark unfinished replies as interrupted without automatically resending paid requests. Pinning or deletion while a response is pending is preserved. Deleted consultations can be restored. Database failures are shown as errors; browser drafts are not presented as synchronized history.

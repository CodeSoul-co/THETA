# Agent confirmation cards

**English** | [中文](web-confirmation-cards.md)

## Presentation

Cards appear only for real actions requiring approval. Ordinary discussion and data understanding do not create artificial approval steps. Pending cards appear above the composer and remain available while you browse conversation history.

Training cards identify data, models, selected columns, device, runtime limit, topic count, iterations, embeddings, and external-request scope. Result cards identify the action and task. Details can be expanded. Long descriptions scroll within the card while controls remain visible; narrow layouts use a single column.

Card IDs and content hashes identify the approved version. Routine messages and progress updates do not invalidate a card; a changed plan makes the old card read-only.

## Decisions and feedback

| Situation | Action | Result |
| --- | --- | --- |
| Accept the proposal | Confirm | Approves this card once; controls are disabled during submission |
| Decline | Reject | Continues the conversation without execution |
| Request changes | Edit and submit feedback | Rejects the original authorization; a revised proposal requires a new card |
| Feedback is being edited | Submit changes | The original execution button is replaced until editing is cancelled |
| Refresh or change conversations | Resume | Saved decisions return; unsent feedback is retained for the same tab |
| Network failure | Retry after reviewing the error | Feedback remains; success is not assumed |
| Processed, replaced, or expired card | Inspect history | State is shown explicitly and cannot execute again |
| Execution failed | Inspect the error | Check server state before retrying; consumed approvals cannot be reused |

History retains the original proposal, version, decision, and feedback. Approval and training completion are separate states: an approved action is not necessarily a successful computation.

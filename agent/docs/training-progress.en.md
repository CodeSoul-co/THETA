# Training status and progress

**English** | [中文](training-progress.md)

The terminal checks the current research job every three seconds, including while you type or wait for a model response. Status checks do not call the language model or resubmit training. Failed queries are retried. Background training continues after the terminal closes; resume monitoring with `./theta --session <session-id>`.

Local pipeline percentages such as 20% and 60% identify stages, not measured completion. The terminal shows stage, elapsed time, stage duration, worker heartbeat, and log freshness. Explicit iteration messages are shown as reported iterations without assuming completion or estimating an ETA.

A heartbeat within 30 seconds indicates that the worker responds; it does not prove algorithmic progress or training quality. Missing logs are reported separately. A stale heartbeat makes status uncertain. The job record determines the final state.

`ComputeJob.telemetry` uses the UI-independent `theta.job-observation.v1` structure for observation times, health signals, stage timing, log freshness, and verified iteration counts. Workers without this structure report that detailed progress is unavailable. Monitoring extracts limited activity signals without interpreting training results.

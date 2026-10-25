# THETA 0.3.17

[中文](desktop-v0.3.17.zh.md) | English

- Local Agent with selectable training compute. Local worker remains the default; Settings → Compute accepts a compatible remote worker API, key, and per-experiment cost limit.
- Conversation and manual training share the same compute selection, with explicit data-upload and cost confirmation.
- Remote progress includes training and embedding events. Training is marked complete only after results, split outputs, and charts have been downloaded and verified.
- Existing remote jobs retain their original service binding after settings change. Downloaded evidence is available locally for result browsing and analysis reports.
- User credentials are stored locally and excluded from packages. Existing projects and settings are retained.

Requires a THETA Worker v1 compatible service for remote training; no hosted service is included. Provider billing and infrastructure need separate acceptance testing. See the [integration contract](../worker-api.md).

Preview installers remain unsigned/not notarized. Python and compute dependencies are bundled; model weights are configured separately.

# Local and cloud embeddings

**English** | [中文](local-embedding.md)

THETA zero-shot uses pretrained word and document vectors. It supports local weights and OpenAI-compatible cloud embedding services.

| Provider | Configuration |
| --- | --- |
| `cloud` | Set `EMBEDDING_PROVIDER=cloud`, `EMBEDDING_CLOUD_PROVIDER`, `EMBEDDING_MODEL`, and `EMBEDDING_API_KEY` or a provider-specific key |
| `local` / `qwen` | Set `QWEN_MODEL_0_6B` to the model directory; 4B and 8B use `QWEN_MODEL_4B` and `QWEN_MODEL_8B` |

In the desktop app, use the embedding settings instead of editing environment files. The local default suggestion is Qwen3-Embedding-0.6B; the cloud preset is Zhipu `embedding-3`.

## Prepare local weights

The source installation includes an explicit download helper:

```bash
agent/.venv/bin/python agent/scripts/download_qwen_embedding.py
```

It downloads into `models/qwen3_embedding_0.6B/`. Configure the absolute path in private `agent/.env.local`, preserving any existing settings. A complete model directory needs its configuration, tokenizer, and weights. Readiness checks reject missing configuration or empty weight files.

## Choose an execution mode

Cloud embeddings send the confirmed text scope to your configured service and use an explicit request budget. The Agent default `externalRequestLimit` is 100; inspect the proposal and adjust it for the workload before confirming, up to the supported limit. Failed HTTP attempts also consume the budget.

Local embeddings keep this computation on your machine. Runtime depends on hardware and corpus size; cached embeddings can be reused. Logs identify the embedding mode and local cache paths. Missing weights are not downloaded automatically by a training request.

Both modes use the same native plots, tables, and report structure. Cloud embeddings are available for THETA zero-shot. Fine-tuning, CTM, and BERTopic require compatible local models.

# 本地与云端 Embedding

[English](local-embedding.en.md) | **中文**

THETA 的 `zero_shot` 模式需要预训练词/文档向量，两条路都支持：

| provider | 说明 | 需要的配置 |
| --- | --- | --- |
| `cloud` | OpenAI 兼容的 embedding 服务（默认 zhipu `embedding-3`） | `EMBEDDING_PROVIDER=cloud`、`EMBEDDING_CLOUD_PROVIDER`、`EMBEDDING_MODEL`、`EMBEDDING_API_KEY`（或 provider 专属 key） |
| `local` / `qwen` | 本机 Qwen3-Embedding 权重，完全不出网 | `QWEN_MODEL_0_6B=/abs/path/to/models/qwen3_embedding_0.6B`（4B/8B 用 `QWEN_MODEL_4B` / `QWEN_MODEL_8B`） |

## 本地权重准备

HuggingFace 在部分网络下不可达，ModelScope 可用：

```bash
# 约 1.2 GB，落到 models/qwen3_embedding_0.6B/
agent/.venv/bin/python agent/scripts/download_qwen_embedding.py
# 然后让 Agent 看到它
echo 'QWEN_MODEL_0_6B='"$PWD"'/models/qwen3_embedding_0.6B' >> agent/.env.local
```

权重目录至少要包含 `config.json` 与一个非空的 `*.safetensors`——就绪检查
（`runtime_check`）会校验这两项，缺任何一项 `modelAssetsReady` 都是 false。

## 两种 provider 怎么选

- **要快 / 语料小**：`cloud`。注意外部调用有额度：默认 `externalRequestLimit=100`，
  5000 词表 + 上千文档的场景要在确认卡里把它提到 1000，否则会以
  `本次外部调用额度已用完` 失败（THETA 的常见失败原因）。
- **不想出网 / 额度用完**：`local`。CPU 上 2400 篇文档 + Qwen 0.6B 首次 embedding
  需要几十分钟，之后走缓存；训练日志里会写 `Using pretrained word embeddings (frozen)`，
  并给出本地 `data/embeddings` 路径，可据此确认没有调用外部服务。

两条路产物一致：同一套 `native/` 图表、`tables/` 数据表与原生报告，结果页与对话共用。

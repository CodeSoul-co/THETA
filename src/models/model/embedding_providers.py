"""
Unified embedding providers for local and cloud embedding services.

Cloud providers use OpenAI-compatible /v1/embeddings APIs. Provider-specific
defaults can be overridden with EMBEDDING_API_BASE, EMBEDDING_MODEL, and
EMBEDDING_API_KEY or a provider-specific key variable.
"""

import json
import os
import time
import urllib.error
import urllib.request
import http.client
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


LOCAL_PROVIDERS = {"local", "qwen"}


def split_utf8_text(text: str, maximum_bytes: int) -> List[str]:
    """Conservative token bound without downloading a provider tokenizer.

    UTF-8 bytes upper-bound byte-token counts; never split a Unicode character.
    """
    chunks, current, size = [], [], 0
    for char in text:
        width = len(char.encode('utf-8'))
        if size + width > maximum_bytes and current:
            chunks.append(''.join(current)); current, size = [], 0
        current.append(char); size += width
    if current:
        chunks.append(''.join(current))
    return chunks

PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "text-embedding-3-small",
    },
    "dashscope": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "text-embedding-v4",
    },
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model": "BAAI/bge-m3",
    },
    "zhipu": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPUAI_API_KEY",
        "model": "embedding-3",
    },
    "volcengine": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "ARK_API_KEY",
        "model": "doubao-embedding-text-240715",
    },
    "openai_compatible": {
        "api_base": "",
        "api_key_env": "EMBEDDING_API_KEY",
        "model": "",
    },
}


@dataclass
class EmbeddingProviderSettings:
    provider: str
    cloud_provider: str
    api_base: str
    api_key_env: str
    model: str
    dimensions: Optional[int] = None
    normalize: bool = True
    timeout: int = 120
    max_retries: int = 2

    @property
    def is_cloud(self) -> bool:
        return self.provider not in LOCAL_PROVIDERS

    @property
    def api_key(self) -> Optional[str]:
        # A task-selected key name must win over the generic fallback. This lets
        # one Worker host serve multiple embedding providers without an
        # unrelated EMBEDDING_API_KEY silently overriding the requested secret.
        return os.environ.get(self.api_key_env) or os.environ.get("EMBEDDING_API_KEY")


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_embedding_config_attr(config: Any, name: str, default: Any = None) -> Any:
    embedding = getattr(config, "embedding", config)
    return getattr(embedding, name, default)


def resolve_embedding_settings(
    config: Any = None,
    provider: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> EmbeddingProviderSettings:
    configured_provider = (
        provider
        or _get_embedding_config_attr(config, "provider", None)
        or os.environ.get("EMBEDDING_PROVIDER")
        or "cloud"
    ).strip().lower()

    if configured_provider in LOCAL_PROVIDERS:
        return EmbeddingProviderSettings(
            provider=configured_provider,
            cloud_provider="",
            api_base="",
            api_key_env="",
            model="",
        )

    if configured_provider == "cloud":
        configured_cloud_provider = (
            cloud_provider
            or _get_embedding_config_attr(config, "cloud_provider", None)
            or os.environ.get("EMBEDDING_CLOUD_PROVIDER")
            or "openai"
        ).strip().lower()
    elif configured_provider in PROVIDER_DEFAULTS:
        configured_cloud_provider = configured_provider
    else:
        configured_cloud_provider = (
            cloud_provider
            or os.environ.get("EMBEDDING_CLOUD_PROVIDER")
            or _get_embedding_config_attr(config, "cloud_provider", None)
            or configured_provider
        ).strip().lower()

    defaults = PROVIDER_DEFAULTS.get(configured_cloud_provider, PROVIDER_DEFAULTS["openai_compatible"])
    resolved_api_base = (
        api_base
        or _get_embedding_config_attr(config, "api_base", None)
        or os.environ.get("EMBEDDING_API_BASE")
        or defaults.get("api_base", "")
    ).rstrip("/")
    resolved_model = (
        model
        or _get_embedding_config_attr(config, "model", None)
        or os.environ.get("EMBEDDING_MODEL")
        or defaults.get("model", "")
    )
    resolved_api_key_env = (
        api_key_env
        or _get_embedding_config_attr(config, "api_key_env", None)
        or os.environ.get("EMBEDDING_API_KEY_ENV")
        or defaults.get("api_key_env", "EMBEDDING_API_KEY")
    )
    resolved_dimensions = _coerce_int(
        dimensions
        or _get_embedding_config_attr(config, "dimensions", None)
        or os.environ.get("EMBEDDING_DIMENSIONS")
    )

    return EmbeddingProviderSettings(
        provider=configured_provider,
        cloud_provider=configured_cloud_provider,
        api_base=resolved_api_base,
        api_key_env=resolved_api_key_env,
        model=resolved_model,
        dimensions=resolved_dimensions,
        normalize=_coerce_bool(
            _get_embedding_config_attr(config, "normalize", os.environ.get("EMBEDDING_NORMALIZE")),
            default=True,
        ),
        timeout=int(os.environ.get("EMBEDDING_TIMEOUT", "120")),
        max_retries=int(os.environ.get("EMBEDDING_MAX_RETRIES", "2")),
    )


def is_cloud_embedding(config: Any = None, provider: Optional[str] = None) -> bool:
    return resolve_embedding_settings(config=config, provider=provider).is_cloud


class OpenAICompatibleEmbeddingProvider:
    """Embedding client for providers exposing an OpenAI-compatible API."""

    def __init__(self, settings: EmbeddingProviderSettings):
        if not settings.api_base:
            raise ValueError(
                "EMBEDDING_API_BASE is required for openai_compatible embedding provider."
            )
        if not settings.model:
            raise ValueError("EMBEDDING_MODEL is required for cloud embedding provider.")
        if not settings.api_key:
            raise ValueError(
                f"Missing embedding API key. Set EMBEDDING_API_KEY or {settings.api_key_env}."
            )
        self.settings = settings

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        desc: str = "Generating cloud embeddings",
    ) -> np.ndarray:
        from utils.embedding_progress import report_embedding
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if batch_size < 1 or any(not text.strip() for text in texts):
            raise ValueError('云端嵌入批大小必须为正数，文本不能为空')
        # embedding-3: <=64 inputs and <=3072 tokens per input. Keep all text
        # through conservative UTF-8 windows, then length-weighted pooling.
        bounded = self.settings.cloud_provider == 'zhipu' and self.settings.model == 'embedding-3'
        if bounded:
            batch_size = min(batch_size, 64)
        chunks, owners, weights = [], [], []
        for index, text in enumerate(texts):
            for chunk in split_utf8_text(text, 3070) if bounded else [text]:
                chunks.append(chunk); owners.append(index); weights.append(len(chunk.encode('utf-8')))
        if show_progress:
            print(f'云端嵌入：{len(texts)} 条文本，共 {len(chunks)} 个文本块，预计 {(len(chunks) + batch_size - 1) // batch_size} 次请求（不含重试）', flush=True)
        embeddings: List[np.ndarray] = []
        iterator = range(0, len(chunks), batch_size)
        scope = 'vocabulary' if desc == 'Embedding vocabulary' else 'documents'
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        if show_progress:
            report_embedding('cloud', scope, 0, len(texts), 0, chunks=0, chunk_total=len(chunks), batch_total=total_batches)

        for start in iterator:
            batch_texts = chunks[start:start + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
            end = start + len(batch_texts)
            completed = owners[end - 1] + int(end == len(chunks) or owners[end] != owners[end - 1])
            if show_progress:
                report_embedding('cloud', scope, completed, len(texts), start // batch_size + 1,
                                 chunks=end, chunk_total=len(chunks), batch_total=total_batches)

        matrix = np.zeros((len(texts), len(embeddings[0])), dtype=np.float32)
        total_weights = np.zeros(len(texts), dtype=np.float32)
        for owner, weight, embedding in zip(owners, weights, embeddings):
            matrix[owner] += embedding * weight
            total_weights[owner] += weight
        matrix /= total_weights[:, None]
        if self.settings.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / (norms + 1e-8)
        return matrix

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "input": texts,
        }
        if self.settings.dimensions:
            payload["dimensions"] = self.settings.dimensions

        url = f"{self.settings.api_base}/embeddings"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return self._parse_embeddings(result, len(texts))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Embedding API error {exc.code}: {error_body}")
                if exc.code < 500 and exc.code != 429:
                    raise last_error from exc
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc

            if attempt < self.settings.max_retries:
                print(f'云端嵌入响应未完成，准备第 {attempt + 1}/{self.settings.max_retries} 次重试；失败请求仍计入预算。', flush=True)
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"Cloud embedding request failed: {last_error}") from last_error

    @staticmethod
    def _parse_embeddings(result: Dict[str, Any], expected_count: Optional[int] = None) -> List[np.ndarray]:
        data = result.get("data")
        if not isinstance(data, list):
            raise ValueError(f"Invalid embedding response: missing data list, got keys={list(result.keys())}")
        count = len(data) if expected_count is None else expected_count
        if not data or len(data) != count or sorted(item.get('index', -1) for item in data) != list(range(count)):
            raise ValueError('云端返回的向量数量或索引与输入不一致')

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise ValueError("Invalid embedding response: item missing embedding vector")
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim != 1 or not array.size or not np.isfinite(array).all() or (embeddings and array.shape != embeddings[0].shape):
                raise ValueError('云端返回的向量维度或数值无效')
            embeddings.append(array)
        return embeddings


def create_cloud_embedding_provider(
    config: Any = None,
    provider: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OpenAICompatibleEmbeddingProvider:
    settings = resolve_embedding_settings(
        config=config,
        provider=provider,
        cloud_provider=cloud_provider,
        api_base=api_base,
        api_key_env=api_key_env,
        model=model,
        dimensions=dimensions,
    )
    if not settings.is_cloud:
        raise ValueError("Local embedding provider does not use cloud embedding client.")
    return OpenAICompatibleEmbeddingProvider(settings)

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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm


LOCAL_PROVIDERS = {"local", "qwen"}

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
        return os.environ.get("EMBEDDING_API_KEY") or os.environ.get(self.api_key_env)


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
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, total=(len(texts) + batch_size - 1) // batch_size)

        for start in iterator:
            batch_texts = texts[start:start + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)

        matrix = np.vstack(embeddings).astype(np.float32)
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
                return self._parse_embeddings(result)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Embedding API error {exc.code}: {error_body}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc

            if attempt < self.settings.max_retries:
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"Cloud embedding request failed: {last_error}") from last_error

    @staticmethod
    def _parse_embeddings(result: Dict[str, Any]) -> List[np.ndarray]:
        data = result.get("data")
        if not isinstance(data, list):
            raise ValueError(f"Invalid embedding response: missing data list, got keys={list(result.keys())}")

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise ValueError("Invalid embedding response: item missing embedding vector")
            embeddings.append(np.asarray(vector, dtype=np.float32))
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

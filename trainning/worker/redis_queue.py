from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .config import RedisConfig
from .errors import RetryableJobError
from .protocol import ExecutionSpec


LOGGER = logging.getLogger(__name__)


RENEW_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    values: Mapping[str, str]


@dataclass(frozen=True)
class JobLease:
    key: str
    token: str


class RedisQueue:
    def __init__(self, config: RedisConfig, client: Any | None = None):
        self.config = config
        self.client = client or self._create_client(config)

    @staticmethod
    def _create_client(config: RedisConfig) -> Any:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Redis support requires redis; install worker/requirements.txt") from exc
        common = {
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": max(10, config.block_ms / 1000 + 5),
            "health_check_interval": 30,
        }
        if config.url:
            return redis.Redis.from_url(config.url, **common)
        return redis.Redis(
            host=config.host,
            port=config.port,
            username=config.username or None,
            password=config.password or None,
            db=config.database,
            **common,
        )

    def ping(self) -> None:
        try:
            self.client.ping()
        except Exception as exc:
            raise RetryableJobError(f"connect Redis failed: {exc}") from exc

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(
                name=self.config.job_stream,
                groupname=self.config.consumer_group,
                id="0-0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise RetryableJobError(f"create Redis consumer group failed: {exc}") from exc

    def read(self) -> list[QueueMessage]:
        try:
            response = self.client.xreadgroup(
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_name,
                streams={self.config.job_stream: ">"},
                count=self.config.batch_size,
                block=self.config.block_ms,
            )
        except Exception as exc:
            raise RetryableJobError(f"read Redis job stream failed: {exc}") from exc
        return self._messages_from_response(response)

    def reclaim_pending(self) -> list[QueueMessage]:
        try:
            response = self.client.xautoclaim(
                name=self.config.job_stream,
                groupname=self.config.consumer_group,
                consumername=self.config.consumer_name,
                min_idle_time=self.config.claim_idle_ms,
                start_id="0-0",
                count=self.config.batch_size,
            )
        except Exception as exc:
            # Redis older than 6.2 cannot reclaim via XAUTOCLAIM. New messages still work.
            LOGGER.warning("could not reclaim pending jobs: %s", exc)
            return []
        if not response or len(response) < 2:
            return []
        return [QueueMessage(str(message_id), self._string_map(values)) for message_id, values in response[1]]

    def decode_payload(self, message: QueueMessage) -> dict[str, Any]:
        raw = message.values.get("payload")
        if not raw:
            raise ValueError("Redis message does not contain payload")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Redis payload must be a JSON object")
        return value

    def acquire_lease(self, spec: ExecutionSpec) -> JobLease | None:
        key = self.config.lock_key_prefix + spec.identity
        token = uuid.uuid4().hex
        acquired = self.client.set(key, token, nx=True, ex=self.config.lock_ttl_seconds)
        return JobLease(key, token) if acquired else None

    def renew_lease(self, lease: JobLease) -> bool:
        result = self.client.eval(
            RENEW_LOCK_SCRIPT, 1, lease.key, lease.token, self.config.lock_ttl_seconds
        )
        return bool(result)

    def release_lease(self, lease: JobLease) -> None:
        try:
            self.client.eval(RELEASE_LOCK_SCRIPT, 1, lease.key, lease.token)
        except Exception as exc:
            LOGGER.warning("release job lease %s failed: %s", lease.key, exc)

    def is_cancelled(self, spec: ExecutionSpec) -> bool:
        value = self.client.get(self.config.cancel_key_prefix + str(spec.task_id))
        return value is not None and str(value) == str(spec.attempt)

    def is_done(self, spec: ExecutionSpec) -> bool:
        return bool(self.client.exists(self.config.done_key_prefix + spec.identity))

    def mark_done(self, spec: ExecutionSpec, terminal_type: str) -> None:
        self.client.set(
            self.config.done_key_prefix + spec.identity,
            terminal_type,
            ex=self.config.done_ttl_seconds,
        )

    def publish_event(self, event: Mapping[str, Any]) -> str:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
        values = {
            "event_id": str(event["event_id"]),
            "type": str(event["type"]),
            "task_id": str(event["task_id"]),
            "attempt": str(event["attempt"]),
            "payload": payload,
        }
        return str(
            self.client.xadd(
                self.config.event_stream,
                values,
                maxlen=self.config.stream_max_length,
                approximate=True,
            )
        )

    def dead_letter(self, message: QueueMessage, error: str) -> None:
        values = {
            "source_stream": self.config.job_stream,
            "source_message_id": message.message_id,
            "worker": self.config.consumer_name,
            "error": error[:2048],
            "payload": message.values.get("payload", ""),
        }
        self.client.xadd(
            self.config.dead_letter_stream,
            values,
            maxlen=self.config.stream_max_length,
            approximate=True,
        )

    def acknowledge(self, message: QueueMessage) -> None:
        self.client.xack(
            self.config.job_stream, self.config.consumer_group, message.message_id
        )

    @classmethod
    def _messages_from_response(cls, response: Iterable[Any]) -> list[QueueMessage]:
        messages: list[QueueMessage] = []
        for _stream, entries in response or []:
            for message_id, values in entries:
                messages.append(QueueMessage(str(message_id), cls._string_map(values)))
        return messages

    @staticmethod
    def _string_map(values: Mapping[Any, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in values.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="replace")
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            result[str(key)] = str(value)
        return result
